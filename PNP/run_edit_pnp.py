import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionPipeline
import numpy as np
from PIL import Image
import os
import sys
sys.path.append('.')
sys.path.append('..')
import json
import random
import argparse
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from P2P.CFGInv_withloss import CFGInversion
from P2P.scheduler_dev import DDIMSchedulerDev
from typing import Optional, Union, List
from PIL import Image
import numpy as np
from tqdm import tqdm
from P2P import ptp_utils
import torchvision.transforms as T
from pnp_utils import register_attention_control_efficient,register_time,register_conv_control_efficient,get_timesteps

from utils.utils import txt_draw,load_512,latent2image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')

class PNP(nn.Module):
    def __init__(self, model_key,n_timesteps,device="cuda"):
        super().__init__()
        self.device = device

        # Create SD models
        print('Loading SD model')

        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float16).to("cuda")
        pipe.enable_xformers_memory_efficient_attention()

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(n_timesteps, device=self.device)
        self.n_timesteps=n_timesteps
        print('SD model loaded')

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def decode_latent(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img

    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def get_data(self,image_path):
        # load image
        image = Image.open(image_path).convert('RGB') 
        image = image.resize((512, 512), resample=Image.Resampling.LANCZOS)
        image = T.ToTensor()(image).to(self.device)
        return image

    @torch.no_grad()
    def denoise_step(self, x, t,guidance_scale,noisy_latent):
        # register the time step and features in pnp injection modules
        latent_model_input = torch.cat(([noisy_latent]+[x] * 2))

        register_time(self, t.item())

        # compute text embeddings
        text_embed_input = torch.cat([self.pnp_guidance_embeds, self.text_embeds], dim=0)

        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

        # perform guidance
        _,noise_pred_uncond, noise_pred_cond = noise_pred.chunk(3)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # compute the denoising step with the reference model
        denoised_latent = self.scheduler.step(noise_pred, t, x)['prev_sample']
        return denoised_latent

    def init_pnp(self, conv_injection_t, qk_injection_t):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        register_attention_control_efficient(self, self.qk_injection_timesteps)
        register_conv_control_efficient(self, self.conv_injection_timesteps)

    def run_pnp(self,image_path,noisy_latent,target_prompt,guidance_scale=7.5,pnp_f_t=0.8,pnp_attn_t=0.5):
        
        # load image
        self.image = self.get_data(image_path)
        self.eps = noisy_latent[-1]

        self.text_embeds = self.get_text_embeds(target_prompt, "ugly, blurry, black, low res, unrealistic")
        self.pnp_guidance_embeds = self.get_text_embeds("", "").chunk(2)[0]
        
        pnp_f_t = int(self.n_timesteps * pnp_f_t)
        pnp_attn_t = int(self.n_timesteps * pnp_attn_t)
        self.init_pnp(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t)
        edited_img = self.sample_loop(self.eps,guidance_scale,noisy_latent)
        
        return edited_img

    def sample_loop(self, x,guidance_scale,noisy_latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(self.scheduler.timesteps, desc="Sampling"):
                x = self.denoise_step(x, t,guidance_scale,noisy_latent[-1-i])

            decoded_latent = self.decode_latent(x)
                
        return decoded_latent

def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class Preprocess(nn.Module):
    def __init__(self, device,model_key):
        super().__init__()

        self.device = device
        self.use_depth = False

        print(f'[INFO] loading stable diffusion...')
        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae", 
                                                 torch_dtype=torch.float16).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder", revision="fp16",
                                                          torch_dtype=torch.float16).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet", revision="fp16",
                                                         torch_dtype=torch.float16).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        print(f'[INFO] loaded stable diffusion!')


    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(device))[0]
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def decode_latents(self, latents):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latents = 1 / 0.18215 * latents
            imgs = self.vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def load_img(self, image_path):
        image_pil = T.Resize(512)(Image.open(image_path).convert("RGB"))
        image = T.ToTensor()(image_pil).unsqueeze(0).to(device)
        return image

    @torch.no_grad()
    def encode_imgs(self, imgs):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            imgs = 2 * imgs - 1
            posterior = self.vae.encode(imgs).latent_dist
            latents = posterior.mean * 0.18215
        return latents

    @torch.no_grad()
    def ddim_inversion(self, cond, latent):
        latent_list=[latent]
        timesteps = reversed(self.scheduler.timesteps)
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(timesteps):
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                eps = self.unet(latent, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (latent - sigma_prev * eps) / mu_prev
                latent = mu * pred_x0 + sigma * eps
                latent_list.append(latent)
        return latent_list

    @torch.no_grad()
    def ddim_sample(self, x, cond):
        timesteps = self.scheduler.timesteps
        latent_list=[]
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(timesteps):
                    cond_batch = cond.repeat(x.shape[0], 1, 1)
                    alpha_prod_t = self.scheduler.alphas_cumprod[t]
                    alpha_prod_t_prev = (
                        self.scheduler.alphas_cumprod[timesteps[i + 1]]
                        if i < len(timesteps) - 1
                        else self.scheduler.final_alpha_cumprod
                    )
                    mu = alpha_prod_t ** 0.5
                    sigma = (1 - alpha_prod_t) ** 0.5
                    mu_prev = alpha_prod_t_prev ** 0.5
                    sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                    eps = self.unet(x, t, encoder_hidden_states=cond_batch).sample

                    pred_x0 = (x - sigma * eps) / mu
                    x = mu_prev * pred_x0 + sigma_prev * eps
                    latent_list.append(x)
        return latent_list

    @torch.no_grad()
    def extract_latents(self, num_steps, data_path,
                        inversion_prompt=''):
        self.scheduler.set_timesteps(num_steps)

        cond = self.get_text_embeds(inversion_prompt, "")[1].unsqueeze(0)
        image = self.load_img(data_path)
        latent = self.encode_imgs(image)

        inverted_x = self.ddim_inversion(cond, latent)
        latent_reconstruction = self.ddim_sample(inverted_x[-1], cond)
        rgb_reconstruction = self.decode_latents(latent_reconstruction[-1])
        latent_reconstruction.reverse()
        return inverted_x, rgb_reconstruction, latent_reconstruction

def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)

#TODO: 但是这个地方写的好别捏啊,要不重构一下，但是 代价有点大啊 ， fuck    
NUM_DDIM_STEPS = 50
model_key = "runwayml/stable-diffusion-v1-5"
toy_scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
toy_scheduler.set_timesteps(NUM_DDIM_STEPS)

timesteps_to_save, num_inference_steps = get_timesteps(toy_scheduler, num_inference_steps=NUM_DDIM_STEPS,
                                                           strength=1.0,
                                                           device=device)
model = Preprocess(device, model_key=model_key)
pnp = PNP(model_key,NUM_DDIM_STEPS)
scheduler = DDIMSchedulerDev(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                 set_alpha_to_one=False)
# 这个我是最愁的一个 ，还不知道能不能跑呢
ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler).to(
        device)

@torch.no_grad()
def editing_p2p(
        model,
        prompt: List[str],
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        uncond_embeddings=None,
        return_type='image',
        inference_stage=True,
        x_stars=None,
        **kwargs,
):
    batch_size = len(prompt)
    height = width = 512

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    start_time = num_inference_steps
    model.scheduler.set_timesteps(num_inference_steps)
    with torch.no_grad():
        for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:], total=num_inference_steps)):
            if uncond_embeddings_ is None:
                context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
            else:
                context = torch.cat([uncond_embeddings_, text_embeddings])
            latents = ptp_utils.diffusion_step(model, None, latents, context, t, guidance_scale,
                                               low_resource=False,
                                               inference_stage=inference_stage, x_stars=x_stars, i=i, **kwargs)
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent

def get_PNP_input(
        image_path,
        prompt_src,
        device,
        guidance_scale=7.5,
        npi_interp=0,
        offsets=(0, 0, 0, 0),
        K_round=25,
        num_of_ddim_steps=50,
        learning_rate=0.001,
        delta_threshold=5e-6,
        enable_threshold=True):
        
        SPD_inversion =CFGInversion(ldm_stable, K_round=K_round, num_ddim_steps=num_of_ddim_steps,
                                                         learning_rate=learning_rate, delta_threshold=delta_threshold,
                                                         enable_threhold=enable_threshold)
        (image_gt, image_enc, image_enc_latent), x_stars, uncond_embeddings = SPD_inversion.invert(
        image_path, prompt_src, offsets=offsets, npi_interp=npi_interp, verbose=True)

        z_inverted_noise_code = x_stars[-1]
        del SPD_inversion
        torch.cuda.empty_cache()
        prompts = [prompt_src]
        rgb_reconstruction, latent_reconstruction = editing_p2p(ldm_stable, prompts, latent=z_inverted_noise_code,
                            num_inference_steps=num_of_ddim_steps,
                            guidance_scale=guidance_scale,
                            uncond_embeddings=uncond_embeddings, x_stars=x_stars, )
        ref_latent = image_enc_latent
        return z_inverted_noise_code, rgb_reconstruction, latent_reconstruction, ref_latent

#NOTE: ddim+pnp 
def edit_image_ddim_PnP(
    image_path,
    prompt_src,
    prompt_tar,
    guidance_scale=7.5,
    image_shape=[512,512]
):
    torch.cuda.empty_cache()
    image_gt = load_512(image_path)
    _, rgb_reconstruction, latent_reconstruction = model.extract_latents(data_path=image_path,
                                         num_steps=NUM_DDIM_STEPS,
                                         inversion_prompt=prompt_src)
    
    edited_image=pnp.run_pnp(image_path,latent_reconstruction,prompt_tar,guidance_scale)
    
    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

    return Image.fromarray(np.concatenate((
        image_instruct,
        image_gt,
        np.uint8(255*np.array(rgb_reconstruction[0].permute(1,2,0).cpu().detach())),
        np.uint8(255*np.array(edited_image[0].permute(1,2,0).cpu().detach())),
        ),1))

# NOTE:snp+pnp 
def edit_image_pnp(image_path,prompt_src, prompt_tar,guidance_scale=7.5):
    
    torch.cuda.empty_cache()
    image_gt = load_512(image_path)
    _, rgb_reconstruction, latent_reconstruction,ref_latent = get_PNP_input(image_path=image_path,prompt_src=prompt_src)
    
    # 
    
    edited_image=pnp.run_pnp(image_path,latent_reconstruction,prompt_tar,guidance_scale)
    
    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
    #这里面需要修改一下啊 , 
    return Image.fromarray(np.concatenate((
        image_instruct,
        image_gt,
        np.uint8(255*np.array(rgb_reconstruction[0].permute(1,2,0).cpu().detach())),
        np.uint8(255*np.array(edited_image[0].permute(1,2,0).cpu().detach())),
        ),1))


#NOTE: 这个暂时先不管了
def edit_image_directinversion_PnP(
    image_path,
    prompt_src,
    prompt_tar,
    guidance_scale=7.5,
    image_shape=[512,512]
):
    torch.cuda.empty_cache()
    image_gt = load_512(image_path)
    inverted_x, rgb_reconstruction, _ = model.extract_latents(data_path=image_path,
                                         num_steps=NUM_DDIM_STEPS,
                                         inversion_prompt=prompt_src)

    edited_image=pnp.run_pnp(image_path,inverted_x,prompt_tar,guidance_scale)
    
    image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

    return Image.fromarray(np.concatenate((
        image_instruct,
        image_gt,
        np.uint8(np.array(latent2image(model=pnp.vae, latents=inverted_x[1].to(pnp.vae.dtype))[0])),
        np.uint8(255*np.array(edited_image[0].permute(1,2,0).cpu().detach())),
        ),1))


def mask_decode(encoded_mask,image_shape=[512,512]):
    length=image_shape[0]*image_shape[1]
    mask_array=np.zeros((length,))
    
    for i in range(0,len(encoded_mask),2):
        splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j]=1
            
    mask_array=mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0,:]=1
    mask_array[-1,:]=1
    mask_array[:,0]=1
    mask_array[:,-1]=1
            
    return mask_array



image_save_paths={
    "ddim+pnp":"ddim+pnp",
    "directinversion+pnp":"directinversion+pnp",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action= "store_true") # rerun existing images
    parser.add_argument('--data_path', type=str, default="data") # the editing category that needed to run
    parser.add_argument('--output_path', type=str, default="output") # the editing category that needed to run
    parser.add_argument('--edit_category_list', nargs = '+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) # the editing category that needed to run
    parser.add_argument('--edit_method_list', nargs = '+', type=str, default=["ddim+pnp","directinversion+pnp"]) # the editing methods that needed to run
    args = parser.parse_args()
    rerun_exist_images=args.rerun_exist_images
    data_path=args.data_path
    output_path=args.output_path
    edit_category_list=args.edit_category_list
    edit_method_list=args.edit_method_list
    with open(f"{data_path}/mapping_file.json", "r") as f:
        editing_instruction = json.load(f)
    for key, item in editing_instruction.items():
        
        if item["editing_type_id"] not in edit_category_list:
            continue        
        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        image_path = os.path.join(f"{data_path}/annotation_images", item["image_path"])
        editing_instruction = item["editing_instruction"]
        blended_word = item["blended_word"].split(" ") if item["blended_word"] != "" else []
        mask = Image.fromarray(np.uint8(mask_decode(item["mask"])[:,:,np.newaxis].repeat(3,2))).convert("L")

        for edit_method in edit_method_list:
            present_image_save_path=image_path.replace(data_path, os.path.join(output_path,image_save_paths[edit_method]))
            if ((not os.path.exists(present_image_save_path)) or rerun_exist_images):
                print(f"editing image [{image_path}] with [{edit_method}]")
                setup_seed()
                torch.cuda.empty_cache()
                if edit_method=="ddim+pnp":
                    edited_image = edit_image_ddim_PnP(
                        image_path=image_path,
                        prompt_src=original_prompt,
                        prompt_tar=editing_prompt,
                        guidance_scale=7.5,
                    )
                elif edit_method=="directinversion+pnp":
                    edited_image = edit_image_directinversion_PnP(
                        image_path=image_path,
                        prompt_src=original_prompt,
                        prompt_tar=editing_prompt,
                        guidance_scale=7.5,
                    )
                elif edit_method=="snp+pnp":pass
                elif edit_method=="prior+pnp":pass
                elif edit_method=="null+pnp":pass
                #TODO: 
                else:
                    raise NotImplementedError(f"No edit method named {edit_method}")
                if not os.path.exists(os.path.dirname(present_image_save_path)):
                    os.makedirs(os.path.dirname(present_image_save_path))
                edited_image.save(present_image_save_path)
                
                print(f"finish")
                
            else:
                print(f"skip image [{image_path}] with [{edit_method}]")
        
        
        
