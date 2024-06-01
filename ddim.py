#this file is for running ddim-inverison 

from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
import torch
from P2P import ptp_utils
from diffusers import DDIMScheduler
from diffusers import DiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
import abc
from P2P import seq_aligner
import shutil
from torch.optim.adam import Adam
from utils.control_utils import load_512,AttentionControl,AttentionStore
from PIL import Image

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


class NullInversion:
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).half() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents
    
    #TODO: not used in this file 
    # def null_optimization(self, latents, num_inner_steps, epsilon):

        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return uncond_embeddings_list
    
    def invert(self, image_path: str, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt)
        # ptp_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Null-text optimization...")

        return (image_gt, image_rec), ddim_latents[-1]
        
    
    def __init__(self, model):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def diffusion_step(model,latents, context, t, guidance_scale, low_resource=False,
                   inference_stage=True, prox="l0", quantile=0.7,
                   image_enc=None, recon_lr=1, recon_t=400,
                   inversion_guidance=True, x_stars=None, i=0, noise_loss=None, **kwargs):
                    # inversion_guidance = False, x_stars = None, i = 0, noise_loss = None, ** kwargs):
    bs = latents.shape[0]
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    step_kwargs = {
        'ref_image': None,
        'recon_lr': 0,
        'recon_mask': None,
    }
    mask_edit = None
    # 应该是 p2p improved 的那篇 
    if inference_stage and prox is not None:
        if prox == 'l1':
            score_delta = noise_prediction_text - noise_pred_uncond
            if quantile > 0:
                threshold = score_delta.abs().quantile(quantile)
            else:
                threshold = -quantile  # if quantile is negative, use it as a fixed threshold
            score_delta -= score_delta.clamp(-threshold, threshold)
            score_delta = torch.where(score_delta > 0, score_delta-threshold, score_delta)
            score_delta = torch.where(score_delta < 0, score_delta+threshold, score_delta)
            if (recon_t > 0 and t < recon_t) or (recon_t < 0 and t > -recon_t):
                step_kwargs['ref_image'] = image_enc
                step_kwargs['recon_lr'] = recon_lr
                mask_edit = (score_delta.abs() > threshold).float()
                radius = 1
                mask_edit = dilate(mask_edit.float(), kernel_size=2*radius+1, padding=radius)
                step_kwargs['recon_mask'] = 1 - mask_edit
        elif prox == 'l0':
            score_delta = noise_prediction_text - noise_pred_uncond
            if quantile > 0:
                threshold = score_delta.abs().quantile(quantile)
            else:
                threshold = -quantile  # if quantile is negative, use it as a fixed threshold
            score_delta -= score_delta.clamp(-threshold, threshold)
            if (recon_t > 0 and t < recon_t) or (recon_t < 0 and t > -recon_t):
                step_kwargs['ref_image'] = image_enc
                step_kwargs['recon_lr'] = recon_lr
                mask_edit = (score_delta.abs() > threshold).float()
                radius = 1
                mask_edit = dilate(mask_edit.float(), kernel_size=2*radius+1, padding=radius)
                step_kwargs['recon_mask'] = 1 - mask_edit
        else:
            raise NotImplementedError
        noise_pred = noise_pred_uncond + guidance_scale * score_delta
    else:
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents, **step_kwargs)["prev_sample"]
    if mask_edit is not None and inversion_guidance and (recon_t > 0 and t < recon_t) or (recon_t < 0 and t > -recon_t):
        recon_mask = 1 - mask_edit
        latents = latents - recon_lr * (latents - x_stars[len(x_stars)-i-2].expand_as(latents)) * recon_mask
    if noise_loss is not None:
        print("add shift")
        latents = torch.concat((latents[:1]+noise_loss[:1],latents[1:]))
    return latents






@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    # controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    height = width = 256
    batch_size = len(prompt)
    # ptp_utils.register_attention_control(model, controller)
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)

    # set timesteps
    extra_set_kwargs = {"offset": 1}
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, latents, context, t, guidance_scale, low_resource)
    image = ptp_utils.latent2image(model.vae, latents)
    return image, latent




if __name__ == "__main__":
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    MY_TOKEN = ''
    LOW_RESOURCE = False 
    NUM_DDIM_STEPS = 25
    GUIDANCE_SCALE = 7.5
    MAX_NUM_WORDS = 77
    output_dir="ddim_res"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ldm_stable = DiffusionPipeline.from_pretrained.from_pretrained("CompVis/ldm-text2im-large-256",torch_dtype=torch.float16).to(device)
    tokenizer = ldm_stable.tokenizer
    null_inversion = NullInversion(ldm_stable)

    image_path = "./images/gnochi_mirror.jpeg"
    prompt = "a cat sitting next to a mirror"
    (image_gt, image_enc), x_t = null_inversion.invert(image_path, prompt, offsets=(0,0,200,0), verbose=True)

    print("Modify or remove offsets according to your image!")
    # controller = AttentionStore()

    images, x_t = text2image_ldm_stable(ldm_stable,prompt, latent=x_t, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE)

    filename = image_path.split('/')[-1].replace(".jpg",".png")
    # 重构以及直接编辑的结果 
    Image.fromarray(np.concatenate(images, axis=1)).save(f"{output_dir}/_ddim_res_{filename}")










