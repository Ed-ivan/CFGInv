# 实现 spn的 inversion 
from typing import Optional, Union, List
from tqdm import tqdm
import torch
import json
from diffusers import StableDiffusionPipeline
import numpy as np

from editor import Editor
from P2P import ptp_utils
from PIL import Image
import os
from P2P.scheduler_dev import DDIMSchedulerDev
import argparse

import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
from utils.control_utils import load_512, make_controller
#from P2P.SPDInv import SourcePromptDisentanglementInversion
from P2P.CFGInv_withloss import CFGInversion 
# this file is to run rescontruction results 

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

@torch.no_grad()
def recontruction(
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
    # ptp_utils.register_attention_control(model, controller)
    # 应该是在 进行了注册 
    height = width = 512

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    # [2,77,768] 
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
    controller=None
    with torch.no_grad():
        for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:], total=num_inference_steps)):
            if uncond_embeddings_ is None:
                context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
            else:
                context = torch.cat([uncond_embeddings_, text_embeddings])
            latents = ptp_utils.diffusion_step(model,controller,latents, context, t, guidance_scale,
                                               low_resource=False,
                                               inference_stage=inference_stage, x_stars=x_stars, i=i, **kwargs)
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent
@torch.no_grad()
def P2P_inversion_and_recontruction(
        image_path,
        prompt_src,
        prompt_tar,
        output_dir='output',
        guidance_scale=7.5,
        npi_interp=0,
        cross_replace_steps=0.8,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        offsets=(0, 0, 0, 0),
        is_replace_controller=False,
        use_inversion_guidance=True,
        K_round=25,
        num_of_ddim_steps=50,
        learning_rate=0.001,
        delta_threshold=5e-6,
        enable_threshold=True,
        **kwargs
):
    os.makedirs(output_dir, exist_ok=True)
    sample_count = len(os.listdir(output_dir))

    scheduler = DDIMSchedulerDev(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                 set_alpha_to_one=False)
    ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler).to(
        device)
    
    SPD_inversion = CFGInversion(ldm_stable, K_round=K_round, num_ddim_steps=num_of_ddim_steps,
                                                         learning_rate=learning_rate, delta_threshold=delta_threshold,
                                                         enable_threshold=enable_threshold)
    (image_gt, image_enc, image_enc_latent), x_stars, uncond_embeddings = SPD_inversion.invert(
        image_path, prompt_src, offsets=offsets, npi_interp=npi_interp, verbose=True)

    z_inverted_noise_code = x_stars[-1]
    
    del SPD_inversion

    torch.cuda.empty_cache()

    ########## edit ##########
    prompts = [prompt_src, prompt_tar]

    images, _ = recontruction(ldm_stable, prompts,latent=z_inverted_noise_code,
                            num_inference_steps=num_of_ddim_steps,
                            #TODO: 记得修改一下 
                            #guidance_scale=1,
                            guidance_scale=guidance_scale,
                            uncond_embeddings=uncond_embeddings,
                            inversion_guidance=use_inversion_guidance, x_stars=x_stars, )

    filename = image_path.split('/')[-1].replace(".jpg",".png")
    Image.fromarray(np.concatenate(images, axis=1)).save(f"{output_dir}/{sample_count}_P2P_{filename}")

def parse_args():
    parser = argparse.ArgumentParser(description="Input your dataset path")
    parser.add_argument('--data_path', type=str, default="ple_images/") # the editing category that needed to run
    parser.add_argument('--edit_category_list', nargs = '+', type=str, default=["0","1","2","3","4","5","6","7","8","9"]) # the editing category that needed to run
    parser.add_argument(
        "--K_round", 
        type=int,
        default=10,
        help="Optimization Round",
    )
    parser.add_argument(
        "--num_of_ddim_steps",
        type=int,
        default=50,
        help="Blended word needed for P2P",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--delta_threshold",
        type=float,
        default=5e-6,
        help="Delta threshold",
    )
    parser.add_argument(
        "--enable_threshold",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--eq_params",
        type=float,
        default=2.,
        help="Eq parameter weight for P2P",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_loss_4e-6",
        help="Save editing results",
    )
    args = parser.parse_args()
    return args

# 里面的具体编辑还得看下人家 direactinversion 的  
if __name__ == "__main__":
    args = parse_args()
    params = {}
    params['guidance_scale'] = args.guidance_scale
    params['K_round'] = args.K_round
    params['num_of_ddim_steps'] = args.num_of_ddim_steps
    params['learning_rate'] = args.learning_rate
    params['enable_threshold'] = args.enable_threshold
    params['delta_threshold'] = args.delta_threshold
    params['output_dir'] = args.output
    params['data_path'] = args.data_path
    data_path=args.data_path
    output_path=args.output
    edit_method="p2p"
    params['edit_method'] =edit_method

    edit_category_list=args.edit_category_list
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    editor=Editor(edit_method, device,delta_threshold=args.delta_threshold,num_ddim_steps=args.num_of_ddim_steps,
                  K_round=args.K_round,learning_rate=args.learning_rate)
    # self, method_list, device,delta_threshold,enable_threshold=True, num_ddim_steps=50,K_round=25,learning_rate=0.001

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

        present_image_save_path=image_path.replace(data_path, os.path.join(output_path))
        if (not os.path.exists(present_image_save_path)):
                print(f"editing image [{image_path}] with [p2p]")
                #setup_seed()
                torch.cuda.empty_cache() 
                #TODO: 里面的模型直接写死了就是 "CompVis/stable-diffusion-v1-4"
                #然后 eq_ 的这里写一下 
                edited_image = editor(edit_method,
                                         image_path=image_path,
                                        prompt_src=original_prompt,
                                        prompt_tar=editing_prompt,
                                        guidance_scale=7.5,
                                        cross_replace_steps=0.4,
                                        self_replace_steps=0.6,
                                        blend_word=(((blended_word[0], ),
                                                    (blended_word[1], ))) if len(blended_word) else None,
                                        eq_params={
                                            "words": (blended_word[1], ),
                                            "values": (2, )
                                        } if len(blended_word) else None,
                                        proximal="l0",
                                        quantile=0.75,
                                        use_inversion_guidance=True,
                                        recon_lr=1,
                                        recon_t=400,
                                        )
                if not os.path.exists(os.path.dirname(present_image_save_path)):
                    os.makedirs(os.path.dirname(present_image_save_path))
                #是不是,要按照它的写法才行啊
                edited_image.save(present_image_save_path)

                print(f"finish")
                
    
    





