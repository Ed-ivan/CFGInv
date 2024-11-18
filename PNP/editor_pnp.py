from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionPipeline
from P2P.CFGInv_withloss import CFGInversion
from P2P.scheduler_dev import DDIMSchedulerDev
import torch
import torch.nn as nn
from typing import Optional, Union, List
from PIL import Image
import numpy as np
from tqdm import tqdm
from P2P import ptp_utils
# 编辑 pnp 的方法 




class Editor:
    def __init__(self, method_list, device,delta_threshold,enable_threshold=True, num_ddim_steps=50,K_round=25,learning_rate=0.001) -> None:
        self.device=device
        self.num_ddim_steps=num_ddim_steps
        self.K_round=K_round
        self.learning_rate=learning_rate
        self.delta_threshold=delta_threshold
        self.enable_threshold=enable_threshold
        # init model
        self.scheduler = DDIMSchedulerDev(beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule="scaled_linear",
                                    clip_sample=False,
                                    set_alpha_to_one=False)
        self.ldm_stable = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", scheduler=self.scheduler).to(self.device)
        self.ldm_stable.scheduler.set_timesteps(self.num_ddim_steps)

        
        
    def __call__(self, 
                edit_method,
                image_path,
                prompt_src,
                prompt_tar,
                guidance_scale=7.5,
                proximal=None,
                quantile=0.7,
                prox:str = None, 
                use_reconstruction_guidance=False,
                recon_t=400,
                recon_lr=0.1,
                cross_replace_steps=0.4,
                self_replace_steps=0.6,
                blend_word=None,
                eq_params=None,
                is_replace_controller=False,
                use_inversion_guidance=False,
                dilate_mask=1,**kwargs):
        if edit_method=="snp+pnp":
            return self.edit_image_p2p(image_path, prompt_src, prompt_tar, guidance_scale=guidance_scale, num_of_ddim_steps=self.num_ddim_steps,
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller)
        
        else:
            raise NotImplementedError(f"No edit method named {edit_method}")
    #TODO:  这里肯定 不使用 proximal 方式来做的    
    
    @torch.no_grad()
    def editing_p2p(self,
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
        **kwargs,):
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
    
    

