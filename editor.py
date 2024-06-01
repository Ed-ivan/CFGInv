# 这个文件抽象出各个edit 方法 p2p ,negative-prompts ,mastrol等 
import torch
from P2P.scheduler_dev import DDIMSchedulerDev
from utils.control_utils import EmptyControl, AttentionStore, make_controller
from diffusers import StableDiffusionPipeline
from utils.utils import load_512, latent2image, txt_draw
from typing import Optional, Union, List
from PIL import Image
import numpy as np
from tqdm import tqdm
from P2P import ptp_utils
from P2P.SPDInv import SourcePromptDisentanglementInversion

class Editor:
    def __init__(self, method_list, device,delta_threshold,enable_threshold=True, num_ddim_steps=50,K_round=25,learning_rate=0.001) -> None:
        self.device=device
        self.method_list=method_list
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
            "CompVis/stable-diffusion-v1-4", scheduler=self.scheduler).to(device)
        self.ldm_stable.scheduler.set_timesteps(self.num_ddim_steps)

    #
    def __call__(self, 
                edit_method,
                image_path,
                prompt_src,
                prompt_tar,
                guidance_scale=7.5,
                proximal=None,
                quantile=0.7,
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
        if edit_method=="p2p":
            return self.edit_image_p2p(image_path, prompt_src, prompt_tar, guidance_scale=guidance_scale, num_of_ddim_steps=self.num_ddim_steps,
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller)
        elif edit_method in ["null-text-inversion+p2p", "null-text-inversion+p2p_a800", "null-text-inversion+p2p_3090"]:
            return self.edit_image_null_text_inversion(image_path, prompt_src, prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller)
        elif edit_method=="negative-prompt-inversion+p2p":
            return self.edit_image_negative_prompt_inversion(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar,
                                        guidance_scale=guidance_scale, proximal=None, quantile=quantile, use_reconstruction_guidance=use_reconstruction_guidance,
                                        recon_t=recon_t, recon_lr=recon_lr, cross_replace_steps=cross_replace_steps,
                                        self_replace_steps=self_replace_steps, blend_word=blend_word, eq_params=eq_params,
                                        is_replace_controller=is_replace_controller, use_inversion_guidance=use_inversion_guidance,
                                        dilate_mask=dilate_mask)
        elif edit_method=="directinversion+p2p":
            return self.edit_image_directinversion(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller)
        else:
            raise NotImplementedError(f"No edit method named {edit_method}")




    def edit_image_p2p(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        num_of_ddim_steps,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        generator: Optional[torch.Generator] = None,
        blend_word=None,
        npi_interp=0,
        eq_params=None,
        offsets=(0, 0, 0, 0),
        inference_stage=True,
        is_replace_controller=False,
        num_inference_steps: int = 50,
        **kwargs
        #TODO： 这样写  
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]
        SPD_inversion = SourcePromptDisentanglementInversion(self.ldm_stable, K_round=self.K_round, num_ddim_steps=num_of_ddim_steps,
                                                         learning_rate=self.learning_rate, delta_threshold=self.delta_threshold,
                                                         enable_threshold=self.enable_threshold)
        (image_gt, image_enc, image_enc_latent), x_stars, uncond_embeddings = SPD_inversion.invert(
        image_path, prompt_src, offsets=offsets, npi_interp=npi_interp, verbose=True)

        z_inverted_noise_code = x_stars[-1]
    
        del SPD_inversion

        torch.cuda.empty_cache()

        controller = make_controller(self.ldm_stable, prompts, is_replace_controller, cross_replace_steps, self_replace_steps,
                                 blend_word, eq_params, num_ddim_steps=num_of_ddim_steps)
        #reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]
        #image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")

        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }
        batch_size = len(prompts)
        ptp_utils.register_attention_control(self.ldm_stable, controller)
    # 应该是在 进行了注册 
        height = width = 512

        text_input = self.ldm_stable.tokenizer(
            prompts,
        padding="max_length",
        max_length=self.ldm_stable.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
        )
        text_embeddings = self.ldm_stable.text_encoder(text_input.input_ids.to(self.ldm_stable.device))[0]
        # [2,77.768] 
        max_length = text_input.input_ids.shape[-1]
        if uncond_embeddings is None:
            uncond_input = self.ldm_stable.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
         )
            uncond_embeddings_ = self.ldm_stable.text_encoder(uncond_input.input_ids.to(self.ldm_stable.device))[0]
        else:
            uncond_embeddings_ = None

        latent, latents = ptp_utils.init_latent(z_inverted_noise_code, self.ldm_stable, height, width, generator, batch_size)
        start_time = num_inference_steps
        self.ldm_stable.scheduler.set_timesteps(num_inference_steps)
        with torch.no_grad():
            for i, t in enumerate(tqdm(self.ldm_stable.scheduler.timesteps[-start_time:], total= num_inference_steps)):
                if uncond_embeddings_ is None:
                    context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
                else:
                    context = torch.cat([uncond_embeddings_, text_embeddings])
                latents = ptp_utils.diffusion_step(self.ldm_stable, controller, latents, context, t, guidance_scale,
                                               low_resource=False,
                                               inference_stage=inference_stage, x_stars=x_stars, i=i, **kwargs)
            
            images = ptp_utils.latent2image(self.ldm_stable.vae, latents)
        # you need to change this when you have a high ram 
        #return Image.fromarray(np.concatenate((image_instruct, image_gt, images[0],images[-1]),axis=1))
        return Image.fromarray(np.concatenate((image_gt, images[0],images[-1]),axis=1))
