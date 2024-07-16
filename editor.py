# 这个文件抽象出各个edit 方法 p2p ,negative-prompts ,mastrol等 
import torch
from P2P.scheduler_dev import DDIMSchedulerDev
from utils.control_utils import EmptyControl, AttentionStore, make_controller
from diffusers import StableDiffusionPipeline
from utils.utils import load_512, latent2image, txt_draw,load_image
from typing import Optional, Union, List
from PIL import Image
import numpy as np
from tqdm import tqdm
from P2P import ptp_utils
from P2P.CFGInv_withloss import CFGInversion
from P2P.inversion import NullInversion,NegativePromptInversion,DirectInversion
from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import regiter_attention_editor_diffusers
from masactrl.masactrl import MutualSelfAttentionControl           

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
        # self.ldm_stable = StableDiffusionPipeline.from_pretrained(
        #     "CompVis/stable-diffusion-v1-4", scheduler=self.scheduler).to(self.device)
        
        #NOTE: when you run masactrl, just Comment out the line  
        self.ldm_stable = MasaCtrlPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=self.scheduler, cross_attention_kwargs={"scale": 0.5}).to(self.device)
        self.ldm_stable.scheduler.set_timesteps(self.num_ddim_steps)
        
    def direct_inversion_p2p_guidance_diffusion_step(self,model, controller, latents, context, t, guidance_scale, noise_loss, low_resource=False,add_offset=True):
        if low_resource:
            noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
            noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
        else:
            latents_input = torch.cat([latents] * 2)
            noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
            noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
        if add_offset:
            latents = torch.concat((latents[:1]+noise_loss[:1],latents[1:]))
        latents = controller.step_callback(latents)
        return latents

    @torch.no_grad()
    def direct_inversion_p2p_guidance_forward(self,
        prompts,
        controller,
        latent=None,
        num_inference_steps: int = 50,
        guidance_scale = 7.5,
        generator = None,
        noise_loss_list = None,
        add_offset=True
    ):
        batch_size = len(prompts)
        ptp_utils.register_attention_control(self.ldm_stable, controller)
        height = width = 512
    
        text_input = self.ldm_stable.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.ldm_stable.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.ldm_stable.text_encoder(text_input.input_ids.to(self.ldm_stable.device))[0]
        max_length = text_input.input_ids.shape[-1]
    
        uncond_input =self.ldm_stable.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings =self.ldm_stable.text_encoder(uncond_input.input_ids.to(self.ldm_stable.device))[0]

        latent, latents = ptp_utils.init_latent(latent, self.ldm_stable, height, width, generator, batch_size)
        self.ldm_stable.scheduler.set_timesteps(num_inference_steps)
        for i, t in enumerate(self.ldm_stable.scheduler.timesteps):
        
            context = torch.cat([uncond_embeddings, text_embeddings])
            latents = self.direct_inversion_p2p_guidance_diffusion_step(self.ldm_stable, controller, latents, context, t, guidance_scale, noise_loss_list[i],low_resource=False,add_offset=add_offset)
        
        return latents, latent

    @torch.no_grad()
    def  p2p_guidance_forward(self,prompts,controller,num_inference_steps:int =50,
        guidance_scale = 7.5, generator=None,return_latents=False,z_inverted_noise_code=None,uncond_embeddings=None,inference_stage=True,x_stars=None,**kwargs):

        '''
                    model, controller, latents, context, t, guidance_scale, low_resource=False,
                   inference_stage=True, prox=None, quantile=0.7,
                   image_enc=None, recon_lr=1, recon_t=400,
                   inversion_guidance=True, x_stars=None, i=0, noise_loss=None
        '''
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
                # TODO: 如果想要做修改的话， 肯定是在这里面了，但是不对啊 别的方法也会用到这个函数 ，应该怎么办？？
                if uncond_embeddings_ is None:
                    context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
                else:
                    context = torch.cat([uncond_embeddings_, text_embeddings])
                latents = ptp_utils.diffusion_step(self.ldm_stable, controller, latents, context, t, guidance_scale,
                                               low_resource=False,
                                               inference_stage=inference_stage, x_stars=x_stars, i=i, **kwargs)
                        
            images = ptp_utils.latent2image(self.ldm_stable.vae, latents)
        if return_latents:
            return latents,latent
        return images[0],images[-1]
        #return Image.fromarray(np.concatenate((image_gt, images[0],images[-1]),axis=1))

    @torch.no_grad()
    def prior_guidance_forward(self,prompts,controller,num_inference_steps:int=50,guidance_scale = 7.5, generator=None,return_latents=False,z_inverted_noise_code=None,uncond_embeddings=None,inference_stage=True,x_stars=None,**kwargs):
        #TODO: 完成这个函数写的优雅一点 
        '''
        dynamic guidance forward 动态的引导  当  tau1<t<=tau2 时候 guidance_scale  位7.5 其他 1 
        '''
        def linear_schedule(t, guidance_scale, tau1=0.4, tau2=0.8):
            t = t / self.num_ddim_steps
            if t <= tau1:
                return guidance_scale
            if t >= tau2:
                return 1.0
            gamma = (tau2 - t) / (tau2 - tau1) * (guidance_scale - 1.0) + 1.0

            return gamma

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
                # TODO: 如果想要做修改的话， 肯定是在这里面了，但是不对啊 别的方法也会用到这个函数 ，应该怎么办？？
                if uncond_embeddings_ is None:
                    context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
                else:
                    context = torch.cat([uncond_embeddings_, text_embeddings])
                guidance_scale =linear_schedule(t,guidance_scale)
                latents = ptp_utils.diffusion_step(self.ldm_stable, controller, latents, context, t, guidance_scale,
                                               low_resource=False,
                                               inference_stage=inference_stage, x_stars=x_stars, i=i, **kwargs)
                        
            images = ptp_utils.latent2image(self.ldm_stable.vae, latents)
        if return_latents:
            return latents,latent
        return images[0],images[-1]



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
        elif edit_method in ["snp+masactrl","prior_masactrl"]:
            return self.edit_image_masactrl(image_path=image_path,prompt_src=prompt_src,prompt_tar=prompt_tar,guidance_scale=guidance_scale,num_of_ddim_steps=self.num_ddim_steps)
        elif edit_method =="ddim_masactrl":
            return self.edit_image_ddim_masactrl(image_path=image_path,prompt_src=prompt_src,prompt_tar=prompt_tar,guidance_scale=guidance_scale)
        elif edit_method =="directinversion+masactrl":
            return self.edit_image_directinversion_masactrl(image_path=image_path,prompt_src=prompt_src,prompt_tar=prompt_tar,guidance_scale=guidance_scale)
        elif edit_method=="directinversion+p2p":
            return self.edit_image_directinversion_p2p(image_path=image_path, prompt_src=prompt_src, prompt_tar=prompt_tar, guidance_scale=guidance_scale, 
                                        cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, 
                                        blend_word=blend_word, eq_params=eq_params, is_replace_controller=is_replace_controller)
        else:
            raise NotImplementedError(f"No edit method named {edit_method}")
    #TODO:  这里肯定 不使用 proximal 方式来做的
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
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]
        SPD_inversion = CFGInversion(self.ldm_stable, K_round=self.K_round, num_ddim_steps=num_of_ddim_steps,
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
        image1, image2 = self.p2p_guidance_forward(prompts=prompts,controller=controller,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,generator=generator,z_inverted_noise_code=z_inverted_noise_code,
         uncond_embeddings=uncond_embeddings,inference_stage=inference_stage,x_stars=x_stars,**kwargs)
        return Image.fromarray(np.concatenate((image_gt, image1,image2),axis=1))
    # 编辑null text inversion 的方法 
    def edit_image_null_text_inversion(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
        **kwargs
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = NullInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        # 应该是经过优化的吧 
        _, _, x_stars, uncond_embeddings = null_inversion.invert(
            image_gt=image_gt, prompt=prompt_src,guidance_scale=guidance_scale)
        x_t = x_stars[-1]

        controller = AttentionStore()
        
        reconstruct_latent, x_t = self.p2p_guidance_forward( 
                                       prompts=[prompt_src], 
                                       controller=controller, 
                                       z_inverted_noise_code=x_t, 
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None, 
                                       uncond_embeddings=uncond_embeddings,
                                       return_latents=True)

        image_rec=latent2image(self.ldm_stable.vae,latents=reconstruct_latent)[0]

        #image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }
        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps)
        #应该没有错误啊 为啥还是感觉这么有问题?? 
        
        image1, image2 = self.p2p_guidance_forward(
                                       prompts=prompts, 
                                       controller=controller, 
                                       z_inverted_noise_code=x_t, 
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None, 
                                       uncond_embeddings=uncond_embeddings)
        return Image.fromarray(np.concatenate((image_gt,image_rec,image2),axis=1))   
    # npi 的方法
    def edit_image_negative_prompt_inversion(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        generator: Optional[torch.Generator] = None,
        proximal=None,
        quantile=0.7,
        use_reconstruction_guidance=False,
        recon_t=400,
        recon_lr=0.1,
        npi_interp=0,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
        use_inversion_guidance=False,
        dilate_mask=1,
        inference_stage=True,
        num_inference_steps: int = 50,
        **kwargs
    ):
        
        self.ldm_stable = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", scheduler=self.scheduler).to(self.device)
        self.ldm_stable.scheduler.set_timesteps(self.num_ddim_steps)
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]
        null_inversion = NegativePromptInversion(model=self.ldm_stable,
                                                num_ddim_steps=self.num_ddim_steps)
        _, image_enc_latent, x_stars, uncond_embeddings = null_inversion.invert(
            image_gt=image_gt, prompt=prompt_src, npi_interp=npi_interp)
        x_t = x_stars[-1]

        del null_inversion

        torch.cuda.empty_cache()

        controller = make_controller(self.ldm_stable, prompts, is_replace_controller, cross_replace_steps, self_replace_steps,
                                 blend_word, eq_params, num_ddim_steps=self.num_of_ddim_steps)
        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }
        image1, image2 = self.p2p_guidance_forward(prompts=prompts,controller=controller,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,
        generator=generator,z_inverted_noise_code=x_t,
         uncond_embeddings=uncond_embeddings,inference_stage=inference_stage,x_stars=x_stars,prox=proximal,
         recon_t=recon_t,recon_lr=recon_lr,quantile=quantile,**kwargs)
        return Image.fromarray(np.concatenate((image_gt, image1,image2),axis=1))



        # 应该还是使用 P2p  forward, 因为其实这两种的实质相差的不多,感觉 

    def edit_image_directinversion_p2p(
        self,
        image_path,
        prompt_src,
        prompt_tar,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
        blend_word=None,
        eq_params=None,
        is_replace_controller=False,
    ):
        image_gt = load_512(image_path)
        prompts = [prompt_src, prompt_tar]

        null_inversion = DirectInversion(model=self.ldm_stable,
                                    num_ddim_steps=self.num_ddim_steps)
        _, _, x_stars, noise_loss_list = null_inversion.invert(
            image_gt=image_gt, prompt=prompts,guidance_scale=guidance_scale)
        x_t = x_stars[-1]

        controller = AttentionStore()
        
        reconstruct_latent, x_t =self.direct_inversion_p2p_guidance_forward(
                                       prompts=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale,
                                       generator=None)
    
        
        reconstruct_image = latent2image(model=self.ldm_stable.vae, latents=reconstruct_latent)[0]

        ########## edit ##########
        cross_replace_steps = {
            'default_': cross_replace_steps,
        }
        controller = make_controller(pipeline=self.ldm_stable,
                                    prompts=prompts,
                                    is_replace_controller=is_replace_controller,
                                    cross_replace_steps=cross_replace_steps,
                                    self_replace_steps=self_replace_steps,
                                    blend_words=blend_word,
                                    equilizer_params=eq_params,
                                    num_ddim_steps=self.num_ddim_steps)
        
        latents, _ = self.direct_inversion_p2p_guidance_forward(
                                       prompts=prompts, 
                                       controller=controller, 
                                       noise_loss_list=noise_loss_list, 
                                       latent=x_t,
                                       num_inference_steps=self.num_ddim_steps, 
                                       guidance_scale=guidance_scale, 
                                       generator=None)

        images = latent2image(model=self.ldm_stable.vae, latents=latents)

        
        #image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
        return Image.fromarray(np.concatenate((image_gt, reconstruct_image,images[-1]),axis=1))




    def edit_image_directinversion_masactrl(self,image_path,prompt_src,prompt_tar,guidance_scale,step=4,layper=10):
            source_image=load_image(image_path, self.device)
            image_gt = load_512(image_path)
            prompts=["", prompt_tar]
            from masactrl.masactrl_utils import AttentionBase
            null_inversion = DirectInversion(model=self.ldm_stable,
                                                num_ddim_steps=self.num_ddim_steps)
        
            _, image_enc_latent, x_stars, noise_loss_list = null_inversion.invert(
            image_gt=image_gt, prompt=prompts, guidance_scale=guidance_scale)
            x_t = x_stars[-1]
        
        # results of direct synthesis
            editor = AttentionBase()
            regiter_attention_editor_diffusers(self.ldm_stable, editor)
            image_fixed = self.ldm_stable([prompt_tar],
                            latents=x_t,
                            num_inference_steps=self.num_ddim_steps,
                            guidance_scale=guidance_scale,
                            noise_loss_list=None)
        
            # hijack the attention module
            editor = MutualSelfAttentionControl(step, layper)
            regiter_attention_editor_diffusers(self.ldm_stable, editor)

            # inference the synthesized image
            image_masactrl = self.ldm_stable(prompts,
                            latents= x_t.expand(len(prompts), -1, -1, -1),
                            guidance_scale=guidance_scale,
                            noise_loss_list=noise_loss_list)
        
            #image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        
            out_image=np.concatenate((image_gt,image_masactrl[0],image_masactrl[-1]),1)
        
            return Image.fromarray(out_image)
            #  算了 比一下吧 ，毕竟是ICLR 2024  
            # CUDA_VISIBLE_DEVICES=0 python  run_edit.py --edit_methods="null-text-inversion+p2p"


    #TODO:  因为其实并不打算使用什么关于 npi的东西，所以  prox,npi
    # 的参数 直接 不传入了 
    def edit_image_masactrl(self,
        image_path,
        prompt_src,
        prompt_tar,
        num_of_ddim_steps,
        guidance_scale=7.5,
        masa_step: int = 4,
        masa_layer: int = 10,
        inject_uncond: str = "src",
        inject_cond: str = "src",
        generator: Optional[torch.Generator] = None,
        prox_step: int = 0,
        prox: str = None,
        quantile: float = 0.6,
        npi: bool = False,
        npi_interp: float = 0,
        npi_step: int = 0,
        offsets=(0, 0, 0, 0),
        inference_stage=True,
        num_inference_steps: int = 50,
        **kwargs):
        # 需要参考 masactrl 的方法 
        #TODO: 我觉得可以把 dynamic 的 尝试融进去
        # 
            prompts = [prompt_src, prompt_tar]
            SPD_inversion = CFGInversion(self.ldm_stable, K_round=self.K_round, num_ddim_steps=num_of_ddim_steps,
                                                         learning_rate=self.learning_rate, delta_threshold=self.delta_threshold,
                                                         enable_threshold=self.enable_threshold)
            (image_gt, image_enc, image_enc_latent), x_stars, uncond_embeddings = SPD_inversion.invert(
            image_path, prompt_src, offsets=offsets, npi_interp=npi_interp, verbose=True)
            z_inverted_noise_code = x_stars[-1]
            z_inverted_noise_code =z_inverted_noise_code.expand(len(prompts),-1,-1,-1);
            del SPD_inversion
            torch.cuda.empty_cache()
            editor = MutualSelfAttentionControl(masa_step, masa_layer, inject_uncond=inject_uncond, inject_cond=inject_cond)
            # editor = MutualSelfAttentionControlMaskAuto(masa_step, masa_layer, ref_token_idx=1, cur_token_idx=2)  # NOTE: replace the token idx with the corresponding index in the prompt if needed
            regiter_attention_editor_diffusers(self.ldm_stable, editor)
            # NOTE: 使用 这个 self.ldm_Stable时候 需要 将self.ldm_stable换成Masactrl的 
            image_masactrl = self.ldm_stable(prompts,
                           latents=z_inverted_noise_code,
                           num_inference_steps=num_inference_steps,
                           guidance_scale=[1, guidance_scale],
                           neg_prompt=prompt_src if npi else None,
                           prox=prox,
                           prox_step=prox_step,
                           quantile=quantile,
                           npi_interp=npi_interp,
                           npi_step=npi_step,
                           )
            out_image=Image.fromarray(np.concatenate((image_gt,image_masactrl[0],image_masactrl[-1]),axis=1))
            # print("Real image | Reconstructed image | Edited image")
            return out_image
    

    def edit_image_ddim_masactrl(self, image_path,prompt_src,prompt_tar,guidance_scale,step=4,layper=10):
        source_image=load_image(image_path, self.device)
        

        from masactrl.masactrl_utils import AttentionBase
        prompts=["", prompt_tar]
        
        start_code, latents_list = self.ldm_stable.invert(source_image,
                                            "",
                                            guidance_scale=guidance_scale,
                                            num_inference_steps=self.num_ddim_steps,
                                            return_intermediates=True)
        start_code = start_code.expand(len(prompts), -1, -1, -1)
        
        # results of direct synthesis
        editor = AttentionBase()
        regiter_attention_editor_diffusers(self.ldm_stable, editor)
        image_fixed = self.ldm_stable([prompt_tar],
                            latents=start_code[-1:],
                            num_inference_steps=self.num_ddim_steps,
                            guidance_scale=guidance_scale)
        
        # hijack the attention module
        editor = MutualSelfAttentionControl(step, layper)
        regiter_attention_editor_diffusers(self.ldm_stable, editor)

        # inference the synthesized image
        image_masactrl = self.ldm_stable(prompts,
                            latents=start_code,
                            guidance_scale=guidance_scale)
        
        #image_instruct = txt_draw(f"source prompt: {prompt_src}\ntarget prompt: {prompt_tar}")
        out_image=np.concatenate((
                                ((source_image[0].permute(1,2,0).detach().cpu().numpy() * 0.5 + 0.5)*255).astype(np.uint8),
                                image_masactrl[0],image_masactrl[-1]),1)
        
        return Image.fromarray(out_image)
    

    

    # pnp的话 则是需要环境为0.17.0 里面的register 方法 真的需要仔细一点看 才可以 
    def edit_pnp(self):
        pass





        