import torch
from torch import nn, autocast
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler


class CustomTextToImageModel(nn.Module):
    def __init__(self, config, device, from_pretrained=False):
        super().__init__()
        self.config = config
        self.latent_scaling_factor = config.latent_scaling_factor
        self.device = device

        # Seed generator to create the inital latent noise
        self.generator=torch.manual_seed(32)
        
        # check whether to load model from pretrained weights or saved weights file
        if from_pretrained:
            self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            self.clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
            self.unet_model = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_auth_token=True).to(device)
            self.kl_model = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=True).to(device)
        else:
            self.clip_tokenizer = CLIPTokenizer.from_pretrained('./model/tokenizer/')
            self.clip_model = CLIPTextModel.from_pretrained('./model/cliptext').to(device)
            self.unet_model = UNet2DConditionModel.from_pretrained('./model/unet').to(device)
            self.kl_model = AutoencoderKL.from_pretrained('./model/vae').to(device)

        # generate UNet scheduler
        self.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

        # unconditional input
        self.uncond_input = [""] * config.batch_size


    def forward(self, prompt):
        latents = self.setup_scheduler()
        latents = latents.to(self.device)
        text_embeddings = self.embed_prompts(prompt)
        latents = self.generate_latent_vector_with_unet(text_embeddings, latents)
        latents = self.latent_scaling_factor * latents
        image = self.kl_model.decode(latents).sample
        return image


    def embed_prompts(self, prompt):
        text_embeddings, max_length = self.tokenize_and_embed_input_text(prompt, self.clip_tokenizer.model_max_length)
        uncond_embeddings, _ = self.tokenize_and_embed_input_text(self.uncond_input, max_length, truncation=False)

        # For classifier-free guidance, we need 2 forward steps - One with text_embeddings, and other with unconditional embeddings
        # In practice, we could simply concatenate 2 vectors so that we do not need to run the forward call twice.
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings
    
    def tokenize_and_embed_input_text(self, text, max_len, truncation=True):
        tokenized_texts = self.clip_tokenizer(text, padding="max_length", max_length=max_len, truncation=truncation, return_tensors="pt")
        text_embeddings = self.clip_model(tokenized_texts.input_ids.to(self.device))[0]
        max_length = tokenized_texts.input_ids.shape[-1]
        return text_embeddings, max_length


    def generate_latent_vector_with_unet(self, text_embeddings, latents):
        if self.device == 'cuda':
            with autocast('cuda'):
                return self.run_denoising_loop(text_embeddings, latents)
        else:
            return self.run_denoising_loop(text_embeddings, latents)
    
    def run_denoising_loop(self, text_embeddings, latents):
        scheduler = self.scheduler
        for i, t in enumerate(scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma = scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            noise_pred = self.unet_model(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, i, latents).prev_sample
        return latents



    def setup_scheduler(self):
        latents = torch.randn(
            (self.config.batch_size, self.unet_model.in_channels, self.config.height // 8, self.config.width // 8),
            generator=self.generator,
        )
        num_inference_steps = self.config.num_inference_steps
        self.scheduler.set_timesteps(num_inference_steps)
        return latents * self.scheduler.sigmas[0]
