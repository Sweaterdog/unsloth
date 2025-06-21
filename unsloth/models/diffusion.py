# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from ..kernels import post_patch_loss_function # May not be relevant for diffusion
from ._utils import __version__, get_statistics, patch_tokenizer, is_bfloat16_supported
from ..save import patch_saving_functions
from peft import LoraConfig, TaskType, get_peft_model as _get_peft_model # Placeholder
from unsloth_zoo.utils import _get_dtype
from unsloth_zoo.patching_utils import patch_model_and_tokenizer # May need diffusion specific
from unsloth_zoo.training_utils import prepare_model_for_training # May need diffusion specific
import os
import gc
import platform
from huggingface_hub.utils import get_token
from transformers import set_seed as transformers_set_seed
from .loader_utils import get_xformers_version, HAS_FLASH_ATTENTION

# Placeholder for PEFT regex for UNet
def get_unet_peft_regex(model):
    """
    Placeholder function to get regex for targeting modules in a UNet
    for LoRA/PEFT. This will need to be adapted based on UNet structure.
    """
    print("Unsloth: Placeholder get_unet_peft_regex called. Implement for actual UNet.")
    # Example: target attention blocks and resnets
    # This is highly dependent on the specific UNet architecture from diffusers
    target_modules = []
    for name, module in model.named_modules():
        if "attn" in name or "resnet" in name: # A very generic example
            if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
                target_modules.append(name)
    return target_modules


class FastDiffusionModel:

    @staticmethod
    def from_pretrained(
        model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base",
        dtype             = None,
        load_in_4bit      = False, # Diffusion models typically not 4-bit quantized in same way
        load_in_8bit      = False, # Diffusion models typically not 8-bit quantized in same way
        token             = None,
        device_map        = "auto", # Diffusers handles device mapping
        torch_compile     = False,  # Placeholder for torch.compile
        **kwargs,
    ):
        os.environ["UNSLOTH_USE_NEW_MODEL"] = "1" # Keep consistent if used elsewhere
        if token is None: token = get_token()
        SUPPORTS_BFLOAT16 = is_bfloat16_supported()
        gpu_stats = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
        platform_system = platform.system()
        xformers_version = get_xformers_version()

        statistics = \
           f"==((====))==  Unsloth {__version__}: Fast DiffusionModel patching. Diffusers: {import('diffusers').__version__}\n"
        if gpu_stats:
            statistics += \
               f"   \\\\   /|    {gpu_stats.name}. Num GPUs = {torch.cuda.device_count()}. Max memory: {round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)} GB. Platform: {platform_system}.\n"\
               f"O^O/ \\_/ \\    Torch: {torch.__version__}. CUDA: {gpu_stats.major}.{gpu_stats.minor}. CUDA Toolkit: {torch.version.cuda}. Triton: {import('triton').__version__ if import('triton') else 'N/A'}\n"\
               f"\\        /    Bfloat16 = {str(SUPPORTS_BFLOAT16).upper()}. FA [Xformers = {xformers_version}. FA2 = {HAS_FLASH_ATTENTION}]\n"
        else:
            statistics += f"   \\\\   /|    No GPU detected. Platform: {platform_system}.\n"\
                          f"O^O/ \\_/ \\    Torch: {torch.__version__}. Bfloat16 = {str(SUPPORTS_BFLOAT16).upper()}.\n"
        statistics += f' "-____-"     Free license: http://github.com/unslothai/unsloth'
        print(statistics)

        get_statistics()

        if dtype is None:
            dtype = torch.float16 if not SUPPORTS_BFLOAT16 else torch.bfloat16
        elif dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16:
            print("Unsloth: Device does not support bfloat16. Will change to float16.")
            dtype = torch.float16

        if load_in_4bit or load_in_8bit:
            print("Unsloth: load_in_4bit/8bit is not typically used with diffusers pipelines in the same way as transformers. Ignoring.")
            load_in_4bit = False
            load_in_8bit = False

        # Load the pipeline
        # For diffusion models, we usually load the whole pipeline.
        # Individual components like unet, vae, text_encoder can be accessed from the pipeline.
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            token=token,
            **kwargs, # Pass other pipeline args like custom scheduler
        )

        # For Unsloth, we might want to manage components individually if we're optimizing them.
        # Let's assume the primary model to optimize/PEFT is the UNet.
        model = pipeline.unet
        model.tokenizer = pipeline.tokenizer
        model.text_encoder = pipeline.text_encoder
        model.vae = pipeline.vae
        model.scheduler = pipeline.scheduler
        model.feature_extractor = pipeline.feature_extractor # Can be None
        model.pipeline = pipeline # Keep a reference to the full pipeline

        model.config = model.config if hasattr(model, "config") else {} # UNet has a config
        model.config.update({"unsloth_version" : __version__})

        # Placeholder for patching functions if needed for diffusion models
        # patch_saving_functions(model, vision=False, diffusion=True)

        # Placeholder: Diffusion models don't use _saved_temp_tokenizer in the same way
        # model.max_seq_length = # Not directly applicable to UNet in the same way as LLMs

        # Placeholder for torch.compile for the UNet
        if torch_compile:
            print("Unsloth: Applying torch.compile to the UNet (experimental).")
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False) # Example config

        # Clear deleted GPU items
        for _ in range(3):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"Unsloth: Loaded diffusion model [{model_name_or_path}] into Unsloth FastDiffusionModel.")
        print(f"Unsloth: Target UNet for PEFT/LoRA: {type(model)}")

        # Add for_inference and for_training stubs
        model.for_training  = lambda use_gradient_checkpointing=True: FastDiffusionModel.for_training(model, use_gradient_checkpointing)
        model.for_inference = lambda: FastDiffusionModel.for_inference(model)

        return model, pipeline.tokenizer, pipeline # Return model (UNet), tokenizer, and full pipeline

    @staticmethod
    def get_peft_model(
        model, # This should be the UNet
        r                          = 16,
        lora_alpha                 = 16,
        lora_dropout               = 0,
        target_modules             = None, # Will need specific logic for UNets
        bias                       = "none",
        use_gradient_checkpointing = True, # For UNet
        random_state               = 3407,
        **kwargs
    ):
        transformers_set_seed(random_state)
        # transformers_set_seed(random_state) # Already called in from_pretrained or should be managed by user

        if isinstance(model, str):
            raise ValueError("Unsloth: Pass the UNet model itself to `get_peft_model`, not the model name or path.")

        # Refined target_modules identification for UNets
        # Standard LoRA for Stable Diffusion often targets attention query, key, value, output
        # and sometimes feedforward layers within attention blocks.
        # Conv2d layers in resnet blocks can also be targeted.
        if target_modules is None or target_modules == "all-linear" or target_modules == "default":
            target_modules = set()
            print("Unsloth: Determining target modules for UNet LoRA...")
            for name, module in model.named_modules():
                # Target linear layers in attention blocks (q, k, v, out)
                if isinstance(module, torch.nn.Linear) and \
                   any(key in name for key in [".to_q", ".to_k", ".to_v", ".to_out", "proj_attn"]): # Common naming
                    target_modules.add(name)
                # Target specific Conv2d layers if desired, e.g., in ResBlocks or Attention
                # Example: if isinstance(module, torch.nn.Conv2d) and ("ff.net" in name or "resnets" in name):
                #    target_modules.add(name)

            if not target_modules:
                warnings.warn(
                    "Unsloth: Could not automatically identify standard LoRA target modules in the UNet. "
                    "PEFT application might fail or be ineffective. "
                    "Please specify `target_modules` manually. Common targets are Linear layers in attention blocks "
                    "(e.g., names containing 'to_q', 'to_k', 'to_v', 'to_out', 'proj_attn')."
                )
            else:
                print(f"Unsloth: Automatically determined target_modules for UNet: {list(target_modules)[:10]}...")
            target_modules = list(target_modules)
            if not target_modules: # If still empty after auto-detection, provide a fallback to avoid error but warn heavily
                warnings.warn("Unsloth: target_modules list is empty. LoRA will not be applied effectively.")

        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            # TaskType is often not needed or set to None for custom models / UNets with PEFT
            # PEFT will attempt to determine layer types; ensure target_modules are compatible (Linear, Conv2d)
        )

        if use_gradient_checkpointing and hasattr(model, "enable_gradient_checkpointing"):
            print("Unsloth: Enabling gradient checkpointing for UNet.")
            model.enable_gradient_checkpointing()

        # Before applying PEFT, ensure the base model is in training mode if GC is enabled
        # This is because `enable_gradient_checkpointing()` might check for `model.training`
        if use_gradient_checkpointing : model.train()

        try:
            peft_model = _get_peft_model(model, lora_config)
            print("Unsloth: Successfully applied PEFT to UNet.")
        except Exception as e:
            print(f"Unsloth: Failed to apply PEFT to UNet. Error: {e}")
            print("Ensure `target_modules` are valid names of torch.nn.Linear or torch.nn.Conv2d layers.")
            return model # Return original model if PEFT application fails

        # Add for_inference and for_training methods to the PEFT model
        peft_model.for_training  = lambda use_gradient_checkpointing=True: FastDiffusionModel.for_training(peft_model, use_gradient_checkpointing)
        peft_model.for_inference = lambda: FastDiffusionModel.for_inference(peft_model)

        return peft_model


    @staticmethod
    def sample(model, prompt: Union[str, List[str]], num_inference_steps: int = 50, guidance_scale: float = 7.5, **kwargs):
        """
        Generates an image using the Stable Diffusion pipeline associated with the UNet model.
        Make sure to call `model.for_inference()` before using this method.
        """
        if not hasattr(model, 'pipeline'):
            raise ValueError(
                "Unsloth: UNet model does not have a 'pipeline' attribute. "
                "Ensure it was loaded correctly using FastDiffusionModel.from_pretrained "
                "and that `model.for_inference()` has been called."
            )

        # `for_inference()` should have already moved the pipeline to the correct device
        # and set it to eval mode.
        if model.training:
            warnings.warn(
                "Unsloth: Model is in training mode. Call `model.for_inference()` for proper image generation."
            )

        print(f"Unsloth: Generating image with prompt: '{prompt if isinstance(prompt, str) else prompt[0]}...'")
        with torch.no_grad():
            # The pipeline __call__ method handles device placement internally for its components
            # assuming the pipeline itself is on the correct device.
            output = model.pipeline(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                **kwargs
            ).images[0]
        return image

    # Alias for sample
    generate_image = sample

    @staticmethod
    def for_inference(model):
        """ Sets the model (UNet) and its pipeline to evaluation mode and moves to model's device. """
        current_device = model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if hasattr(model, 'eval'): model.eval()

        if hasattr(model, 'pipeline'):
            # Move pipeline components to the UNet's current device
            model.pipeline.to(current_device)
            # Set pipeline components to eval mode
            if hasattr(model.pipeline.vae, 'eval'): model.pipeline.vae.eval()
            if hasattr(model.pipeline.text_encoder, 'eval'): model.pipeline.text_encoder.eval()
        else:
            warnings.warn("Unsloth: UNet model does not have a 'pipeline' attribute. Full inference capabilities may be limited.")

        # Disable gradient checkpointing during inference for the UNet
        if hasattr(model, "disable_gradient_checkpointing"):
            model.disable_gradient_checkpointing()
        elif hasattr(model, "gradient_checkpointing_disable"): # common alternative name
            model.gradient_checkpointing_disable()
        if hasattr(model, "gradient_checkpointing") and isinstance(model.gradient_checkpointing, bool):
             model.gradient_checkpointing = False

        if hasattr(model, "training") and isinstance(model.training, bool): model.training = False

        print(f"Unsloth: FastDiffusionModel (UNet and pipeline) set to inference mode on device {current_device}.")
        return model

    @staticmethod
    def for_training(model, use_gradient_checkpointing=True):
        """ Sets the model (UNet) to training mode. """
        if hasattr(model, 'train'): model.train()
        if use_gradient_checkpointing:
            if hasattr(model, "enable_gradient_checkpointing"): model.enable_gradient_checkpointing()
            if hasattr(model, "gradient_checkpointing"): model.gradient_checkpointing = True
        if hasattr(model, "training"): model.training = True
        print("Unsloth: FastDiffusionModel (UNet) set to training mode.")
        return model

# Example Usage (for testing purposes, would not run in final script)
if __name__ == '__main__':
    # This section is for local testing and would be removed or commented out.
    # Make sure to install diffusers, transformers, accelerate
    # pip install diffusers transformers accelerate safetensors

    # Only run if a GPU is available for a meaningful test
    if torch.cuda.is_available():
        print("Unsloth: Running FastDiffusionModel example...")
        unet, tokenizer, pipeline = FastDiffusionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5", # Using a smaller model for quicker test
            # revision="fp16", # Some models have fp16 revisions
            torch_dtype=torch.float16,
        )

        # Move pipeline to GPU
        pipeline.to("cuda")
        unet.to("cuda") # unet is part of pipeline, but if extracted, move it too

        # Test PEFT (placeholder)
        # unet_peft = FastDiffusionModel.get_peft_model(unet, r=8)

        # Test sampling (placeholder)
        prompt = "A photo of an astronaut riding a horse on the moon"
        try:
            image = FastDiffusionModel.sample(unet, prompt, num_inference_steps=20) # Use fewer steps for test
            image.save("test_astronaut.png")
            print("Unsloth: Example image saved to test_astronaut.png")
        except Exception as e:
            print(f"Unsloth: Error during example image generation: {e}")
            print("Ensure you have `torch` and `diffusers` properly installed and a GPU is available.")

        print("Unsloth: FastDiffusionModel example finished.")
    else:
        print("Unsloth: Skipping FastDiffusionModel example as no CUDA GPU is available.")

pass
