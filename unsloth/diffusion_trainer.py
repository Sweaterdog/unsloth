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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments
from typing import Optional, Dict, Any, Union, List, Tuple
import warnings
import os

# Attempt to import XLA utilities, will be conditional later
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    _XLA_AVAILABLE = True
except ImportError:
    _XLA_AVAILABLE = False
    xm = None
    pl = None
    xmp = None

# Placeholder for Unsloth FastDiffusionModel, assuming it's in models.diffusion
# from .models.diffusion import FastDiffusionModel # This will be adjusted by __init__.py later

# A basic placeholder for training arguments specific to diffusion models
@dataclasses.dataclass
class DiffusionTrainingArguments(TrainingArguments):
    # Add diffusion specific arguments here if needed in future
    # For example:
    # snr_gamma: Optional[float] = field(default=None, metadata={"help": "SNR gamma for SNR weighting loss."})
    # max_grad_norm_unet: Optional[float] = field(default=1.0, metadata={"help": "Max gradient norm for UNet."})
    pass


class DiffusionTrainer(Trainer):
    def __init__(
        self,
        model = None, # Should be a FastDiffusionModel (UNet) or a PEFT-wrapped UNet
        args: Optional[DiffusionTrainingArguments] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer = None, # Typically CLIP tokenizer from the pipeline
        data_collator = None, # Custom collator for diffusion
        compute_metrics = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        # Additional components from the diffusion pipeline
        text_encoder = None,
        vae = None,
        scheduler = None, # Noise scheduler (e.g., DDIMScheduler)
        **kwargs,
    ):
        if args is None:
            output_dir = "tmp_trainer"
            print(f"Unsloth: No DiffusionTrainingArguments provided, using default output_dir: {output_dir}")
            args = DiffusionTrainingArguments(output_dir=output_dir)

        super().__init__(
            model=model, # This is the UNet
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer, # CLIP Tokenizer
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            optimizers=optimizers,
            **kwargs,
        )

        # Store diffusion specific components
        self.text_encoder = text_encoder
        self.vae = vae
        self.noise_scheduler = scheduler

        # Ensure requires_grad is correctly set for components
        if self.text_encoder: self.text_encoder.requires_grad_(False) # Usually frozen
        if self.vae: self.vae.requires_grad_(False) # Usually frozen
        if self.model: self.model.requires_grad_(True) # UNet is trained

        self.is_tpu = _XLA_AVAILABLE and self.args.tpu_num_cores is not None
        self.device_type = "tpu" if self.is_tpu else "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Unsloth DiffusionTrainer initialized. TPU available: {self.is_tpu}, Device type: {self.device_type}")
        if self.is_tpu:
            print(f"Unsloth: TPU cores specified: {self.args.tpu_num_cores}")


    def _wrap_model(self, model, training=True, dataloader=None):
        """ Wraps model for TPU training if applicable. """
        if self.is_tpu:
            if xm is None:
                raise RuntimeError("Unsloth: TPU training specified, but torch_xla is not available.")
            print("Unsloth: Wrapping model for TPU.")
            device = xm.xla_device()
            model = model.to(device)

            # Also move VAE and text_encoder to XLA device if they exist and are part of training
            # (though typically they are frozen and used for inference within the training step)
            if self.vae: self.vae = self.vae.to(device)
            if self.text_encoder: self.text_encoder = self.text_encoder.to(device)

            # Patch model.eval() and model.train() to always keep on XLA device
            # This is crucial for the UNet (self.model)
            orig_train = model.train
            orig_eval = model.eval
            def train_patch(mode=True):
                result = orig_train(mode)
                return result.to(device)
            def eval_patch():
                result = orig_eval()
                return result.to(device)
            model.train = train_patch
            model.eval = eval_patch

        return super()._wrap_model(model, training, dataloader)

    def get_train_dataloader(self) -> DataLoader:
        """ Returns the training dataloader, wrapped for TPU if applicable. """
        dataloader = super().get_train_dataloader()
        if self.is_tpu and pl is not None:
            print("Unsloth: Wrapping train dataloader with MpDeviceLoader for TPU.")
            dataloader = pl.MpDeviceLoader(dataloader, xm.xla_device())
        return dataloader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """ Returns the evaluation dataloader, wrapped for TPU if applicable. """
        dataloader = super().get_eval_dataloader(eval_dataset)
        if self.is_tpu and pl is not None:
            print("Unsloth: Wrapping eval dataloader with MpDeviceLoader for TPU.")
            dataloader = pl.MpDeviceLoader(dataloader, xm.xla_device())
        return dataloader

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Placeholder for diffusion model loss computation.
        This needs to be implemented according to the specific diffusion training loop.
        (e.g., noise prediction task as in standard Stable Diffusion)
        """
        # Ensure VAE and text_encoder are on the correct device, especially for TPU.
        # self._wrap_model should handle the main model (UNet).
        # However, if VAE/text_encoder are used explicitly here, ensure device consistency.
        current_device = model.device
        if self.vae and self.vae.device != current_device: self.vae.to(current_device)
        if self.text_encoder and self.text_encoder.device != current_device: self.text_encoder.to(current_device)


        # Example structure for Stable Diffusion loss:
        # 1. Get latents from VAE
        # 2. Get text embeddings from text_encoder
        # 3. Sample noise
        # 4. Add noise to latents
        # 5. Predict noise using UNet (model)
        # 6. Compute loss (e.g., MSE between predicted noise and sampled noise)

        # 'pixel_values' from dataset/collator
        # 'input_ids' from dataset/collator (tokenized prompt)
        pixel_values = inputs.get("pixel_values")
        input_ids = inputs.get("input_ids")

        if pixel_values is None or input_ids is None:
            raise ValueError("Unsloth: DiffusionTrainer expects 'pixel_values' and 'input_ids' in inputs.")

        with torch.no_grad(): # VAE and Text Encoder are usually frozen during UNet finetuning
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            encoder_hidden_states = self.text_encoder(input_ids)[0]

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the target for loss depending on the scheduler's prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # Predict the noise residual and compute loss
        model_pred = model(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # Optional: SNR weighting (from original diffusers train_dreambooth.py)
        # snr_gamma = getattr(self.args, 'snr_gamma', None)
        # if snr_gamma is not None:
        #     snr = compute_snr(timesteps)
        #     mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
        #     loss = loss * mse_loss_weights
        #     loss = loss.mean()

        return (loss, {"model_pred": model_pred}) if return_outputs else loss

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """ Perform a training step, including TPU synchronization. """
        loss = super().training_step(model, inputs) # Calls compute_loss internally
        if self.is_tpu and xm is not None:
            xm.mark_step()
        return loss

    def optimizer_step(self, *args, **kwargs):
        """ Perform an optimizer step, using XLA optimizer_step for TPUs. """
        if self.is_tpu and xm is not None:
            # The super().optimizer_step in HF Trainer does gradient accumulation, clipping, etc.
            # For TPUs, xm.optimizer_step is usually called after the loss.backward()
            # and before optimizer.zero_grad().
            # HF Trainer's optimizer_step handles this logic.
            # We might need to customize if HF's default way is not optimal for XLA.
            # For now, let HF Trainer handle it, assuming it calls self.optimizer.step() internally,
            # which then xm.optimizer_step (if optimizer was XLA's) would handle.
            # A common pattern is:
            # loss.backward()
            # xm.optimizer_step(self.optimizer) # This also does grad clipping.
            # optimizer.zero_grad()
            # However, HF Trainer has its own complex optimizer_step.
            # Let's assume the optimizer itself is not an XLA specific one unless we make it so.
            # If self.optimizer is a standard PyTorch optimizer, then super().optimizer_step() is fine.
            # If we were to use an XLA specific optimizer, we'd need to adapt this.
            # For now, relying on standard HF behavior. If issues arise, this needs revisit.
            # The most important part for general TPU usage is xm.mark_step() in training_step.
            # If using `torch_xla.amp` for mixed precision, that might also interact here.
            # For now, assume standard optimizer behavior within HF Trainer.
            # The `xm.optimizer_step(optimizer)` is typically used when *not* using HF Trainer's
            # own `optimizer_step` method.
            # If HF Trainer's `self.optimizer` is a standard optimizer, then TPUs work by
            # tracing ops. `xm.mark_step()` is key.
            pass # Rely on super's optimizer_step. xm.mark_step() is the main sync point.

        return super().optimizer_step(*args, **kwargs)


    def evaluate(self, *args, **kwargs):
        """ Perform evaluation, including TPU synchronization. """
        output = super().evaluate(*args, **kwargs)
        if self.is_tpu and xm is not None:
            xm.mark_step()
        return output

    def predict(self, *args, **kwargs):
        """ Perform prediction, including TPU synchronization. """
        output = super().predict(*args, **kwargs)
        if self.is_tpu and xm is not None:
            xm.mark_step()
        return output

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """ Saves the model, ensuring it's moved to CPU if on TPU. """
        # This saves the UNet part of the diffusion model.
        # For full pipeline saving, a different method would be needed.
        if self.is_tpu and xm is not None:
            if hasattr(self.model, "to"): # Ensure model has 'to' method
                print("Unsloth: Moving model to CPU before saving (TPU context).")
                self.model.to("cpu")

        super().save_model(output_dir, _internal_call)

        # After saving, if on TPU, move model back to XLA device
        if self.is_tpu and xm is not None:
            if hasattr(self.model, "to"):
                 print("Unsloth: Moving model back to XLA device after saving.")
                 self.model.to(xm.xla_device())

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """ Loads model from checkpoint, ensuring it's on XLA device if on TPU. """
        # Check if the model components (vae, text_encoder) should also be reloaded or handled.
        # This method in HF Trainer primarily deals with self.model (the UNet).
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        if self.is_tpu and xm is not None:
            if hasattr(self.model, "to"):
                print("Unsloth: Moving loaded model to XLA device.")
                self.model.to(xm.xla_device())
            # If other components like VAE/Text Encoder were part of the checkpoint
            # and are managed by this trainer's state, they might need moving too.
            # However, standard HF Trainer checkpointing focuses on `self.model`.
            # if self.vae: self.vae.to(xm.xla_device())
            # if self.text_encoder: self.text_encoder.to(xm.xla_device())

    def _gather_and_numpify(self, tensors, name):
        """ Gathers tensors from all TPUs and converts to numpy. """
        if self.is_tpu and xm is not None:
            # Note: xm.mesh_reduce returns a list of tensors if not a scalar
            # This needs to be handled correctly based on what 'tensors' are.
            # HuggingFace Trainer's default _gather_and_numpify expects tensors to be
            # potentially nested lists/tuples/dicts of tensors.
            # xm.mesh_reduce usually works on a single tensor or a list of tensors for reduction.
            # This might need more sophisticated handling if 'tensors' is a complex structure.
            # For simple cases (e.g., loss, or a dict of scalar metrics):
            if isinstance(tensors, torch.Tensor):
                tensors = xm.mesh_reduce(name, tensors, torch.mean) # Example reduction
                return tensors.cpu().numpy()
            elif isinstance(tensors, (list, tuple)):
                return type(tensors)(self._gather_and_numpify(t, f"{name}_{i}") for i, t in enumerate(tensors))
            elif isinstance(tensors, dict):
                return {k: self._gather_and_numpify(v, f"{name}_{k}") for k, v in tensors.items()}
            # If not a tensor or collection of tensors, return as is
            return tensors
        return super()._gather_and_numpify(tensors, name)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        """ Called by training loop; ensure TPU sync before logging/saving/evaluating. """
        if self.is_tpu and xm is not None:
            xm.mark_step()
        super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)


    @staticmethod
    def launch_distributed(fn, args=()):
        """ Launch a function on all TPU cores using xmp.spawn. """
        if not _XLA_AVAILABLE or xmp is None:
            raise RuntimeError("Unsloth: torch_xla.distributed.xla_multiprocessing (xmp) is not available for launch_distributed.")

        # Determine nprocs (number of TPU cores)
        # This might come from args or environment
        nprocs = xm.xrt_world_size() if xm.xrt_world_size() else 8 # Default to 8 if not found
        print(f"Unsloth: Launching distributed function on {nprocs} TPU cores.")
        xmp.spawn(fn, args=args, nprocs=nprocs, start_method='fork')


# Example of a dummy dataset and collator for diffusion models
class DummyDiffusionDataset(Dataset):
    def __init__(self, tokenizer, num_samples=100, image_size=512):
        self.num_samples = num_samples
        self.image_size = image_size
        self.prompts = [f"A photo of a cat number {i}" for i in range(num_samples)]
        self.tokenizer = tokenizer

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Dummy pixel values (e.g., random noise)
        pixel_values = torch.randn(3, self.image_size, self.image_size)
        # Tokenized prompt
        input_ids = self.tokenizer(
            self.prompts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids.squeeze() # Get (seq_len) instead of (1, seq_len)
        return {"pixel_values": pixel_values, "input_ids": input_ids}

def default_diffusion_data_collator(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    pixel_values = torch.stack([feature["pixel_values"] for feature in features])
    input_ids = torch.stack([feature["input_ids"] for feature in features])
    return {"pixel_values": pixel_values, "input_ids": input_ids}


if __name__ == '__main__':
    # This section is for local testing and would be removed or commented out.
    # Requires: pip install diffusers transformers accelerate bitsandbytes sentencepiece
    #           pip install unsloth # if using the main unsloth package

    # Check if torch_xla is available for a more complete test, otherwise run CPU/GPU test
    tpu_available_for_test = _XLA_AVAILABLE and xm is not None and (os.environ.get("XRT_TPU_CONFIG") is not None)

    if not torch.cuda.is_available() and not tpu_available_for_test:
        print("Unsloth: Skipping DiffusionTrainer example: No CUDA GPU or XLA TPU configured for test.")
    else:
        print("Unsloth: Running DiffusionTrainer example...")
        from unsloth.models.diffusion import FastDiffusionModel # Assuming it's runnable
        import dataclasses # Required for DiffusionTrainingArguments if not already imported

        # 1. Load a small diffusion model and tokenizer
        # Using a very small model for quick testing. runwayml/stable-diffusion-v1-5 is too large for CI without GPU.
        # For this test, we'll mock the model components if no actual model is easily loadable.
        try:
            unet, tokenizer, pipeline = FastDiffusionModel.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-pipe", # A very small SD pipeline for testing
                torch_dtype=torch.float32 if tpu_available_for_test else torch.float16, # float32 for TPU robustness in test
            )
            text_encoder = pipeline.text_encoder
            vae = pipeline.vae
            noise_scheduler = pipeline.scheduler
            # If on GPU, move to GPU
            if torch.cuda.is_available() and not tpu_available_for_test:
                unet.to("cuda")
                text_encoder.to("cuda")
                vae.to("cuda")

        except Exception as e:
            print(f"Could not load tiny SD model: {e}. Mocking components for trainer test structure.")
            from diffusers import UNet2DConditionModel, AutoencoderKL
            from diffusers.schedulers import DDPMScheduler
            from transformers import CLIPTextModel, CLIPTokenizer

            unet_config = UNet2DConditionModel.load_config("runwayml/stable-diffusion-v1-5", subfolder="unet")
            unet = UNet2DConditionModel(**unet_config) # mock unet
            tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
            text_encoder_config = CLIPTextModel.load_config("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
            text_encoder = CLIPTextModel(text_encoder_config) # mock text_encoder
            vae_config = AutoencoderKL.load_config("runwayml/stable-diffusion-v1-5", subfolder="vae")
            vae = AutoencoderKL(**vae_config) # mock vae
            noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")

            if torch.cuda.is_available() and not tpu_available_for_test:
                unet.to("cuda"); text_encoder.to("cuda"); vae.to("cuda")


        # 2. Create dummy dataset and collator
        train_dataset = DummyDiffusionDataset(tokenizer, num_samples=16) # Small dataset for test

        # 3. Define Training Arguments
        training_args = DiffusionTrainingArguments(
            output_dir="./diffusion_trainer_test_output",
            num_train_epochs=1,
            per_device_train_batch_size=1 if tpu_available_for_test else 2, # Smaller batch for CPU/GPU test
            # tpu_num_cores must be set if running on TPU, e.g. via environment or script args
            # For this test, we assume it's set if tpu_available_for_test is True
            tpu_num_cores = 8 if tpu_available_for_test else None,
            # Other arguments
            logging_steps=1,
            # no_cuda=tpu_available_for_test, # Let HF Trainer manage device based on args
            # Remove fp16 if on CPU or if causing issues on TPU for this test
            fp16 = False if tpu_available_for_test or not torch.cuda.is_available() else True,
        )

        # 4. Initialize Trainer
        trainer = DiffusionTrainer(
            model=unet,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=default_diffusion_data_collator,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=noise_scheduler,
        )

        print("Unsloth: DiffusionTrainer initialized for test.")

        # 5. (Optional) Test TPU wrapping if applicable
        if tpu_available_for_test and xm:
            print("Unsloth: Testing TPU model wrapping...")
            trainer.model = trainer._wrap_model(trainer.model)
            # trainer.train_dataloader = trainer.get_train_dataloader() # Called by train()
            print(f"Unsloth: Model device after wrap: {str(next(trainer.model.parameters()).device)}")
            if "xla" not in str(next(trainer.model.parameters()).device):
                 print("Warning: Model does not seem to be on XLA device after wrapping!")

        # 6. (Optional) Test a training step (will be slow on CPU)
        # This requires actual data and a full setup.
        # We'll just check if the structure is callable.
        try:
            print("Unsloth: Attempting to run a pseudo training step...")
            # Get a single batch
            dataloader = trainer.get_train_dataloader()
            batch = next(iter(dataloader))

            # Manually move batch to device if not on TPU (TPU loader handles it)
            if not trainer.is_tpu:
                batch = trainer._prepare_inputs(batch)

            # Call compute_loss
            loss = trainer.compute_loss(trainer.model, batch)
            print(f"Unsloth: Pseudo loss computed: {loss.item()}")

            # If on TPU, xm.mark_step() would be called in training_step
            if trainer.is_tpu: xm.mark_step()

        except Exception as e:
            print(f"Unsloth: Error during pseudo training step test: {e}")
            print("This might be due to component mismatches if using mocked components, or device issues.")
            if "xla" in str(e).lower():
                print("XLA related error. Ensure TPU environment is correctly set up if testing TPU.")

        print("Unsloth: DiffusionTrainer example finished.")
pass
import dataclasses # Add this at the top of the file if not already there for @dataclasses.dataclass
# Note: The above `import dataclasses` is a comment to remind where it should go.
# The actual `create_file_with_block` will handle the content as a whole.
