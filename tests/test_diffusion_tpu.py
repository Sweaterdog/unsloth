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

import unittest
import torch
import os
import shutil # For cleanup
from typing import Dict, Any, List
from peft import PeftModel # For loading LoRA adapters

# Attempt to import XLA utilities for mocking or conditional testing
try:
    import torch_xla.core.xla_model as xm
    _XLA_AVAILABLE = True
except ImportError:
    _XLA_AVAILABLE = False
    xm = None

# Import Unsloth components to be tested
# This assumes 'unsloth' is in PYTHONPATH or installed
try:
    from unsloth import FastDiffusionModel, DiffusionTrainer, DiffusionTrainingArguments
except ImportError:
    # Adjust path if running tests directly from a subdirectory of the project root
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from unsloth import FastDiffusionModel, DiffusionTrainer, DiffusionTrainingArguments


# A mock XLA environment for local testing if TPU is not available
class MockXm:
    def xla_device(self):
        return torch.device("cpu") # Simulate XLA device with CPU for tests

    def xrt_world_size(self):
        return 1 # Simulate a single device world

    def mark_step(self):
        pass # No-op

    def mesh_reduce(self, name, data, reduction):
        return data # No-op, returns data as is

    def optimizer_step(self, optimizer, optimizer_args={}):
        # Simulate optimizer step, respecting optimizer_args if any (like 'grad_clipping')
        # This is a very basic simulation.
        if hasattr(optimizer, 'step'):
             optimizer.step()
        return None


@unittest.skipIf(not _XLA_AVAILABLE and "CI_TEST_XLA_MOCK" not in os.environ, "Skipping TPU tests if torch_xla is not available (and not in CI mock mode)")
class TestDiffusionTPU(unittest.TestCase):

    def setUp(self):
        # Use a very small model for testing to speed up downloads and loading
        self.model_name = "hf-internal-testing/tiny-stable-diffusion-pipe"
        self.output_dir = "./test_diffusion_output_dir" # For artifacts

        self.is_on_tpu_environment = _XLA_AVAILABLE and os.environ.get("XRT_TPU_CONFIG") is not None

        if not _XLA_AVAILABLE and "CI_TEST_XLA_MOCK" in os.environ:
            print("Mocking XLA environment for CI test...")
            global xm
            xm = MockXm()
            # Ensure DiffusionTrainer also uses the mock
            import unsloth.diffusion_trainer
            unsloth.diffusion_trainer.xm = xm
            unsloth.diffusion_trainer._XLA_AVAILABLE = True
            self.is_on_tpu_environment = True # Pretend we are on TPU for mocked tests

    def test_load_fast_diffusion_model(self):
        """Test loading a diffusion model using FastDiffusionModel."""
        unet, tokenizer, pipeline = FastDiffusionModel.from_pretrained(
            model_name_or_path=self.model_name,
            torch_dtype=torch.float32, # Use float32 for more robust testing on CPU/mocked TPU
        )
        self.assertIsNotNone(unet, "UNet should not be None")
        self.assertIsNotNone(tokenizer, "Tokenizer should not be None")
        self.assertIsNotNone(pipeline, "Pipeline should not be None")
        self.assertTrue(hasattr(unet, "pipeline"), "UNet should have a reference to the pipeline")
        print(f"Successfully loaded {self.model_name} with FastDiffusionModel.")

    def test_diffusion_trainer_instantiation_tpu(self):
        """Test instantiating DiffusionTrainer, simulating TPU environment if necessary."""
        unet, tokenizer, pipeline = FastDiffusionModel.from_pretrained(
            model_name_or_path=self.model_name,
            torch_dtype=torch.float32,
        )

        # Dummy dataset and collator
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, num_samples=10):
                self.num_samples = num_samples
                self.pixel_values = torch.randn(num_samples, 3, 32, 32) # Tiny images
                self.input_ids = torch.randint(0, tokenizer.vocab_size, (num_samples, 10)) # Tiny sequences

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                return {"pixel_values": self.pixel_values[idx], "input_ids": self.input_ids[idx]}

        def collate_fn(examples: List[Dict[str, Any]]):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            input_ids = torch.stack([example["input_ids"] for example in examples])
            return {"pixel_values": pixel_values, "input_ids": input_ids}

        train_dataset = DummyDataset()

        training_args_dict = {
            "output_dir": "./test_diffusion_trainer_tpu_output",
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "logging_steps": 1,
            "report_to": "none", # Disable wandb/tensorboard for tests
        }
        if self.is_on_tpu_environment:
            training_args_dict["tpu_num_cores"] = 8 # Typical TPU pod slice, or use xm.xrt_world_size() if available

        training_args = DiffusionTrainingArguments(**training_args_dict)

        trainer = DiffusionTrainer(
            model=unet,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=collate_fn,
            text_encoder=pipeline.text_encoder,
            vae=pipeline.vae,
            scheduler=pipeline.scheduler,
        )
        self.assertIsNotNone(trainer, "DiffusionTrainer instantiation failed.")
        print("DiffusionTrainer instantiated successfully.")

        # Test _wrap_model if on (real or mocked) TPU
        if self.is_on_tpu_environment and trainer.is_tpu:
            print("Testing DiffusionTrainer._wrap_model() on (mocked/real) TPU...")
            trainer.model = trainer._wrap_model(trainer.model) # UNet
            # Check if model and components are on the XLA device
            expected_device_type = "xla" if _XLA_AVAILABLE and os.environ.get("XRT_TPU_CONFIG") is not None else "cpu" # cpu for mock

            if hasattr(trainer.model, "device"):
                 self.assertTrue(str(trainer.model.device).startswith(expected_device_type), f"UNet not on {expected_device_type} device")
            if hasattr(trainer.vae, "device"):
                 self.assertTrue(str(trainer.vae.device).startswith(expected_device_type), f"VAE not on {expected_device_type} device")
            if hasattr(trainer.text_encoder, "device"):
                 self.assertTrue(str(trainer.text_encoder.device).startswith(expected_device_type), f"Text Encoder not on {expected_device_type} device")
            print("DiffusionTrainer._wrap_model() executed, device checks passed (or skipped if not applicable).")

    def tearDown(self):
        """Clean up any artifacts created during tests."""
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        # Clean up other potential directories from other tests if necessary
        if os.path.exists("./test_diffusion_trainer_tpu_output"):
             shutil.rmtree("./test_diffusion_trainer_tpu_output")

    def test_full_lora_finetune_and_inference(self):
        """Test full LoRA fine-tuning, saving, loading adapters, and inference."""
        # 1. Load base model
        unet, tokenizer, pipeline = FastDiffusionModel.from_pretrained(
            model_name_or_path=self.model_name,
            torch_dtype=torch.float32, # Using float32 for testing robustness
        )
        self.assertIsNotNone(unet)
        self.assertIsNotNone(tokenizer)
        self.assertIsNotNone(pipeline)

        # 2. Apply LoRA
        lora_unet = FastDiffusionModel.get_peft_model(
            unet,
            r=4, # Small rank for testing
            lora_alpha=8,
            target_modules=None, # Use auto-detection
            use_gradient_checkpointing=False, # Disable for simpler test run
        )
        self.assertIsNotNone(lora_unet, "Applying LoRA failed.")
        # Check if it's a PEFT model
        from peft import PeftModel as PeftModelType # Avoid conflict with class name
        self.assertIsInstance(lora_unet, PeftModelType, "Model is not a PeftModel after get_peft_model.")


        # 3. Prepare dummy dataset and collator
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, num_samples=8): # Reduced samples for faster test
                self.num_samples = num_samples
                # Using pipeline's tokenizer and actual image size from pipeline if available
                img_size = pipeline.unet.config.sample_size
                self.pixel_values = torch.randn(num_samples, 3, img_size, img_size)
                self.input_ids = torch.randint(0, tokenizer.vocab_size, (num_samples, tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length else 77 ))

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                return {"pixel_values": self.pixel_values[idx], "input_ids": self.input_ids[idx]}

        def collate_fn(examples: List[Dict[str, Any]]):
            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            input_ids = torch.stack([example["input_ids"] for example in examples])
            return {"pixel_values": pixel_values, "input_ids": input_ids}
        train_dataset = DummyDataset()

        # 4. Training Arguments
        training_args_dict = {
            "output_dir": self.output_dir, # Use self.output_dir for cleanup
            "max_steps": 2, # Run only for 2 steps for speed
            "per_device_train_batch_size": 1,
            "logging_steps": 1,
            "report_to": "none",
            "remove_unused_columns": False, # Important for custom datasets
        }
        if self.is_on_tpu_environment:
            training_args_dict["tpu_num_cores"] = xm.xrt_world_size() if _XLA_AVAILABLE and hasattr(xm, "xrt_world_size") and xm.xrt_world_size() is not None else 1


        training_args = DiffusionTrainingArguments(**training_args_dict)

        # 5. Instantiate Trainer
        trainer = DiffusionTrainer(
            model=lora_unet,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=collate_fn,
            text_encoder=pipeline.text_encoder,
            vae=pipeline.vae,
            scheduler=pipeline.scheduler,
        )
        self.assertIsNotNone(trainer)

        # 6. Minimal training execution
        print("Starting minimal training run...")
        train_result = trainer.train()
        self.assertIsNotNone(train_result, "Trainer.train() returned None.")
        self.assertTrue(train_result.training_loss > 0, "Training loss should be positive.")
        print(f"Minimal training completed. Loss: {train_result.training_loss}")

        # 7. Save LoRA adapters
        # In a real distributed scenario, only master would save. For this test, it's fine.
        lora_adapters_path = os.path.join(self.output_dir, "lora_adapters_test")
        lora_unet.save_pretrained(lora_adapters_path)
        self.assertTrue(os.path.exists(os.path.join(lora_adapters_path, "adapter_model.safetensors")) or \
                        os.path.exists(os.path.join(lora_adapters_path, "adapter_model.bin")) )
        print(f"LoRA adapters saved to {lora_adapters_path}")

        # 8. Load fresh base model for inference
        unet_base_for_inference, _, pipeline_for_inference = FastDiffusionModel.from_pretrained(
            model_name_or_path=self.model_name,
            torch_dtype=torch.float32,
        )

        # 9. Load adapters into the new base model
        loaded_lora_unet = PeftModel.from_pretrained(unet_base_for_inference, lora_adapters_path)
        self.assertIsNotNone(loaded_lora_unet)
        print("LoRA adapters loaded into a fresh base model.")

        # 10. Image generation with loaded adapters
        # Move to appropriate device
        current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.is_on_tpu_environment and _XLA_AVAILABLE and xm :
            current_device = xm.xla_device()

        loaded_lora_unet.to(current_device)
        pipeline_for_inference.vae.to(current_device)
        pipeline_for_inference.text_encoder.to(current_device)
        pipeline_for_inference.scheduler.to(current_device)
        pipeline_for_inference.unet = loaded_lora_unet # Replace UNet in pipeline

        loaded_lora_unet.eval() # Set to eval mode

        prompt = "A tiny red square" # Match dummy data
        try:
            image = pipeline_for_inference(prompt, num_inference_steps=2, height=32, width=32).images[0]
            self.assertIsNotNone(image, "Generated image should not be None.")
            self.assertTrue(hasattr(image, "size"), "Generated object does not look like an image.")
            print(f"Image generated successfully with loaded LoRA adapters: '{prompt}'")
        except Exception as e:
            self.fail(f"Image generation with loaded LoRA adapters failed: {e}")


    def test_generate_image_basic(self):
        """Test basic image generation with FastDiffusionModel."""
        unet, _, pipeline = FastDiffusionModel.from_pretrained(
            model_name_or_path=self.model_name,
            torch_dtype=torch.float32,
        )

        # Ensure model is on the correct device for inference (CPU for this test if no real TPU/GPU)
        current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.is_on_tpu_environment and _XLA_AVAILABLE and xm: # Real or mocked XLA
            current_device = xm.xla_device()

        unet.to(current_device) # Move UNet
        unet.for_inference() # This should also move pipeline components

        prompt = "A tiny cat"
        try:
            # Use a very small number of steps for speed
            image = unet.generate_image(prompt, num_inference_steps=1, height=32, width=32)
            self.assertIsNotNone(image, "Generated image should not be None")
            # PIL images from diffusers usually have a 'size' attribute
            self.assertTrue(hasattr(image, "size"), "Generated object does not look like an image.")
            print(f"Image generated successfully with prompt: '{prompt}'")
        except Exception as e:
            self.fail(f"Image generation failed with error: {e}")


if __name__ == "__main__":
    # To run tests with XLA mock even if no torch_xla is installed:
    # CI_TEST_XLA_MOCK=1 python -m unittest tests.test_diffusion_tpu.py
    unittest.main()
