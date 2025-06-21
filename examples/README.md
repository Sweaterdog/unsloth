# Unsloth Examples

This directory contains example Jupyter notebooks demonstrating various features of the Unsloth library.

## Notebooks

1.  **`tpu_diffusion_finetuning.ipynb`**:
    *   **Description:** Demonstrates how to fine-tune a pre-trained Stable Diffusion model (e.g., from Stability AI or RunwayML) on a custom dataset using LoRA (Low-Rank Adaptation) on Google TPUs.
    *   **Features Covered:**
        *   Using `FastDiffusionModel` to load diffusion models.
        *   Applying LoRA to the UNet component.
        *   Setting up and using `DiffusionTrainer` for distributed training on TPUs.
        *   Saving LoRA adapters.
        *   Performing inference with the fine-tuned adapters.
    *   **Environment:** Google Colab with a TPU runtime.
    *   **Key Unsloth Components:** `FastDiffusionModel`, `DiffusionTrainer`, `DiffusionTrainingArguments`, `DiffusionTrainer.launch_distributed`.

---

*More examples will be added here as they are developed.*
