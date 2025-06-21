# Unsloth TPU Support Documentation

Unsloth now supports fine-tuning on Google Cloud TPUs and Colab TPUs using PyTorch/XLA.

## Requirements
- `torch_xla` (install with `pip install torch_xla`)
- TPU-enabled runtime (Google Colab, Kaggle, or GCP)

## Usage
Unsloth will automatically detect and use the TPU if available. No code changes are required for basic usage.

### Example
```python
from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
args = UnslothTrainingArguments(
    output_dir="/tmp/test_tpu_trainer",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    learning_rate=1e-4,
    logging_steps=1,
    no_cuda=True,  # disables CUDA, not XLA
)
trainer = UnslothTrainer(
    model=model,
    args=args,
    train_dataset=[{"input_ids": torch.tensor([0,1,2,3])}],
    tokenizer=tokenizer,
)
trainer.train()
```

## Distributed TPU Training
To use all 8 TPU cores for Language Models, use `UnslothTrainer`:
```python
# For Language Models
def train_fn_llm(index, ...):
    # Your LLM training code here using UnslothTrainer
    ...
UnslothTrainer.launch_distributed(train_fn_llm, args=(...))
```

For Diffusion Models, use `DiffusionTrainer`:
```python
# For Diffusion Models
def train_fn_diffusion(index, ...):
    # Your Diffusion model training code here using DiffusionTrainer
    ...
DiffusionTrainer.launch_distributed(train_fn_diffusion, args=(...))
```

Refer to the respective example notebooks for detailed usage.

## Fine-tuning Diffusion Models on TPU

Unsloth now also supports fine-tuning diffusion models (e.g., Stable Diffusion) on TPUs. This is facilitated by the new `FastDiffusionModel` for loading diffusion pipelines and `DiffusionTrainer` for handling the training loop on TPUs.

**Key Features:**
- **Efficient Fine-tuning with LoRA:** Similar to LLMs, LoRA (Low-Rank Adaptation) is the primary method supported for efficiently fine-tuning the UNet component of diffusion models.
- **Distributed Training:** Leverage all available TPU cores for faster training using `DiffusionTrainer.launch_distributed`.

**Example Notebook:**
For a detailed walkthrough and runnable code, please refer to the example notebook:
[`examples/tpu_diffusion_finetuning.ipynb`](./examples/tpu_diffusion_finetuning.ipynb)

This notebook covers:
- Setting up the TPU environment.
- Loading a diffusion model with `FastDiffusionModel`.
- Preparing a dataset for image-caption fine-tuning.
- Applying LoRA to the UNet.
- Using `DiffusionTrainer` for distributed fine-tuning on TPUs.
- Performing inference with the fine-tuned LoRA adapters.

## Notes
- TPU support is experimental. Please report issues on GitHub.
- Some advanced features (e.g., custom kernels, Triton) may not be available on TPU.
- For best results, use bfloat16 precision (default on TPU).

## Limitations
- Only PyTorch/XLA is supported (JAX support is planned).
- Ensure your datasets and models fit in TPU memory.
- Not all CUDA-specific optimizations are available on TPU.
- For diffusion models, current support focuses on UNet fine-tuning with LoRA. Complex pipeline modifications or training other components (like the text encoder) on TPU might require custom setups.
- Performance and compatibility can vary depending on the specific diffusion model architecture and XLA version.
