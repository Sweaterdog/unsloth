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
To use all 8 TPU cores, use:
```python
def train_fn(index, ...):
    # Your training code here
    ...
UnslothTrainer.launch_distributed(train_fn, args=(...))
```

## Notes
- TPU support is experimental. Please report issues on GitHub.
- Some advanced features (e.g., custom kernels, Triton) may not be available on TPU.
- For best results, use bfloat16 precision (default on TPU).

## Limitations
- Only PyTorch/XLA is supported (JAX support is planned).
- Ensure your datasets and models fit in TPU memory.
- Not all CUDA-specific optimizations are available on TPU.
