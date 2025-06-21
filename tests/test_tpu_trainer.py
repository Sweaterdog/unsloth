"""
Test TPU integration for UnslothTrainer.
Run this on a TPU-enabled environment (e.g., Google Colab with TPU runtime).
"""
import pytest
import torch
import warnings

pytestmark = pytest.mark.skipif(
    not (torch.__version__ >= '2.0.0'),
    reason="Requires torch >= 2.0.0"
)

try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_tpu_trainer_basic():
    if not HAS_XLA:
        warnings.warn("torch_xla not installed; skipping TPU test.")
        return
    device = xm.xla_device()
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2").to(device)
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
    xm.mark_step()  # Final sync
