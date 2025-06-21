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

import warnings
from dataclasses import dataclass, field
from typing import Optional
from functools import wraps

import trl
import inspect
from trl import SFTTrainer
from . import is_bfloat16_supported
from unsloth_zoo.training_utils import (
    unsloth_train as _unsloth_train,
)
from unsloth_zoo.vision_utils import (
    UnslothVisionDataCollator,
)
from packaging.version import Version
import dataclasses

__all__ = [
    "UnslothTrainingArguments",
    "UnslothTrainer",
    "unsloth_train",
    "_patch_trl_trainer",
    "UnslothVisionDataCollator",
]

# Unsloth gradient accumulation fix:
from transformers import __version__ as transformers_version
if Version(transformers_version) > Version("4.45.2"):
    def unsloth_train(trainer, *args, **kwargs):
        return trainer.train(*args, **kwargs)
    pass
else:
    def unsloth_train(trainer, *args, **kwargs):
        if len(args) != 0 or len(kwargs) != 0:
            raise RuntimeError(
                "Unsloth: Our custom gradient accumulation fixed trainer does not support other arguments.\n"\
                "If you want to use our fix inside of HF, please update `transformers` to the latest version via:\n"\
                '`pip uninstall transformers -y && pip install --upgrade --no-cache-dir transformers`'
            )
        print(
            "Unsloth: Using our custom gradient accumulation fixed trainer, which is not feature complete.\n"\
            "If you want to use our fix inside of HF, please update `transformers` to the latest version via:\n"\
            '`pip uninstall transformers -y && pip install --upgrade --no-cache-dir transformers`'
        )
        return _unsloth_train(trainer)
    pass
pass

try:
    from trl import SFTConfig as TrainingArguments
except:
    from transformers import TrainingArguments
pass
@dataclass
class UnslothTrainingArguments(TrainingArguments):
    embedding_learning_rate : Optional[float] = field(
        default = None,
        metadata = {"help" : "Different learning rates for embeddings and lm_head."}
    )
pass


def _create_unsloth_optimizer(
    model,
    optimizer_cls,
    optimizer_kwargs,
    embedding_lr = 5e-5,
):
    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    param_groups = \
    {
        "non_embeddings" : {},
        "embeddings"     : {},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if name.endswith("modules_to_save.default.weight"):
            partial_name = name[:-len(".modules_to_save.default.weight")]
            partial_name = partial_name[partial_name.rfind(".")+1:]
            print(f"Unsloth: Setting lr = {embedding_lr:.2e} instead of {lr:.2e} for {partial_name}.")
            param_groups["embeddings"]    [name] = param
        else:
            param_groups["non_embeddings"][name] = param
        pass
    pass

    optimizer_grouped_parameters = [
        {
            "params"       : list(param_groups["non_embeddings"].values()),
            "weight_decay" : weight_decay,
            "lr"           : lr,
        },
        {
            "params"       : list(param_groups["embeddings"].values()),
            "weight_decay" : weight_decay,
            "lr"           : embedding_lr,
        },
    ]
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer
pass


from unsloth import DEVICE_TYPE

class UnslothTrainer(SFTTrainer):
    def create_optimizer(self):
        embedding_learning_rate = getattr(self.args, "embedding_learning_rate", None)
        if embedding_learning_rate is None:
            return super().create_optimizer()

        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = SFTTrainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = _create_unsloth_optimizer(
                self.model,
                optimizer_cls,
                optimizer_kwargs,
                embedding_learning_rate,
            )
        return self.optimizer

    def training_step(self, *args, **kwargs):
        # TPU/XLA: Use xm.mark_step() to sync
        output = super().training_step(*args, **kwargs)
        if DEVICE_TYPE == "tpu":
            import torch_xla.core.xla_model as xm
            xm.mark_step()
        return output

    def _wrap_model(self, model, training=True, dataloader=None):
        # Move model to TPU if needed
        if DEVICE_TYPE == "tpu":
            import torch_xla.core.xla_model as xm
            model = model.to(xm.xla_device())
            # Patch model.eval() and model.train() to always keep on XLA device
            orig_train = model.train
            orig_eval = model.eval
            def train_patch(mode=True):
                result = orig_train(mode)
                return result.to(xm.xla_device())
            def eval_patch():
                result = orig_eval()
                return result.to(xm.xla_device())
            model.train = train_patch
            model.eval = eval_patch
        return super()._wrap_model(model, training, dataloader)

    def get_train_dataloader(self):
        dataloader = super().get_train_dataloader()
        if DEVICE_TYPE == "tpu":
            try:
                from torch_xla.distributed.parallel_loader import MpDeviceLoader
                import torch_xla.core.xla_model as xm
                dataloader = MpDeviceLoader(dataloader, xm.xla_device())
            except ImportError:
                warnings.warn("torch_xla is not installed. TPU DataLoader will not be used.")
        return dataloader

    def optimizer_step(self, *args, **kwargs):
        if DEVICE_TYPE == "tpu":
            import torch_xla.core.xla_model as xm
            xm.optimizer_step(self.optimizer)
        else:
            return super().optimizer_step(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        output = super().evaluate(*args, **kwargs)
        if DEVICE_TYPE == "tpu":
            import torch_xla.core.xla_model as xm
            xm.mark_step()
        return output

    def predict(self, *args, **kwargs):
        output = super().predict(*args, **kwargs)
        if DEVICE_TYPE == "tpu":
            import torch_xla.core.xla_model as xm
            xm.mark_step()
        return output

    def save_model(self, output_dir=None, _internal_call=False):
        # Ensure model is moved to CPU before saving on TPU
        if DEVICE_TYPE == "tpu":
            self.model.to("cpu")
        return super().save_model(output_dir, _internal_call)

    @staticmethod
    def launch_distributed(fn, args=()):
        """Launch a function on all TPU cores using xmp.spawn."""
        try:
            import torch_xla.distributed.xla_multiprocessing as xmp
            xmp.spawn(fn, args=args, nprocs=8, start_method='fork')
        except ImportError:
            raise RuntimeError("torch_xla is required for distributed TPU training.")
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        # On TPU, sync before logging/saving/evaluating
        if DEVICE_TYPE == "tpu":
            import torch_xla.core.xla_model as xm
            xm.mark_step()
        return super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        # Ensure model is loaded to XLA device on TPU
        model = super()._load_from_checkpoint(resume_from_checkpoint, model)
        if DEVICE_TYPE == "tpu":
            import torch_xla.core.xla_model as xm
            model = model.to(xm.xla_device())
        return model

    def _gather_and_numpify(self, tensors, name):
        # On TPU, use xm.mesh_reduce to aggregate metrics
        if DEVICE_TYPE == "tpu":
            import torch_xla.core.xla_model as xm
            import numpy as np
            result = xm.mesh_reduce(name, tensors, np.mean)
            return result
        return super()._gather_and_numpify(tensors, name)
pass

# From `trl>=0.13.0`, they changed how to pass several params to the trainer
# We need to patch to make the transition smooth
def _backwards_compatible_trainer(trainer_class, config_class):
    original_init = trainer_class.__init__
    
    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        # All Trainer tokenizer are now called processing_class
        trainer_params = set(inspect.signature(original_init).parameters.keys())

        if "processing_class" in trainer_params and "tokenizer" in kwargs:
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        pass

        if ("args" in kwargs) and (Version(trl.__version__) >= Version("0.13.0.dev0")):
            training_args = kwargs.pop("args", None)

            # Get parameters that Trainer.__init__ actually expects
            trainer_params.remove('self')
            trainer_params.remove('args')

            # Get fields that should be passed to Config init
            config_fields = {
                field.name: field for field in dataclasses.fields(config_class) 
                if field.init
            }
            
            # Create config dict with valid fields from training_args
            config_dict = {
                name: getattr(training_args, name)
                for name in config_fields
                if hasattr(training_args, name)
            }

            # Get parameters that exist in Config but not in TrainingArguments
            from transformers import TrainingArguments
            moved_params = \
                set(inspect.signature(config_class)     .parameters.keys()) - \
                set(inspect.signature(TrainingArguments).parameters.keys())
            
            # Separate kwargs into trainer kwargs and config kwargs
            trainer_kwargs = {}
            additional_config_kwargs = {}

            for key, value in kwargs.items():
                if key in trainer_params: trainer_kwargs[key] = value
                elif key in moved_params or key in config_fields:
                    additional_config_kwargs[key] = value
                else:
                    additional_config_kwargs[key] = value
                pass
            pass

            # Update config_dict with additional kwargs
            config_dict.update(additional_config_kwargs)

            # Create Config with all the collected parameters
            # Reinitialising config class with parameters (that were none initially but populated on first init)
            # causes the 2nd init to fail as there are mutual exclusive checks on pairs of parameters.
            # Refer: https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py#L499-L502 for example
            # So we only create config class if the previous init was not TrainingArguments
            if not isinstance(training_args, TrainingArguments):
                config = config_class(**config_dict)
            else:
                config = training_args

            # Reconstruct kwargs for Trainer
            kwargs = trainer_kwargs
            kwargs["args"] = config
        pass
        original_init(self, *args, **kwargs)
    pass
    return new_init
pass


def _patch_trl_trainer():
    import trl
    if hasattr(trl, "__UNSLOTH_BACKWARDS_COMPATIBLE__"): return
    if Version(transformers_version) <= Version("0.11.0"): return

    import trl.trainer
    trl_classes = dir(trl.trainer)
    trl_trainers = set(x[:-len("Trainer")] for x in trl_classes if x.endswith("Trainer"))
    trl_configs  = set(x[:-len("Config")]  for x in trl_classes if x.endswith("Config"))
    trl_classes = list(trl_trainers & trl_configs)

    for x in trl_classes:
        try:    exec(f"trl.{x}Trainer.__init__ = _backwards_compatible_trainer(trl.{x}Trainer, trl.{x}Config)", globals())
        except: continue
    pass

    trl.__UNSLOTH_BACKWARDS_COMPATIBLE__ = True
pass
