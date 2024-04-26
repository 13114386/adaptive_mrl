from __future__ import unicode_literals, print_function, division
from typing import Optional, Union
from torch.optim import Optimizer
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION
from transformers.trainer_utils import SchedulerType


def get_scheduler_plus(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    num_cycles: Optional[int] = None,
):
    """
        Extend get_scheduler (https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py#L233)
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    if num_cycles and name == SchedulerType.COSINE_WITH_RESTARTS:
        return schedule_func(optimizer,
                             num_warmup_steps=num_warmup_steps,
                             num_training_steps=num_training_steps,
                             num_cycles=num_cycles)
    else:
        return schedule_func(optimizer,
                             num_warmup_steps=num_warmup_steps,
                             num_training_steps=num_training_steps)
