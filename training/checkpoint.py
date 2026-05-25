import gc
import re
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)


def load_checkpoint(checkpoint_dir, model, optimizer, scheduler):
    checkpoint_dir = Path(checkpoint_dir)

    state_dict_options = StateDictOptions(full_state_dict=False)
    state_dict = {
        "model_state": get_model_state_dict(model, options=state_dict_options),
        "optimizer_state": get_optimizer_state_dict(model, optimizer, options=state_dict_options),
    }

    dcp.load(state_dict, checkpoint_id=str(checkpoint_dir))

    set_model_state_dict(model, state_dict["model_state"], options=state_dict_options)
    set_optimizer_state_dict(model, optimizer, state_dict["optimizer_state"], options=state_dict_options)

    scheduler.load_state_dict(
        torch.load(checkpoint_dir / "scheduler.pt", weights_only=True)
    )

    match = re.match(r"epoch_(\d+)_step_(\d+)", checkpoint_dir.name)
    epoch, step = int(match.group(1)), int(match.group(2))

    dist.barrier()
    return epoch, step


def save_checkpoint(output_dir, epoch, step, model, optimizer, scheduler):
    checkpoint_name = f"epoch_{epoch}_step_{step}"
    checkpoint_dir = Path(output_dir) / "checkpoints" / checkpoint_name

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state_dict_options = StateDictOptions(full_state_dict=False)
    state_dict = {
        "model_state": get_model_state_dict(model, options=state_dict_options),
        "optimizer_state": get_optimizer_state_dict(model, optimizer, options=state_dict_options),
    }

    gc.collect()
    torch.cuda.empty_cache()

    dcp.save(state_dict, checkpoint_id=str(checkpoint_dir))

    if dist.get_rank() == 0:
        torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")

    dist.barrier()
    return checkpoint_dir
