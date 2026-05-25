"""
...

Usage:
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 train.py
"""

import os
import argparse
import faulthandler
import gc
import logging
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed._composable import checkpoint
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Enable TF32 for faster training on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from transformers.optimization import get_cosine_schedule_with_warmup

import wandb

logger = logging.getLogger(__name__)

from utils.config import load_config
from data.dataset import LatentSteeringDataset
from data.collator import collate_batch
from model.latent_steering import LatentSteering
from training.checkpoint import load_checkpoint, save_checkpoint
from training.fsdp import get_fsdp_shard_modules, get_fsdp_checkpointing_modules
from training.loss import WeightedLoss
from training.optimizer import split_param_groups
from training.sampler import DistributedSampler


def main():
    """Main training loop for latent-steering model."""
    parser = argparse.ArgumentParser(description="Train latent-steering model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from last checkpoint")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    logger.info("Rank %d/%d (local_rank=%d)", global_rank, world_size, local_rank)
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()

    if global_rank != 0:
        logging.disable(logging.INFO)

    config = load_config(args.config)

    resume = args.resume
    resume_path = args.resume_path

    seed = config["seed"]
    logger.info("Setting random seed %d (base=%d)", seed + global_rank, seed)
    np.random.seed(seed + global_rank)
    torch.manual_seed(seed + global_rank)
    torch.cuda.manual_seed_all(seed + global_rank)

    timestamp = int(datetime.now().timestamp())
    timestamp = torch.tensor(timestamp, device="cuda")
    dist.broadcast(timestamp, src=0)

    output_dir = Path(config["training"]["output_dir"])
    output_name = datetime.fromtimestamp(timestamp.item()).strftime("%Y-%m-%d_%H-%M-%S")
    output_path = output_dir / output_name
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_path)

    dist.barrier()

    dataset = LatentSteeringDataset(config)
    logger.info("Dataset loaded: %d samples", len(dataset))

    logger.info("Initializing model")
    model = LatentSteering(config)

    loss_weights = torch.as_tensor(
        config["training"]["loss"]["weights"], dtype=torch.float32
    )
    loss_fn = WeightedLoss(weights_tensor=loss_weights).cuda()

    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    num_nodes = world_size // local_world_size

    mesh = init_device_mesh("cuda", (num_nodes, local_world_size) if num_nodes > 1 else (local_world_size,),
        mesh_dim_names=("replicate", "shard") if num_nodes > 1 else ("shard",))

    gradient_checkpointing = config["training"]["gradient_checkpointing"]
    if gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        checkpointing_modules = get_fsdp_checkpointing_modules()
        for module in model.modules():
            if isinstance(module, tuple(checkpointing_modules)):
                checkpoint(module)

    reshard_after_forward = config["training"]["distributed"]["reshard_after_forward"]

    fsdp_shard_modules = get_fsdp_shard_modules()
    for module in reversed(list(model.modules())):
        if isinstance(module, tuple(fsdp_shard_modules)):
            fully_shard(module, mesh=mesh, reshard_after_forward=reshard_after_forward)
    fully_shard(model, mesh=mesh, reshard_after_forward=reshard_after_forward)
    logger.info("Model wrapped with FSDP with mesh=%s, reshard_after_forward=%s",
        str(mesh), reshard_after_forward)

    dist.barrier()

    batch_size = config["training"]["dataloader"]["batch_size"]
    effective_batch_size = config["training"]["dataloader"]["effective_batch_size"]

    grad_accumulation_steps = effective_batch_size // (batch_size * world_size)

    lr = config["training"]["optimizer"]["lr"]
    weight_decay = config["training"]["optimizer"]["weight_decay"]
    betas = tuple(config["training"]["optimizer"]["betas"])

    param_groups = split_param_groups(model, lr, weight_decay)
    optimizer = AdamW(param_groups, betas=betas)
    logger.info("Optimizer: AdamW, lr=%.2e, weight_decay=%.2e, betas=%s", lr, weight_decay, betas)

    num_epochs = config["training"]["num_epochs"]

    num_training_samples = (len(dataset) // effective_batch_size) * effective_batch_size
    num_training_steps = (num_training_samples * num_epochs) // effective_batch_size

    scheduler_type = config["training"]["scheduler"]["type"]
    scheduler_warmup_ratio = config["training"]["scheduler"]["warmup_ratio"]

    if scheduler_type == "cosine":
        warmup_steps = int(num_training_steps * scheduler_warmup_ratio)
        logger.info("Scheduler: %s, warmup_steps=%d, total_steps=%d",
            scheduler_type, warmup_steps, num_training_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, num_training_steps,
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    resume_epoch, resume_step = 0, 0
    if resume:
        if not resume_path:
            raise ValueError("--resume requires --resume_path or resume_path in config")
        logger.info("Resuming from checkpoint %s", resume_path)
        resume_epoch, resume_step = load_checkpoint(resume_path, model, optimizer, scheduler)
        logger.info("Resumed at epoch %d, step %d", resume_epoch, resume_step)

    model.train()

    steps_per_epoch = num_training_samples // effective_batch_size

    sampler_start_index = 0
    if resume:
        steps_into_epoch = resume_step - resume_epoch * steps_per_epoch
        sampler_start_index = max(0, steps_into_epoch) * grad_accumulation_steps * batch_size

    num_vision_tokens = config["model"]["attentive_pooler_past_cfg"]["num_queries"]
    num_steering_tokens = config["model"]["attentive_pooler_future_cfg"]["num_queries"]
    action_horizon = config["model"]["action_horizon"]

    collate_fn = partial(
        collate_batch,
        num_vision_tokens=num_vision_tokens,
        num_steering_tokens=num_steering_tokens,
        action_horizon=action_horizon,
    )

    num_workers = config["training"]["dataloader"]["num_workers"]
    prefetch_factor = config["training"]["dataloader"]["prefetch_factor"]
    def worker_init_fn(worker_id):
        worker_seed = seed + world_size + global_rank * num_workers + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
        seed=seed,
        start_index=sampler_start_index,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
        pin_memory=num_workers > 0,
        drop_last=True,
    )

    logger.info("Train dataloader ready: num_workers=%d, prefetch_factor=%s, steps_per_epoch=%d",
        num_workers, prefetch_factor, steps_per_epoch)

    if global_rank == 0:
        wandb_project = config["logging"]["wandb"]["project"]
        wandb_entity = config["logging"]["wandb"]["entity"]
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=output_name,
            config=config,
            resume="allow",
        )

    max_grad_norm = config["training"]["gradient_clip"]
    checkpoint_steps = config["training"]["checkpoint"]["num_steps"]

    logger.info("Starting training: %d epochs, %d steps, batch_size=%d, grad_accum=%d",
        num_epochs, num_training_steps, effective_batch_size, grad_accumulation_steps)

    step = resume_step
    for epoch in range(resume_epoch, num_epochs):
        logger.info("Starting epoch %d/%d", epoch + 1, num_epochs)

        sampler.set_epoch(epoch)
        if epoch > resume_epoch:
            sampler.start_index = 0

        optimizer.zero_grad()

        loss = torch.zeros((), device="cuda")
        micro_step = 0

        step_start_time = time.perf_counter()
        for batch in dataloader:
            output = model(batch)

            weighted_loss = loss_fn(output["raw_loss"])
            normalized_loss = weighted_loss / grad_accumulation_steps
            normalized_loss.backward()
            loss += normalized_loss.detach()

            micro_step += 1
            if micro_step % grad_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                step += 1

                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                loss_item = loss.item()

                if global_rank == 0:
                    step_elapsed = time.perf_counter() - step_start_time
                    samples_per_sec = effective_batch_size / step_elapsed

                    lr = scheduler.get_last_lr()[0]
                    logger.info("Step %d | loss=%.4f | lr=%.2e | %.1f samples/s",
                        step, loss_item, lr, samples_per_sec)
                    wandb.log({
                        "train/loss": loss_item,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm.item(),
                        "train/samples_per_sec": samples_per_sec,
                    }, step=step)

                loss.zero_()

                if step % checkpoint_steps == 0:
                    logger.info("Saving checkpoint at epoch %d, step %d", epoch, step)
                    save_checkpoint(output_path, epoch, step, model, optimizer, scheduler)

                step_start_time = time.perf_counter()

        logger.info("Epoch %d/%d complete at step %d", epoch + 1, num_epochs, step)
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Training complete at step %d", step)

    if global_rank == 0:
        wandb.finish()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    faulthandler.enable()

    if not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )

    main()
