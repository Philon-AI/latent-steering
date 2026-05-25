"""
...

Usage:
    python eval.py --checkpoint <dcp_dir>
"""

import argparse
import logging
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

from utils.config import load_config
from data.dataset import LatentSteeringDataset
from data.collator import collate_batch
from model.latent_steering import LatentSteering
from training.loss import WeightedLoss


_MOUSE_POSITION_DELTA_NAMES = ["dx", "dy"]
_MOUSE_BUTTON_NAMES = ["attack", "use"]
_KEYBOARD_BUTTON_NAMES = [
    "forward", "left", "back", "right", "jump", "sneak", "sprint", "inventory", "drop",
    "hotbar_1", "hotbar_2", "hotbar_3", "hotbar_4", "hotbar_5", "hotbar_6", "hotbar_7", "hotbar_8", "hotbar_9",
]

ACTION_LAYOUT = (
    [("mouse_delta", n, "continuous") for n in _MOUSE_POSITION_DELTA_NAMES] + 
    [("mouse_buttons", n, "discrete") for n in _MOUSE_BUTTON_NAMES] + 
    [("keyboard_buttons", n, "discrete") for n in _KEYBOARD_BUTTON_NAMES]
)


def _load_dcp_checkpoint(checkpoint_dir: Path) -> dict:
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        dcp_to_torch_save(str(checkpoint_dir), tmp.name)
        state = torch.load(tmp.name, weights_only=True, map_location="cpu")
    return state["model_state"]

def _continuous_metrics(pred: torch.Tensor, target: torch.Tensor):
    mse = ((pred - target) ** 2).mean(0)
    mae = (pred - target).abs().mean(0)
    target_std = target.std(0)
    return {"mse": mse, "mae": mae, "target_std": target_std}

def _discrete_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    p = (pred >= threshold).float()
    t = (target >= 0.5).float()
    tp = (p * t).sum(0)
    fp = (p * (1 - t)).sum(0)
    fn = ((1 - p) * t).sum(0)
    tn = ((1 - p) * (1 - t)).sum(0)
    eps = 1e-9
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    pred_rate = p.mean(0)
    target_rate = t.mean(0)
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "pred_rate": pred_rate, "target_rate": target_rate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a DCP checkpoint directory to load the model weights from.")
    parser.add_argument("--split", type=str, required=True, help="The dataset split to run the evaluation on.")
    parser.add_argument("--batch-size", type=int, default=256, help="Number of samples per batch during the forward pass.")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of subprocesses to use for parallel data loading.")
    parser.add_argument("--max-batches", type=int, default=None, help="Stop evaluation after this many batches (useful for quick debugging).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("Loading config from %s", args.config)
    config = load_config(args.config)

    seed = config["seed"]
    logger.info("Setting random seed %d", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dataset = LatentSteeringDataset(config, split=args.split)
    logger.info("Loaded %s split: %d samples", args.split, len(dataset))

    logger.info("Initializing model")
    model = LatentSteering(config).cuda()

    logger.info("Loading DCP checkpoint from %s", args.checkpoint)
    state_dict = _load_dcp_checkpoint(Path(args.checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys: %d (first 3: %s)", len(missing), missing[:3])
    if unexpected:
        logger.warning("Unexpected keys: %d (first 3: %s)", len(unexpected), unexpected[:3])

    model.eval()

    loss_weights = torch.as_tensor(
        config["training"]["loss"]["weights"], dtype=torch.float32
    )
    loss_fn = WeightedLoss(weights_tensor=loss_weights).cuda()

    layout = ACTION_LAYOUT
    assert len(layout) == config["model"]["action_dim"], (
        f"ACTION_LAYOUT has {len(layout)} entries but config action_dim={config['model']['action_dim']}"
    )
    assert len(loss_weights) == len(layout), (
        f"loss weights length {len(loss_weights)} does not match action layout length {len(layout)}"
    )
    continuous_idx = [i for i, (_, _, t) in enumerate(layout) if t == "continuous"]
    discrete_idx = [i for i, (_, _, t) in enumerate(layout) if t == "discrete"]

    num_vision_tokens = config["model"]["attentive_pooler_past_cfg"]["num_queries"]
    num_steering_tokens = config["model"]["attentive_pooler_future_cfg"]["num_queries"]
    action_horizon = config["model"]["action_horizon"]

    collate_fn = partial(
        collate_batch,
        num_vision_tokens=num_vision_tokens,
        num_steering_tokens=num_steering_tokens,
        action_horizon=action_horizon,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.num_workers > 0,
        shuffle=False,
        drop_last=False,
    )

    action_dim = config["model"]["action_dim"]

    weighted_loss_sum = 0.0
    total_elements = 0
    per_dim_loss_sum = torch.zeros(action_dim, device="cuda")
    per_dim_count = 0
    per_t_loss_sum = torch.zeros(action_horizon, device="cuda")
    per_t_count = 0

    pred_flat_chunks: list[torch.Tensor] = []
    targ_flat_chunks: list[torch.Tensor] = []
    pred_per_t_sq_sum = torch.zeros(action_horizon, device="cuda")
    pred_per_t_count = torch.zeros(action_horizon, device="cuda")

    n_samples = 0
    n_batches = 0

    for batch in dataloader:
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch)

        raw_loss = output["raw_loss"].float()

        weighted_loss_sum += loss_fn(raw_loss).item() * raw_loss.numel()
        total_elements += raw_loss.numel()
        per_dim_loss_sum += raw_loss.sum(dim=(0, 1))
        per_dim_count += raw_loss.shape[0] * raw_loss.shape[1]
        per_t_loss_sum += raw_loss.sum(dim=(0, 2))
        per_t_count += raw_loss.shape[0] * raw_loss.shape[2]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model.get_action(batch)["action_tensor"].float()

        actions = batch["future_actions"].float()
        pred_flat_chunks.append(pred.reshape(-1, action_dim).cpu())
        targ_flat_chunks.append(actions.reshape(-1, action_dim).cpu())
        sq = ((pred - actions) ** 2).mean(dim=(0, 2))
        pred_per_t_sq_sum += sq * pred.shape[0]
        pred_per_t_count += pred.shape[0]

        n_samples += batch["future_actions"].shape[0]
        n_batches += 1

        if n_batches % 10 == 0:
            running = weighted_loss_sum / max(total_elements, 1)
            logger.info("Batch %d (%d samples) | running weighted loss=%.4f",
                n_batches, n_samples, running)

        if args.max_batches and n_batches >= args.max_batches:
            break

    weighted_avg = weighted_loss_sum / max(total_elements, 1)
    per_dim_mean = (per_dim_loss_sum / max(per_dim_count, 1)).cpu()
    per_t_mean = (per_t_loss_sum / max(per_t_count, 1)).cpu()
    unweighted_avg = per_dim_mean.mean().item()

    sep = "=" * 72
    logger.info(sep)
    logger.info("Evaluation: split=%s  batches=%d  samples=%d", args.split, n_batches, n_samples)
    logger.info(sep)
    logger.info("Velocity MSE (training objective)")
    logger.info("  weighted (loss_fn):        %.6f", weighted_avg)
    logger.info("  unweighted (mean of dims): %.6f", unweighted_avg)

    logger.info("")
    logger.info("Velocity MSE per action group:")
    for group in ["mouse_delta", "mouse_buttons", "keyboard_buttons"]:
        idx = [i for i, (g, _, _) in enumerate(layout) if g == group]
        group_mean = per_dim_mean[idx].mean().item()
        logger.info("  %-16s (%2d dims): %.6f", group, len(idx), group_mean)

    logger.info("")
    logger.info("Velocity MSE per dimension:")
    logger.info("  %-4s %-16s %-16s %10s %10s", "idx", "group", "name", "mse", "loss_weight")
    for i, (g, n, _) in enumerate(layout):
        logger.info("  %-4d %-16s %-16s %10.6f %10.3f",
            i, g, n, per_dim_mean[i].item(), loss_weights[i].item())

    logger.info("")
    logger.info("Velocity MSE per horizon step:")
    for t in range(action_horizon):
        logger.info("  t=%d: %.6f", t, per_t_mean[t].item())

    pred_flat = torch.cat(pred_flat_chunks, dim=0)
    targ_flat = torch.cat(targ_flat_chunks, dim=0)

    logger.info("")
    logger.info(sep)
    logger.info("Sampled actions (iterative denoiser, %d steps)",
        model.config.num_inference_timesteps)
    logger.info(sep)

    if continuous_idx:
        cm = _continuous_metrics(pred_flat[:, continuous_idx], targ_flat[:, continuous_idx])
        logger.info("Continuous dims (range ~[0,1] after mu-law):")
        logger.info("  %-4s %-16s %-16s %10s %10s %10s",
            "idx", "group", "name", "mse", "mae", "target_std")
        for k, i in enumerate(continuous_idx):
            g, n, _ = layout[i]
            logger.info("  %-4d %-16s %-16s %10.6f %10.6f %10.6f",
                i, g, n, cm["mse"][k].item(), cm["mae"][k].item(), cm["target_std"][k].item())

    if discrete_idx:
        bp = pred_flat[:, discrete_idx]
        bt = targ_flat[:, discrete_idx]
        m = _discrete_metrics(bp, bt, threshold=0.5)
        logger.info("")
        logger.info("Discrete dims:")
        logger.info("  %-4s %-16s %-16s %8s %8s %8s %8s %8s %8s",
            "idx", "group", "name", "acc", "prec", "rec", "f1", "pred%", "true%")
        for k, i in enumerate(discrete_idx):
            g, n, _ = layout[i]
            logger.info("  %-4d %-16s %-16s %8.3f %8.3f %8.3f %8.3f %7.2f%% %7.2f%%",
                i, g, n,
                m["accuracy"][k].item(), m["precision"][k].item(),
                m["recall"][k].item(), m["f1"][k].item(),
                100 * m["pred_rate"][k].item(), 100 * m["target_rate"][k].item())

        support = m["target_rate"]
        macro_f1 = m["f1"].mean().item()
        weighted_f1 = (m["f1"] * support).sum().item() / max(support.sum().item(), 1e-9)
        logger.info("  macro F1: %.4f   support-weighted F1: %.4f", macro_f1, weighted_f1)

    per_t_mse = (pred_per_t_sq_sum / pred_per_t_count.clamp(min=1)).cpu()
    logger.info("")
    logger.info("Sampled action MSE per horizon step (avg over dims):")
    for t in range(action_horizon):
        logger.info("  t=%d: %.6f", t, per_t_mse[t].item())


if __name__ == "__main__":
    main()
