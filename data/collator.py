import torch

from model.latent_steering import _ACTION_TOKEN, _PAD_TOKEN, _VISION_TOKEN, _STEERING_TOKEN


def collate_batch(batch, num_vision_tokens, num_steering_tokens, action_horizon):
    batch_size = len(batch)

    past_images = torch.stack([
        s["observation.images.cam.past"] for s in batch
    ])

    future_images = torch.stack([
        s["observation.images.cam.future"] for s in batch
    ])

    # past_actions = torch.stack([torch.cat([
    #     s["action.mouse.position_delta.past"],
    #     s["action.mouse.buttons.past"],
    #     s["action.keyboard.buttons.past"]], dim=-1) for s in batch
    # ])

    future_actions = torch.stack([torch.cat([
        s["action.mouse.position_delta.future"],
        s["action.mouse.buttons.future"],
        s["action.keyboard.buttons.future"]], dim=-1) for s in batch
    ])

    mm_token_ids = torch.cat([
        torch.full((batch_size, num_vision_tokens), _VISION_TOKEN, dtype=torch.long),
        torch.full((batch_size, 1), _PAD_TOKEN, dtype=torch.long),
        torch.full((batch_size, num_steering_tokens), _STEERING_TOKEN, dtype=torch.long),
    ], dim=1)

    sa_token_ids = torch.full((batch_size, action_horizon), _ACTION_TOKEN, dtype=torch.long)

    return {
        "past_images": past_images,
        "future_images": future_images,
        # "past_actions": past_actions,
        "future_actions": future_actions,
        "mm_token_ids": mm_token_ids,
        "sa_token_ids": sa_token_ids,
    }
