import json
import logging
import math
import os
from collections import OrderedDict

import torch

logging.getLogger("lerobot").setLevel(logging.WARNING)

from lerobot.datasets import video_utils
from lerobot.datasets.dataset_reader import DatasetReader
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# Patch DatasetReader to build the absolute→relative index map directly from the loaded
# hf_dataset's "index" column instead of scanning per-episode parquet metadata. The upstream
# implementation is O(n_episodes) file reads at startup, which dominates init.
def _build_index_mapping(self):
    self._absolute_to_relative_idx = None
    if self.episodes is not None and self.hf_dataset is not None:
        indices = self.hf_dataset.data.column("index").to_numpy()
        self._absolute_to_relative_idx = dict(zip(indices.tolist(), range(len(indices))))

DatasetReader._build_index_mapping = _build_index_mapping


# Patch the global video decoder cache with an LRU-bounded variant. Upstream's cache grows
# unbounded across the dataset's videos, eventually exhausting file descriptors / RAM during
# long training runs.
class _BoundedVideoDecoderCache(video_utils.VideoDecoderCache):
    def __init__(self, max_size=256):
        super().__init__()
        self._cache = OrderedDict()
        self._max_size = max_size

    def get_decoder(self, video_path):
        decoder = super().get_decoder(video_path)
        with self._lock:
            self._cache.move_to_end(str(video_path))
            while len(self._cache) > self._max_size:
                _, (_, fh) = self._cache.popitem(last=False)
                fh.close()
        return decoder

video_utils._default_decoder_cache = _BoundedVideoDecoderCache()


# Patch torchcodec frame decoding to drop the per-frame Python loop with `.item()` calls
# (each one is a host/device sync), the list-comprehension `torch.stack`, and the float32
# `/255.0` cast. Returns the decoder's native uint8 tensor via a single advanced index.
def _decode_video_frames_torchcodec(
    video_path,
    timestamps,
    tolerance_s,
    log_loaded_timestamps=False,
    decoder_cache=None,
) -> torch.Tensor:
    if decoder_cache is None:
        decoder_cache = video_utils._default_decoder_cache

    decoder = decoder_cache.get_decoder(str(video_path))

    metadata = decoder.metadata
    average_fps = metadata.average_fps

    frame_indices = [round(ts * average_fps) for ts in timestamps]
    frames_batch = decoder.get_frames_at(indices=frame_indices)

    query_ts = torch.tensor(timestamps, dtype=torch.float32)
    loaded_ts = frames_batch.pts_seconds.to(torch.float32)

    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    if not is_within_tol.all():
        raise video_utils.FrameTimestampError(
            f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
            " It means that the closest frame that can be loaded from the video is too far away in time."
            " This might be due to synchronization issues with timestamps during data collection."
            " To be safe, we advise to ignore this item during training."
            f"\nqueried timestamps: {query_ts}"
            f"\nloaded timestamps: {loaded_ts}"
            f"\nvideo: {video_path}"
        )

    return frames_batch.data[argmin_]

video_utils.decode_video_frames_torchcodec = _decode_video_frames_torchcodec


CURSOR = torch.tensor([
  0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,
  0, 102,  42,   0,  85,   0,   0,   0,   0,   0,   0,   0,
  0, 166, 234, 117,   0, 191,   0,   0,   0,   0,   0,   0,
  0, 186, 210, 166, 154,   0, 127,   0,   0,   0,   0,   0,
  0, 216, 167,  54, 193, 135,   0, 191,   0,   0,   0,   0,
  0, 207, 181,   8,  78, 187, 132,   0, 127,   0,   0,   0,
  0, 199, 179,  30,  32,  79, 190, 135,   0, 191,   0,   0,
  0, 199, 183,  32,  53,  34,  87, 193, 154,   0,  85,   0,
  0, 199, 184,  39,  54,  54,  27,  76, 178, 124,   0, 127,
  0, 191, 187,  39,  58,  62, 171, 171, 214, 227,  89, 255,
  0, 207, 177,  79, 201,  87, 162, 172,  68, 102,  46,   0,
  0, 153, 218, 161, 198, 128, 112, 175,   0,   0,   0,   0,
  0, 123, 218, 101,   0, 185,  82, 174, 173,   0,   0,   0,
  0, 109,  92,   0, 255, 167, 179, 187, 105,   0,   0,   0,
  0,   0,   0, 127,   0,   0, 113,  77,   0, 255,   0,   0,
  0,   0, 255,   0,   0,   1,   0,   0,   0,   0,   0,   0,
], dtype=torch.float32).view(1, 16, 12)

CURSOR_ALPHA = torch.tensor([
  0,   0,   0,   2,   0,   0,   0,   0,   0,   0,   0,   0,
  0,  10,  36,   0,   3,   0,   0,   0,   0,   0,   0,   0,
  0,  26, 210,  50,   0,   4,   0,   0,   0,   0,   0,   0,
  0,  26, 255, 224,  48,   0,   4,   0,   0,   0,   0,   0,
  0,  33, 244, 255, 219,  49,   0,   4,   0,   0,   0,   0,
  0,  32, 248, 253, 255, 223,  50,   0,   4,   0,   0,   0,
  0,  32, 248, 255, 251, 255, 219,  49,   0,   4,   0,   0,
  0,  32, 248, 255, 255, 252, 255, 223,  48,   0,   3,   0,
  0,  32, 248, 255, 255, 255, 252, 255, 220,  49,   0,   2,
  0,  32, 248, 254, 254, 255, 254, 248, 255, 214,  37,   1,
  0,  32, 246, 255, 255, 254, 250, 124,  63,  50,  11,   0,
  0,  30, 255, 235, 175, 255, 255, 154,   0,   0,   0,   0,
  0,  31, 230,  88,  42, 227, 254, 239,  28,   0,   0,   0,
  0,  14,  55,   5,   1, 146, 255, 233,  29,   0,   1,   0,
  0,   0,   0,   2,   1,  27, 124,  76,   3,   1,   0,   0,
  0,   0,   2,   0,   1,   0,   0,   0,   1,   0,   0,   0,
], dtype=torch.float32).view(1, 16, 12) / 255.0

CAM_SCALER = 360.0 / 2400.0
CAM_MAXVAL = 10.0
CAM_MU = 10.0


class LatentSteeringDataset(LeRobotDataset):
    def __init__(self, config, split="train"):
        self.config = config["data"]

        dataset_dir = self.config["dataset_dir"]

        with open(os.path.join(dataset_dir, "meta", "info.json")) as f:
            info = json.load(f)

        episodes = list(range(*map(int, info["splits"][split].split(":"))))

        past_window_size = self.config["past_window_size"]
        future_window_size = self.config["future_window_size"]

        total_window = past_window_size + future_window_size

        sample_rate = self.config["sample_rate"]

        boundaries = [round(i * info["fps"] / sample_rate) for i in range(-past_window_size, future_window_size + 1)]
        self._slices = [(boundaries[i] - boundaries[0], boundaries[i + 1] - boundaries[0]) for i in range(total_window)]

        sparse_timestamps = [boundaries[i + 1] / info["fps"] for i in range(total_window)]
        dense_timestamps = [i / info["fps"] for i in range(boundaries[0], boundaries[-1])]

        delta_timestamps = {
            "observation.images.cam": sparse_timestamps,
            "action.mouse.position": sparse_timestamps,
            "action.mouse.position_delta": dense_timestamps,
            "action.mouse.buttons": dense_timestamps,
            "action.keyboard.buttons": dense_timestamps,
            "is_gui_open": sparse_timestamps,
        }

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        super().__init__(
            repo_id="local/latent-steering",
            root=dataset_dir,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
        )

        ep_rows = self.meta.episodes.select(episodes)
        lengths = torch.as_tensor(ep_rows["dataset_to_index"]) - torch.as_tensor(ep_rows["dataset_from_index"])
        offsets = torch.cat([torch.zeros(1, dtype=lengths.dtype), lengths.cumsum(0)])

        forward_window = max(0, max(idx[-1] for idx in self.reader.delta_indices.values()))
        backward_window = max(0, -min(idx[0] for idx in self.reader.delta_indices.values()))
        self._indices = torch.cat([
            torch.arange(offsets[i] + backward_window, offsets[i + 1] - forward_window)
            for i in range(len(lengths)) if lengths[i] > forward_window + backward_window
        ])

    def __len__(self):
        return len(self._indices)
    
    def _overlay_cursor(self, images, mouse_position, is_gui_open):
        images = images.clone()
        _, _, H, W = images.shape
        _, CH, CW = CURSOR.shape
        scale = H / 720
        for t in is_gui_open.nonzero(as_tuple=True)[0].tolist():
            x = int(mouse_position[t, 0].item() * scale)
            y = int(mouse_position[t, 1].item() * scale)
            y0, y1 = max(0, y), min(H, y + CH)
            x0, x1 = max(0, x), min(W, x + CW)
            if y0 >= y1 or x0 >= x1:
                continue
            cursor = CURSOR[:, y0 - y : y1 - y, x0 - x : x1 - x]
            alpha = CURSOR_ALPHA[:, y0 - y : y1 - y, x0 - x : x1 - x]
            images[t, :, y0:y1, x0:x1] = (
                images[t, :, y0:y1, x0:x1] * (1 - alpha) + cursor * alpha
            ).to(torch.uint8)
        return images

    def _normalize_mouse_position_delta(self, mouse_position_delta):
        scaled = mouse_position_delta * CAM_SCALER
        clipped = torch.clamp(scaled, -CAM_MAXVAL, CAM_MAXVAL)
        normalized = clipped / CAM_MAXVAL
        encoded = torch.sign(normalized) * torch.log1p(CAM_MU * normalized.abs()) / math.log1p(CAM_MU)
        return ((encoded + 1.0) / 2.0).to(torch.float32)

    def __getitem__(self, idx):
        item = super().__getitem__(self._indices[idx].item())

        images = item.pop("observation.images.cam")
        mouse_position = item.pop("action.mouse.position")
        is_gui_open = item.pop("is_gui_open").squeeze(-1).bool()

        if is_gui_open.any():
            images = self._overlay_cursor(images, mouse_position, is_gui_open)

        past_window_size = self.config["past_window_size"]
        future_window_size = self.config["future_window_size"]

        item["observation.images.cam.past"] = images[:past_window_size]
        item["observation.images.cam.future"] = images[-future_window_size:]

        mouse_position_delta_dense = item.pop("action.mouse.position_delta")
        mouse_position_delta = self._normalize_mouse_position_delta(
            torch.stack([mouse_position_delta_dense[a:b].sum(dim=0) for a, b in self._slices])
        )
        item["action.mouse.position_delta.past"] = mouse_position_delta[:past_window_size]
        item["action.mouse.position_delta.future"] = mouse_position_delta[-future_window_size:]

        mouse_buttons_dense = item.pop("action.mouse.buttons")
        mouse_buttons = torch.stack(
            [mouse_buttons_dense[a:b].amax(dim=0) for a, b in self._slices]
        ).to(torch.float32)
        item["action.mouse.buttons.past"] = mouse_buttons[:past_window_size]
        item["action.mouse.buttons.future"] = mouse_buttons[-future_window_size:]

        keyboard_buttons_dense = item.pop("action.keyboard.buttons")
        keyboard_buttons = torch.stack(
            [keyboard_buttons_dense[a:b].amax(dim=0) for a, b in self._slices]
        ).to(torch.float32)
        item["action.keyboard.buttons.past"] = keyboard_buttons[:past_window_size]
        item["action.keyboard.buttons.future"] = keyboard_buttons[-future_window_size:]

        return item
