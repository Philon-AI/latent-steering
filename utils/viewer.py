import threading
import time

import cv2
import torch

import foxglove
from foxglove.messages import CompressedImage, Timestamp
from foxglove._foxglove_py.websocket import (
    PlaybackCommand,
    PlaybackControlRequest,
    PlaybackState,
    PlaybackStatus,
)


MOUSE_POSITION_DELTA_NAMES = ["X", "Y"]
MOUSE_BUTTON_NAMES = ["Attack", "Use"]
KEYBOARD_BUTTON_NAMES = [
    "Forward", "Left", "Back", "Right", "Jump", "Sneak", "Sprint", "Inventory", "Drop",
    "Hotbar1", "Hotbar2", "Hotbar3", "Hotbar4", "Hotbar5", "Hotbar6", "Hotbar7", "Hotbar8", "Hotbar9",
]

MOUSE_POSITION_DELTA_SCHEMA = {
    "type": "object", "title": "MousePositionDelta",
    "properties": {n: {"type": "number"} for n in MOUSE_POSITION_DELTA_NAMES},
}
MOUSE_BUTTONS_SCHEMA = {
    "type": "object", "title": "MouseButtons",
    "properties": {b: {"type": "boolean"} for b in MOUSE_BUTTON_NAMES},
}
KEYBOARD_BUTTONS_SCHEMA = {
    "type": "object", "title": "KeyboardButtons",
    "properties": {b: {"type": "boolean"} for b in KEYBOARD_BUTTON_NAMES},
}


def frame_to_ns(frame: int, fps: float) -> int:
    return int(frame / fps * 1e9)

def ns_to_frame(ns: int, fps: float) -> int:
    return int(ns * fps / 1e9)

def ns_to_timestamp(ns: int) -> Timestamp:
    return Timestamp(ns // 1_000_000_000, ns % 1_000_000_000)


class LatentSteeringViewer:
    def __init__(self, sample: dict, fps: float, port: int = 8765, jpeg_quality: int = 90):
        self.fps = fps
        self.port = port

        images = torch.cat([sample["observation.images.cam.past"], sample["observation.images.cam.future"]], dim=0)
        mouse_position_delta = torch.cat([sample["action.mouse.position_delta.past"], sample["action.mouse.position_delta.future"]], dim=0)
        mouse_buttons = torch.cat([sample["action.mouse.buttons.past"], sample["action.mouse.buttons.future"]], dim=0)
        keyboard_buttons = torch.cat([sample["action.keyboard.buttons.past"], sample["action.keyboard.buttons.future"]], dim=0)

        self.total_frames = len(images)
        self.duration_ns = frame_to_ns(self.total_frames, fps)

        self.frames = []
        for i in range(self.total_frames):
            img = cv2.cvtColor(images[i].permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)
            _, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            self.frames.append(jpg.tobytes())

        self.mouse_position_delta = mouse_position_delta.cpu().numpy()
        self.mouse_buttons = mouse_buttons.cpu().numpy().astype(bool)
        self.keyboard_buttons = keyboard_buttons.cpu().numpy().astype(bool)
        
        self.playing = False
        self.speed = 1.0

        self.current_ns = 0
        self._seek_ns: int | None = None

        self._lock = threading.Lock()
        self._wake = threading.Event()

        self.video_chan: foxglove.Channel | None = None
        self.mouse_position_delta_chan: foxglove.Channel | None = None
        self.mouse_buttons_chan: foxglove.Channel | None = None
        self.keyboard_buttons_chan: foxglove.Channel | None = None

    def _on_playback_control(self, req: PlaybackControlRequest) -> PlaybackState:
        with self._lock:
            if req.seek_time is not None:
                self._seek_ns = req.seek_time
            if req.playback_command == PlaybackCommand.Play:
                self.playing = True
            elif req.playback_command == PlaybackCommand.Pause:
                self.playing = False
            if req.playback_speed is not None:
                self.speed = req.playback_speed

            state = PlaybackState(
                status=PlaybackStatus.Playing if self.playing else PlaybackStatus.Paused,
                current_time=self._seek_ns if self._seek_ns is not None else self.current_ns,
                playback_speed=self.speed,
                did_seek=self._seek_ns is not None,
                request_id=req.request_id,
            )

        self._wake.set()
        return state

    def _broadcast_frame(self, server, idx: int) -> None:
        ts_ns = frame_to_ns(idx, self.fps)
        self.current_ns = ts_ns

        self.video_chan.log(
            CompressedImage(
                timestamp=ns_to_timestamp(ts_ns),
                frame_id="video",
                data=self.frames[idx],
                format="jpeg",
            ).encode(),
            log_time=ts_ns,
        )
        self.mouse_position_delta_chan.log(
            {n: float(v) for n, v in zip(MOUSE_POSITION_DELTA_NAMES, self.mouse_position_delta[idx])},
            log_time=ts_ns,
        )
        self.mouse_buttons_chan.log(
            {n: bool(v) for n, v in zip(MOUSE_BUTTON_NAMES, self.mouse_buttons[idx])},
            log_time=ts_ns,
        )
        self.keyboard_buttons_chan.log(
            {n: bool(v) for n, v in zip(KEYBOARD_BUTTON_NAMES, self.keyboard_buttons[idx])},
            log_time=ts_ns,
        )

        server.broadcast_time(ts_ns)

    def _stream_loop(self, server) -> None:
        timeline_idx = 0

        while True:
            with self._lock:
                seek_ns = self._seek_ns
                self._seek_ns = None
                playing = self.playing
                speed = self.speed

            if seek_ns is not None:
                timeline_idx = max(0, min(ns_to_frame(seek_ns, self.fps), self.total_frames - 1))
                self._broadcast_frame(server, timeline_idx)

            if not playing:
                self._wake.wait()
                self._wake.clear()
                continue

            self._wake.clear()

            if timeline_idx >= self.total_frames:
                with self._lock:
                    self.playing = False
                server.broadcast_playback_state(PlaybackState(
                    status=PlaybackStatus.Ended,
                    current_time=self.duration_ns,
                    playback_speed=speed,
                    did_seek=False,
                    request_id=None,
                ))
                self._wake.wait()
                self._wake.clear()
                continue

            self._broadcast_frame(server, timeline_idx)
            timeline_idx += 1

            deadline = time.monotonic() + 1.0 / (self.fps * speed)
            while time.monotonic() < deadline:
                if self._wake.wait(timeout=min(0.005, deadline - time.monotonic())):
                    self._wake.clear()
                    break

    def run(self) -> None:
        viewer = self

        class Listener:
            def on_subscribe(self, client, channel): ...
            def on_unsubscribe(self, client, channel): ...
            def on_playback_control_request(self, req):
                return viewer._on_playback_control(req)

        server = foxglove.start_server(
            name="Latent Steering Viewer",
            host="0.0.0.0",
            port=self.port,
            capabilities=[foxglove.Capability.Time],
            server_listener=Listener(),
            playback_time_range=(0, self.duration_ns),
        )

        self.video_chan = foxglove.Channel(
            "/video", schema=CompressedImage.get_schema(), message_encoding="protobuf")
        self.mouse_position_delta_chan = foxglove.Channel("/mouse_position_delta", schema=MOUSE_POSITION_DELTA_SCHEMA)
        self.mouse_buttons_chan = foxglove.Channel("/mouse_buttons", schema=MOUSE_BUTTONS_SCHEMA)
        self.keyboard_buttons_chan = foxglove.Channel("/keyboard_buttons", schema=KEYBOARD_BUTTONS_SCHEMA)

        print(f"Total frames: {self.total_frames} ({self.fps} fps)")
        print(f"\nServing on ws://localhost:{self.port}")
        url = server.app_url()
        if url:
            print(f"Or open: {url}")

        server.broadcast_time(0)
        self._broadcast_frame(server, 0)

        try:
            self._stream_loop(server)
        except KeyboardInterrupt:
            pass
        finally:
            for ch in (self.video_chan, self.mouse_position_delta_chan, self.mouse_buttons_chan, self.keyboard_buttons_chan):
                if ch:
                    ch.close()
            server.stop()
