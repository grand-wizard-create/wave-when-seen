"""Wave when seen: the arm waves when someone steps in front of the camera.

Runs on the SO-101 follower arm with a camera, until it is stopped. Every second tick it looks at
the camera's newest frame. When enough of the picture changes for a few frames in a row, it runs
Wave hello's wave (copied below: ``rise``, ``wave``, ``back``, at the same speed limits), then goes
back to watching. It waves at most once every 10 seconds.

Every move is streamed, one small step per tick, so no joint ever jumps: glides go at most 45
degrees a second and the wave peaks just under that. Between waves the arm holds near where it
started, 10 degrees inside its limits; when the app stops, the arm's own slow fold to rest does
the last part.
"""

from __future__ import annotations

import logging
import math
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
from monorobot.sdk.app import App

log = logging.getLogger("wave-when-seen")

SO101 = frozenset({"hf_so101_follower", "sim_so101_follower"})
JOINTS: tuple[str, ...] = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

DETECT_EVERY_TICKS = 2       # at 30 Hz, 15 frames a second
COOLDOWN_S = 10.0            # at most one wave this often, counted from the end of the last one

# Wave hello 0.1.0's wave, copied (one file per app).
# Where the arm says hello, in degrees.
HELLO_DEG: dict[str, float] = {
    "shoulder_pan": 0.0, "shoulder_lift": -40.0, "elbow_flex": 45.0, "wrist_flex": -15.0, "wrist_roll": 0.0,
}
# The wave, per joint: (amplitude in degrees, phase in periods) of a sine around HELLO_DEG.
# 15 degrees over PERIOD_S peaks at 43 degrees a second.
WAVE_DEG: dict[str, tuple[float, float]] = {"wrist_flex": (15.0, 0.0)}
PERIOD_S = 2.2
WAVES = 3
GRIPPER_OPEN = 0.6           # 0 closed, 1 open
GLIDE_DEG_PER_S = 45.0

# Every target stays 10 degrees inside the follower's URDF limits, and the base turns at most 60
# degrees either way.
ENVELOPE_DEG: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-60.0, 60.0),
    "shoulder_lift": (-90.0, 90.0),
    "elbow_flex": (-86.8, 86.8),
    "wrist_flex": (-85.0, 85.0),
    "wrist_roll": (-147.0, 152.0),
}


def clip_to_envelope(joints_deg: Sequence[float]) -> list[float]:
    """The pose held inside ``ENVELOPE_DEG``, joint by joint."""
    return [min(max(float(v), ENVELOPE_DEG[name][0]), ENVELOPE_DEG[name][1]) for name, v in zip(JOINTS, joints_deg)]


def wave_pose(t: float) -> list[float]:
    """The arm's pose ``t`` seconds into the wave, in degrees."""
    pose = [HELLO_DEG[name] for name in JOINTS]
    for name, (amplitude, phase) in WAVE_DEG.items():
        pose[JOINTS.index(name)] += amplitude * math.sin(2.0 * math.pi * (t / PERIOD_S + phase))
    return pose


class Glide:
    """One target per tick from ``start_deg`` to ``goal_deg``, on a straight line in joint space.

    Lasts ``max(min_s, largest joint gap / max_deg_per_s)``, so no joint moves faster than
    ``max_deg_per_s``. The gripper, given as ``(from, to)``, ramps over the same ticks.
    """

    def __init__(
        self,
        start_deg: Sequence[float],
        goal_deg: Sequence[float],
        max_deg_per_s: float,
        rate_hz: float,
        min_s: float = 1.0,
        *,
        gripper: tuple[float, float] | None = None,
    ) -> None:
        self.start = [float(v) for v in start_deg]
        self.goal = [float(v) for v in goal_deg]
        gap = max(abs(g - s) for s, g in zip(self.start, self.goal))
        self.duration_s = max(min_s, gap / max_deg_per_s)
        self.ticks = max(1, math.ceil(self.duration_s * rate_hz - 1e-9))
        self.gripper = gripper
        self.i = 0

    @property
    def done(self) -> bool:
        return self.i >= self.ticks

    def next(self) -> tuple[list[float], float | None]:
        """The next target: ``(joints_deg, gripper)``, the gripper None when the glide has none."""
        self.i = min(self.i + 1, self.ticks)
        f = self.i / self.ticks
        joints = [s + (g - s) * f for s, g in zip(self.start, self.goal)]
        if self.gripper is None:
            return joints, None
        a, b = self.gripper
        return joints, a + (b - a) * f


class Wave:
    """Rise to the hello pose, wave, then glide back: one ``(joints_deg, gripper)`` target per tick.

    ``phase`` goes ``rise``, ``wave``, ``back``, then ``done``, when ``step()`` returns None.
    """

    def __init__(
        self,
        start_deg: Sequence[float],
        start_gripper: float,
        rate_hz: float,
        *,
        back_deg: Sequence[float],
        back_gripper: float,
    ) -> None:
        self.rate_hz = rate_hz
        self.back_deg = list(back_deg)
        self.back_gripper = back_gripper
        self.phase = "rise"
        self.wave_ticks = round(WAVES * PERIOD_S * rate_hz)
        self._i = 0
        self._last = list(start_deg)
        self._glide = Glide(start_deg, wave_pose(0.0), GLIDE_DEG_PER_S, rate_hz,
                            gripper=(start_gripper, GRIPPER_OPEN))

    def step(self) -> tuple[list[float], float] | None:
        if self.phase == "rise":
            if not self._glide.done:
                return self._keep(self._glide.next())
            self.phase = "wave"
        if self.phase == "wave":
            if self._i < self.wave_ticks:
                t = self._i / self.rate_hz
                self._i += 1
                return self._keep((wave_pose(t), GRIPPER_OPEN))
            self.phase = "back"
            self._glide = Glide(self._last, self.back_deg, GLIDE_DEG_PER_S, self.rate_hz,
                                gripper=(GRIPPER_OPEN, self.back_gripper))
        if self.phase == "back":
            if not self._glide.done:
                return self._keep(self._glide.next())
            self.phase = "done"
        return None

    def _keep(self, target: tuple[list[float], float | None]) -> tuple[list[float], float]:
        joints, gripper = target
        self._last = joints
        return joints, float(gripper if gripper is not None else GRIPPER_OPEN)


def newest_frame(camera: Any, last_seq: int | None) -> tuple[Any, int | None]:
    """``(frame, seq)`` for the camera's newest frame not seen yet, else ``(None, last_seq)``.

    A camera in its own process (the default) shares its frames, numbered; one in the app's process
    is read with ``get_frame()``.
    """
    latest = getattr(camera, "latest_frame", None)
    if callable(latest):
        frame, seq = latest()
        if frame is None or seq == last_seq:
            return None, last_seq
        return frame, seq
    return camera.get_frame(), None


class MotionDetector:
    """Fires when enough of the picture changes for a few frames in a row.

    Frames are reduced to gray (the mean over channels; a 2-D frame, such as an infrared view, is
    used as is) and downscaled by striding to about ``width`` pixels wide. A pixel changed when it
    differs from a running background (``bg = 0.9 bg + 0.1 frame``) by more than ``threshold``.
    ``feed()`` returns True when the changed fraction is over ``fraction`` for ``frames`` frames
    in a row; the count then starts again.
    """

    def __init__(self, width: int = 64, threshold: float = 25.0, fraction: float = 0.08, frames: int = 5) -> None:
        self.width = width
        self.threshold = threshold
        self.fraction = fraction
        self.frames = frames
        self.reset()

    def reset(self) -> None:
        """Forget the background: the next frame becomes it."""
        self.background: np.ndarray | None = None
        self.streak = 0

    def feed(self, frame: Any) -> bool:
        image = np.asarray(frame)
        if image.ndim not in (2, 3) or image.size == 0:
            raise ValueError(f"a camera frame must be a 2-D or 3-D array, got shape {image.shape}")
        step = max(1, round(image.shape[1] / self.width))
        small = image[::step, ::step].astype(np.float32)
        if small.ndim == 3:
            small = small.mean(axis=2)
        if self.background is None or self.background.shape != small.shape:
            self.background = small
            self.streak = 0
            return False
        changed = float(np.mean(np.abs(small - self.background) > self.threshold))
        self.background = 0.9 * self.background + 0.1 * small
        self.streak = self.streak + 1 if changed > self.fraction else 0
        if self.streak >= self.frames:
            self.streak = 0
            return True
        return False


class WaveWhenSeen(App):
    def setup(self) -> None:
        if self.config.robot.type not in SO101:
            raise RuntimeError("Wave when seen runs on the SO-101 arm.")
        self.camera = (getattr(self.robot, "devices", None) or {}).get("camera")
        if self.camera is None:
            raise RuntimeError("Wave when seen needs a camera. Add one to this robot in My robots.")
        start = [math.degrees(v) for v in self.robot.get_joint_positions()]
        gripper = min(1.0, max(0.0, float(self.robot.gripper_get_width())))
        self.home_deg = clip_to_envelope(start)
        self.home_gripper = gripper
        self.last_deg = start            # the last commanded pose: the next wave starts from it
        self.last_gripper = gripper
        self.detector = MotionDetector()
        self.wave: Wave | None = None
        self.waves = 0
        self.last_wave_end: int | None = None     # the tick the last wave ended on
        self._seq: int | None = None
        log.info("Watching for someone.")

    @property
    def phase(self) -> str:
        return self.wave.phase if self.wave is not None else "watch"

    def loop(self) -> None:
        if self.wave is not None:
            target = self.wave.step()
            if target is not None:
                self._send(target)
                return
            self.wave = None
            self.last_wave_end = self.tick
            self.detector.reset()
            log.info("Watching again.")
        if self.tick % DETECT_EVERY_TICKS:
            return
        frame, self._seq = newest_frame(self.camera, self._seq)
        if frame is None or not self.detector.feed(frame):
            return
        if self.last_wave_end is not None and (self.tick - self.last_wave_end) / self.rate_hz < COOLDOWN_S:
            return
        log.info("Someone's here: waving.")
        self.waves += 1
        self.wave = Wave(self.last_deg, self.last_gripper, self.rate_hz,
                         back_deg=self.home_deg, back_gripper=self.home_gripper)
        target = self.wave.step()
        if target is not None:
            self._send(target)

    def on_shutdown(self) -> None:
        if getattr(self, "waves", 0):
            log.info("Waved %d %s.", self.waves, "time" if self.waves == 1 else "times")

    def _send(self, target: tuple[list[float], float]) -> None:
        joints, gripper = target
        self.robot.send_teleop_action([math.radians(v) for v in joints], gripper)
        self.last_deg, self.last_gripper = joints, gripper


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(WaveWhenSeen.from_config("robot.yaml").run())
