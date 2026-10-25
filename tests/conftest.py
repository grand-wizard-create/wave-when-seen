"""Test doubles and motion checks for Wave when seen.

``main.py`` is loaded by path. ``FakeArm`` keeps a joint state and records every target the app
commands, in order; ``assert_motion_ok`` checks those targets against the rules every seed app
follows (no jump over 3 degrees, at most 45 degrees a second per joint, inside the envelope and
the URDF limits, never ``move_joints`` on an SO-101). ``sim_follower`` is the real simulated
SO-101 follower, in this process, with its state in the test's temp folder; ``run_under_sdk`` runs
the app on one with the SDK's own ``run()``.

Run with the engine's Python: ``"$MR/.venv/bin/python" -m pytest -q``.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

APP_DIR = Path(__file__).resolve().parents[1]
APP_CLASS = "WaveWhenSeen"

ARM_JOINTS: tuple[str, ...] = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
REST_DEG: tuple[float, ...] = (0.0, -98.0, 94.0, -94.0, 0.0)     # the follower's folded rest pose
MID_DEG: tuple[float, ...] = (10.0, -20.0, 30.0, 10.0, 5.0)

MAX_STEP_DEG = 3.0          # D37: no target more than 3 degrees from the last one
MAX_DEG_PER_S = 45.0        # D37: big moves stream at most 45 degrees a second
ENVELOPE_MARGIN_DEG = 10.0  # inside the URDF limits
PAN_LIMIT_DEG = 60.0
EPS = 1e-9

SIM_SPEEDUP = 4.0           # the simulated arm's clock runs this much faster than real time
SINGLE_GOAL = "single goal"
# The follower keeps every goal within 10 degrees of where the arm is, and says so when it cuts one:
# a goal that far from the arm is a jump the app asked for, so the simulated runs fail on it too.
CUT_GOAL = "was cut to"
TOKEN_VARS = ("MONOROBOT_GITHUB_TOKEN", "MONOROBOT_SIGNING_KEY", "GH_TOKEN", "GITHUB_TOKEN", "HF_TOKEN")
# The sim-only interlock and its two switches: on in every test and in every engine a test starts.
SIM_ONLY_SWITCHES = {"MONOROBOT_SIM_ONLY": "1", "MONOROBOT_USB_WATCH": "0", "MONOROBOT_FAKE_CAMERAS": "0"}


def load_main() -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"{APP_DIR.name.replace('-', '_')}_main", APP_DIR / "main.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def main() -> ModuleType:
    return load_main()


@pytest.fixture(autouse=True)
def _sandbox_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """HOME and MONOROBOT_HOME point into the test's temp folder, checked before the test starts:
    nothing lands in the real home or in ~/.monorobot (CONTRACTS 0.5).

    No tokens, and the sim-only interlock is on: no serial port, no USB watcher, no real camera.
    """
    (tmp_path / "home").mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MONOROBOT_HOME", str(tmp_path / "monorobot-home"))
    for name in ("HOME", "MONOROBOT_HOME"):
        assert Path(os.environ[name]).resolve().is_relative_to(tmp_path.resolve()), f"{name} is outside the temp folder"
    assert Path.home().resolve().is_relative_to(tmp_path.resolve()), "the home folder is outside the temp folder"
    for name in TOKEN_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in SIM_ONLY_SWITCHES.items():
        monkeypatch.setenv(name, value)


# --------------------------------------------------------------------------- doubles


class FakeCamera:
    """A camera in its own process: ``latest_frame()`` gives ``(frame, seq)``, a new frame per read.

    It connects and disconnects like a device, so a real robot can carry it (``attach_device``).
    """

    def __init__(self, scene: Callable[[int], np.ndarray] | None = None, *, fail_after: int | None = None) -> None:
        self.scene = scene or (lambda n: np.full((48, 64, 3), 40, dtype=np.uint8))
        self.reads = 0
        self.fail_after = fail_after
        self.connected = False

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def latest_frame(self) -> tuple[np.ndarray | None, int]:
        if self.fail_after is not None and self.reads >= self.fail_after:
            raise RuntimeError("camera unplugged")
        frame = self.scene(self.reads)
        self.reads += 1
        return frame, self.reads


class FakeArm:
    """An SO-101 follower double: keeps a joint state and records every commanded target, in order.

    ``targets`` holds ``(method, joints_deg, gripper, label)``; ``label()`` names the app's phase
    at each command. Reads are counted in ``reads``.
    """

    def __init__(
        self,
        start_deg: tuple[float, ...] | list[float] = REST_DEG,
        gripper: float = 0.0,
        *,
        devices: dict[str, Any] | None = None,
    ) -> None:
        self.start_deg = [float(v) for v in start_deg]
        self.joints_deg = list(self.start_deg)
        self.start_gripper = float(gripper)
        self.gripper = float(gripper)
        self.devices: dict[str, Any] = dict(devices or {})
        self.targets: list[tuple[str, list[float], float | None, str]] = []
        self.reads: dict[str, int] = {"get_joint_positions": 0, "gripper_get_width": 0, "get_telemetry": 0}
        self.label: Callable[[], str] = lambda: ""

    def get_joint_positions(self) -> list[float]:
        self.reads["get_joint_positions"] += 1
        return [math.radians(v) for v in self.joints_deg]

    def gripper_get_width(self) -> float:
        self.reads["gripper_get_width"] += 1
        return self.gripper

    def get_telemetry(self) -> dict[str, Any]:
        self.reads["get_telemetry"] += 1
        return {"joints": self.get_joint_positions(), "gripper": 1.0 - self.gripper}

    def send_teleop_action(self, arm_joints_rad: list[float], gripper_width: float) -> bool:
        assert len(arm_joints_rad) == 5, arm_joints_rad
        assert 0.0 <= gripper_width <= 1.0, gripper_width
        self._record("send_teleop_action", [math.degrees(v) for v in arm_joints_rad], float(gripper_width))
        return True

    def move_joints(self, joints: list[float], speed: float = 0.5, **kwargs: Any) -> bool:
        self._record("move_joints", [math.degrees(v) for v in joints], None)
        return True

    def gripper_open(self, width: float | None = None) -> bool:
        self._record("gripper_open", list(self.joints_deg), 1.0)
        return True

    def gripper_close(self, force: float = 10.0) -> bool:
        self._record("gripper_close", list(self.joints_deg), 0.0)
        return True

    def _record(self, method: str, joints_deg: list[float], gripper: float | None) -> None:
        self.targets.append((method, joints_deg, gripper, self.label()))
        self.joints_deg = list(joints_deg)
        if gripper is not None:
            self.gripper = gripper

    def labels(self) -> list[str]:
        return [label for *_, label in self.targets]


class WithDevices:
    """A robot with these devices attached, for a robot that has none (the simulated follower)."""

    def __init__(self, robot: Any, devices: dict[str, Any]) -> None:
        self._robot = robot
        self.devices = devices

    def __getattr__(self, name: str) -> Any:
        return getattr(self._robot, name)


# --------------------------------------------------------------------------- running the app


def make_app(main: ModuleType, robot: Any, robot_type: str = "hf_so101_follower") -> Any:
    """The app on ``robot``, configured from its own ``robot.yaml`` with ``robot_type`` as the run's type."""
    from monorobot.sdk.app_config import AppConfig

    config = AppConfig.load(APP_DIR / "robot.yaml")
    config.robot.type = robot_type
    app = getattr(main, APP_CLASS)(config)
    app.robot = robot
    if isinstance(robot, FakeArm):
        robot.label = lambda: getattr(app, "phase", "")
    return app


def _count_finish(app: Any) -> list[int]:
    calls: list[int] = []
    original = app.finish

    def finish() -> None:
        calls.append(app.tick)
        original()

    app.finish = finish
    app.finish_calls = calls
    return calls


def run_ticks(app: Any, n: int, *, before_tick: Callable[[Any], None] | None = None) -> Any:
    """``setup()``, then ``loop()`` up to ``n`` times with ``app.tick`` advanced, as the SDK runs it.

    Stops after the tick that calls ``finish()``. No sleeping: apps use ``tick / rate_hz`` as their clock.
    """
    calls = _count_finish(app)
    app.setup()
    for _ in range(n):
        if calls:
            break
        if before_tick is not None:
            before_tick(app)
        app.loop()
        app.tick += 1
    return app


def run_paced(app: Any, n: int, *, speedup: float = SIM_SPEEDUP) -> Any:
    """``run_ticks``, paced so the simulated arm's clock (``speedup`` times real time) keeps step with it."""
    calls = _count_finish(app)
    app.setup()
    period = 1.0 / app.rate_hz / speedup
    next_tick = time.monotonic()
    for _ in range(n):
        if calls:
            break
        next_tick += period
        app.loop()
        app.tick += 1
        delay = next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            next_tick = time.monotonic()
    return app


# --------------------------------------------------------------------------- the envelope and the checks


def urdf_limits_deg() -> list[tuple[float, float]]:
    """The follower's joint limits from its URDF, in degrees to one decimal, in ``ARM_JOINTS`` order.

    The URDF writes them in radians to five decimals (100 degrees is 1.74533), hence the rounding.
    """
    from monorobot.hal.adapters.hf_so101.follower import _URDF_PATH

    limits: dict[str, tuple[float, float]] = {}
    for joint in ET.parse(_URDF_PATH).getroot().iter("joint"):
        limit = joint.find("limit")
        if joint.get("name") in ARM_JOINTS and limit is not None:
            limits[joint.get("name", "")] = (round(math.degrees(float(limit.get("lower", "0"))), 1),
                                             round(math.degrees(float(limit.get("upper", "0"))), 1))
    return [limits[name] for name in ARM_JOINTS]


def envelope_deg() -> list[tuple[float, float]]:
    """10 degrees inside the URDF limits, and the base within 60 degrees either way."""
    out = [(lo + ENVELOPE_MARGIN_DEG, hi - ENVELOPE_MARGIN_DEG) for lo, hi in urdf_limits_deg()]
    out[0] = (max(out[0][0], -PAN_LIMIT_DEG), min(out[0][1], PAN_LIMIT_DEG))
    return out


def outside_by(joints_deg: list[float], envelope: list[tuple[float, float]]) -> list[float]:
    """How far each joint is outside the envelope (0 inside)."""
    return [max(lo - v, v - hi, 0.0) for v, (lo, hi) in zip(joints_deg, envelope)]


def assert_motion_ok(
    arm: FakeArm,
    rate_hz: float,
    phases: dict[str, float] | None = None,
    *,
    max_deg_per_s: float = MAX_DEG_PER_S,
) -> None:
    """Every target the app commanded follows the seed app motion rules.

    - The first target is within the per-tick limit (and 3 degrees) of the pose the arm started in.
    - Consecutive targets differ, per joint, by at most the phase's deg/s (``phases``, by label;
      ``max_deg_per_s`` otherwise) divided by ``rate_hz``, and never by more than 3 degrees.
    - Every target is inside the envelope, or, in a glide that starts outside it, no joint is
      further out than in the target before it.
    - Every target is inside the follower's URDF limits. This is the real limit check: the SDK's
      safety check for the SO-101 follower runs too, but the SO-101 adapters declare no joint
      limits, so it passes any target today.
    - ``move_joints`` is never called on an SO-101 (it ignores ``speed`` there and jumps).
    - The gripper stays in [0, 1] and goes from closed to open in no less than a second.
    """
    from monorobot.hal.adapters.hf_so101.follower import SO101FollowerAdapter
    from monorobot.hal.safety import SafetyMonitor

    assert arm.targets, "the app commanded nothing"
    methods = {method for method, *_ in arm.targets}
    assert "move_joints" not in methods, "move_joints was called on an SO-101"
    monitor = SafetyMonitor.for_adapter(SO101FollowerAdapter)
    urdf = urdf_limits_deg()
    envelope = envelope_deg()
    prev, prev_gripper = arm.start_deg, arm.start_gripper
    for i, (_method, joints, gripper, label) in enumerate(arm.targets):
        limit = min(MAX_STEP_DEG, (phases or {}).get(label, max_deg_per_s) / rate_hz)
        for name, a, b in zip(ARM_JOINTS, prev, joints):
            assert abs(b - a) <= limit + EPS, (
                f"target {i} ({label}): {name} moves {abs(b - a):.3f} degrees in one tick (limit {limit:.3f})"
            )
        out, prev_out = outside_by(joints, envelope), outside_by(prev, envelope)
        if any(out):
            assert all(o <= p + EPS for o, p in zip(out, prev_out)), (
                f"target {i} ({label}): {joints} goes further outside the envelope"
            )
        assert not any(o > EPS for o in outside_by(joints, urdf)), (
            f"target {i} ({label}): {joints} is outside the URDF limits {urdf}"
        )
        monitor.validate_call("move_joints", [[math.radians(v) for v in joints]], {})
        if gripper is not None:
            assert 0.0 <= gripper <= 1.0
            assert abs(gripper - prev_gripper) <= 1.0 / rate_hz + EPS, f"target {i}: the gripper snaps"
            prev_gripper = gripper
        prev = joints


# --------------------------------------------------------------------------- the simulated follower


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def sim_follower(tmp_path: Path) -> Any:
    """The real simulated SO-101 follower, connected, in this process, its clock ``SIM_SPEEDUP`` times real time.

    Fails the test when the arm logs its "single goal ... degrees from present" warning, or cuts a
    goal it was sent to 10 degrees from the arm ("was cut to").
    """
    pytest.importorskip("monorobot.hal.adapters.sim_so101")
    from monorobot.hal.registry import load_adapter

    records = _Records()
    sim_log = logging.getLogger("monorobot.hal.adapters.sim_so101")
    sim_log.addHandler(records)
    arm = load_adapter("sim_so101_follower", isolate=False, robot_id="sim-follower-1",
                       state_dir=tmp_path / "sim", time_scale=SIM_SPEEDUP)
    assert arm.connect()
    arm.state_file = tmp_path / "sim" / "sim-follower-1.json"
    try:
        yield arm
    finally:
        if arm.is_connected():
            arm.disconnect()
        sim_log.removeHandler(records)
    jumps = [m for m in records.messages if SINGLE_GOAL in m or CUT_GOAL in m]
    assert not jumps, jumps
    assert arm.big_goals == 0 and getattr(arm, "cut_goals", 0) == 0


def sim_joints_deg(arm: Any) -> list[float]:
    return [math.degrees(v) for v in arm.get_joint_positions()]


def wait_until_still(arm: Any, goal_deg: list[float], timeout_s: float = 5.0) -> list[float]:
    """Wait until the simulated arm is within 0.01 degrees of ``goal_deg``; return its joints."""
    deadline = time.monotonic() + timeout_s
    joints = sim_joints_deg(arm)
    while time.monotonic() < deadline and max(abs(a - b) for a, b in zip(joints, goal_deg)) > 0.01:
        time.sleep(0.01)
        joints = sim_joints_deg(arm)
    return joints


def run_under_sdk(
    main: ModuleType,
    tmp_path: Path,
    *,
    devices: dict[str, Any] | None = None,
    until: Callable[[Any], bool] | None = None,
    timeout_s: float = 60.0,
) -> tuple[Any, int, dict[str, Any]]:
    """Run the app with the SDK's own ``run()`` on the simulated follower; return ``(app, exit code, arm state)``.

    The app's ``robot.yaml`` with the simulated follower, in this process, in place of the real one
    (its clock ``SIM_SPEEDUP`` times real time), and ``devices`` attached to it as given:
    ``robot.yaml``'s own devices are never opened, so no real camera is. ``until(app)``, checked
    from another thread, ends the run with ``finish()`` once true, for an app that runs until
    stopped. A run still going after ``timeout_s`` is ended the same way and fails the test. The
    arm state is its state file once ``run()`` has returned. Fails on a "single goal" or "was cut to" warning, as
    ``sim_follower`` does.
    """
    import json
    import threading

    pytest.importorskip("monorobot.hal.adapters.sim_so101")
    from monorobot.sdk.app_config import AppConfig

    config = AppConfig.load(APP_DIR / "robot.yaml")
    config.robot.type = "sim_so101_follower"
    config.robot.isolate = False
    config.robot.kwargs = {**config.robot.kwargs, "robot_id": "sim-follower-1",
                           "state_dir": str(tmp_path / "sim"), "time_scale": SIM_SPEEDUP}
    config.devices = []
    app = getattr(main, APP_CLASS)(config)
    build_robot = app.build_robot

    def build_with_devices() -> Any:
        robot = build_robot()
        for name, device in (devices or {}).items():
            robot.attach_device(name, device)
        return robot

    app.build_robot = build_with_devices
    done, timed_out = threading.Event(), threading.Event()

    def watch() -> None:
        deadline = time.monotonic() + timeout_s
        while not done.wait(0.01):
            if time.monotonic() > deadline:
                timed_out.set()
                app.finish()
                return
            try:
                if until is not None and until(app):
                    app.finish()
                    return
            except AttributeError:      # setup() hasn't run yet
                pass

    records = _Records()
    sim_log = logging.getLogger("monorobot.hal.adapters.sim_so101")
    sim_log.addHandler(records)
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        code = app.run()
    finally:
        done.set()
        watcher.join(timeout=1.0)
        sim_log.removeHandler(records)
    assert not timed_out.is_set(), f"the run took over {timeout_s} s"
    jumps = [m for m in records.messages if SINGLE_GOAL in m or CUT_GOAL in m]
    assert not jumps, jumps
    assert app.robot.big_goals == 0 and getattr(app.robot, "cut_goals", 0) == 0
    state = json.loads((tmp_path / "sim" / "sim-follower-1.json").read_text())
    return app, code, state


# --------------------------------------------------------------------------- build


def built_manifest(tmp_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Run ``monorobot build`` on a copy of the app folder; return the package's manifest and member names.

    The engine is the one running these tests. The build runs in a sandbox home with no tokens and
    the sim-only interlock on.
    """
    import json

    copy = tmp_path / "build" / APP_DIR.name
    shutil.copytree(APP_DIR, copy, ignore=shutil.ignore_patterns(".git", "dist", "__pycache__", ".pytest_cache"))
    home = tmp_path / "build-home"
    (home / "gh").mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if k not in TOKEN_VARS}
    env.update({"HOME": str(home), "MONOROBOT_HOME": str(home / ".monorobot"), "GH_CONFIG_DIR": str(home / "gh")})
    env.update(SIM_ONLY_SWITCHES)
    for name in ("HOME", "MONOROBOT_HOME"):
        assert Path(env[name]).resolve().is_relative_to(tmp_path.resolve()), f"{name} is outside the temp folder"
    out = tmp_path / "build" / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "monorobot", "build", str(copy), "--out-dir", str(out)],
        cwd=copy, env=env, capture_output=True, text=True, timeout=120, check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    packages = list(out.glob("*.mrbot"))
    assert len(packages) == 1, packages
    with tarfile.open(packages[0], "r:gz") as tar:
        names = tar.getnames()
        member = tar.extractfile("manifest.json")
        assert member is not None
        manifest = json.loads(member.read())
    return manifest, names
