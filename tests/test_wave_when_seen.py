"""Wave when seen: the motion detector, waves and the cooldown on a fake arm, the simulated follower, the package."""

from __future__ import annotations

import logging
import os
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from conftest import (
    MID_DEG,
    REST_DEG,
    FakeArm,
    FakeCamera,
    WithDevices,
    assert_motion_ok,
    built_manifest,
    make_app,
    run_paced,
    run_ticks,
    run_under_sdk,
    urdf_limits_deg,
    wait_until_still,
)

RATE_HZ = 30
H, W = 480, 640
STARTS = {"rest": REST_DEG, "mid": MID_DEG}


# --------------------------------------------------------------------------- scenes


def empty(_n: int = 0) -> np.ndarray:
    return np.full((H, W, 3), 30, dtype=np.uint8)


def with_square(frame: np.ndarray, x: int, y: int = 140, size: int = 200) -> np.ndarray:
    frame = frame.copy()
    frame[y:y + size, x:x + size] = 250
    return frame


def square_enters_at(first: int):
    """An empty room, then from frame ``first`` on, someone (a bright square) standing still."""
    return lambda n: empty() if n < first else with_square(empty(), 220)


def someone_moving(n: int) -> np.ndarray:
    """Someone moving from side to side in every frame."""
    return with_square(empty(), 60 if n % 2 else 380)


def noise_scene(seed: int = 7, sigma: float = 6.0):
    """A textured, unchanging view with sensor noise on every frame."""
    rng = np.random.default_rng(seed)
    texture = rng.integers(0, 256, size=(H, W, 3)).astype(np.float32)
    return lambda _n: np.clip(texture + rng.normal(0.0, sigma, size=texture.shape), 0, 255).astype(np.uint8)


def trace(app, n: int) -> list[tuple[int, str]]:
    """Run ``n`` ticks; return ``(tick, phase before it)`` for each tick."""
    seen: list[tuple[int, str]] = []
    run_ticks(app, n, before_tick=lambda a: seen.append((a.tick, a.phase)))
    return seen


def wave_starts(seen: list[tuple[int, str]]) -> list[int]:
    """The ticks a wave started on: the phase was ``watch`` before them and ``rise`` after."""
    return [t for (t, before), (_, after) in pairwise(seen) if before == "watch" and after == "rise"]


def wave_ends(seen: list[tuple[int, str]]) -> list[int]:
    """The ticks a wave ended on (it sent nothing more): ``watch`` after them, a wave phase before."""
    return [t for (t, before), (_, after) in pairwise(seen) if before != "watch" and after == "watch"]


# --------------------------------------------------------------------------- the detector


def test_static_noise_never_fires(main):
    detector = main.MotionDetector()
    scene = noise_scene()
    assert not any(detector.feed(scene(n)) for n in range(300))


def test_an_unchanging_view_never_fires(main):
    detector = main.MotionDetector()
    frame = with_square(empty(), 100)
    assert not any(detector.feed(frame) for _ in range(100))


def test_a_bright_square_entering_fires_on_the_fifth_frame(main):
    detector = main.MotionDetector()
    scene = square_enters_at(1)
    fired = [detector.feed(scene(n)) for n in range(6)]
    assert fired == [False, False, False, False, False, True]   # the empty room, then 5 frames with the square


def test_two_d_frames_work(main):
    detector = main.MotionDetector()
    scene = square_enters_at(1)
    fired = [detector.feed(scene(n)[:, :, 0]) for n in range(6)]
    assert fired == [False] * 5 + [True]


def test_a_small_change_does_not_fire(main):
    detector = main.MotionDetector()
    frames = [empty()] + [with_square(empty(), 300, size=100)] * 20    # about 3 percent of the picture
    assert not any(detector.feed(f) for f in frames)


def test_frames_are_downscaled_to_about_64_wide(main):
    detector = main.MotionDetector()
    detector.feed(np.zeros((720, 1280, 3), dtype=np.uint8))
    assert detector.background is not None
    assert 60 <= detector.background.shape[1] <= 70


def test_reset_forgets_the_background(main):
    detector = main.MotionDetector()
    scene = square_enters_at(1)
    for n in range(4):
        detector.feed(scene(n))
    detector.reset()
    assert [detector.feed(scene(1)) for _ in range(6)] == [False] * 6   # the square is the new background


def test_a_bad_frame_is_refused(main):
    with pytest.raises(ValueError):
        main.MotionDetector().feed(np.zeros((4,), dtype=np.uint8))


# --------------------------------------------------------------------------- the app


def test_needs_a_camera(main):
    app = make_app(main, FakeArm())
    with pytest.raises(RuntimeError) as exc:
        app.setup()
    assert str(exc.value) == "Wave when seen needs a camera. Add one to this robot in My robots."


def test_needs_an_so101(main):
    app = make_app(main, FakeArm(devices={"camera": FakeCamera()}), robot_type="lynxmotion_al5d")
    with pytest.raises(RuntimeError) as exc:
        app.setup()
    assert str(exc.value) == "Wave when seen runs on the SO-101 arm."


def test_holds_still_while_nobody_is_there(main):
    arm = FakeArm(REST_DEG, devices={"camera": FakeCamera(noise_scene())})
    app = make_app(main, arm)
    run_ticks(app, 20 * RATE_HZ)
    assert arm.targets == []
    assert app.finish_calls == []                   # it runs until stopped


def test_looks_every_second_tick(main):
    camera = FakeCamera()
    app = make_app(main, FakeArm(devices={"camera": camera}))
    run_ticks(app, 100)
    assert camera.reads == 50


def test_waves_on_the_fifth_frame_with_someone_in_it(main):
    """Frames are read on even ticks: the empty room on tick 0, then the square; the fifth fires on tick 10."""
    arm = FakeArm(REST_DEG, devices={"camera": FakeCamera(square_enters_at(1))})
    app = make_app(main, arm)
    run_ticks(app, 10)
    assert arm.targets == []
    app.loop()
    assert len(arm.targets) == 1
    assert app.phase == "rise"


@pytest.mark.parametrize("start", sorted(STARTS))
def test_two_waves_in_a_row_follow_the_motion_rules(main, start):
    arm = FakeArm(STARTS[start], gripper=0.2, devices={"camera": FakeCamera(someone_moving)})
    app = make_app(main, arm)
    seen = trace(app, 1000)
    assert len(wave_starts(seen)) == 2
    assert_motion_ok(arm, RATE_HZ)
    runs = []
    for label in arm.labels():
        if not runs or runs[-1][0] != label:
            runs.append([label, 0])
        runs[-1][1] += 1
    assert [name for name, _ in runs] == ["rise", "wave", "back"] * 2
    for name, count in runs:
        if name == "wave":
            assert abs(count - main.WAVES * main.PERIOD_S * RATE_HZ) <= 2
    home = main.clip_to_envelope(STARTS[start])
    back_ends = [i for i, (a, b) in enumerate(zip(arm.labels(), arm.labels()[1:] + ["end"])) if a == "back" != b]
    for i in back_ends:
        assert arm.targets[i][1] == pytest.approx(home, abs=1e-9)
        assert arm.targets[i][2] == pytest.approx(0.2)
    assert app.finish_calls == []


def test_the_cooldown_holds(main):
    """Someone keeps moving: the next wave starts 10 s after the last one ended, not before."""
    arm = FakeArm(REST_DEG, devices={"camera": FakeCamera(someone_moving)})
    app = make_app(main, arm)
    seen = trace(app, 1000)
    starts, ends = wave_starts(seen), wave_ends(seen)
    assert len(starts) == 2 and len(ends) >= 1
    gap_s = (starts[1] - ends[0]) / RATE_HZ
    assert main.COOLDOWN_S <= gap_s <= main.COOLDOWN_S + 0.5, gap_s


def test_waves_with_a_two_d_camera(main):
    arm = FakeArm(MID_DEG, devices={"camera": FakeCamera(lambda n: someone_moving(n)[:, :, 1])})
    app = make_app(main, arm)
    run_ticks(app, 20)
    assert app.waves == 1
    assert app.phase == "rise"


def test_logs_are_short_and_plain(main, caplog):
    caplog.set_level(logging.INFO, logger="wave-when-seen")
    app = make_app(main, FakeArm(REST_DEG, devices={"camera": FakeCamera(someone_moving)}))
    run_ticks(app, 1000)
    app.on_shutdown()
    lines = [r.getMessage() for r in caplog.records if r.name == "wave-when-seen"]
    assert lines == [
        "Watching for someone.",
        "Someone's here: waving.", "Watching again.",
        "Someone's here: waving.", "Watching again.",
        "Waved 2 times.",
    ]
    for line in lines:
        assert line[0].isupper() and "—" not in line and "so101" not in line.lower()


def test_the_motion_check_catches_a_jump_and_move_joints():
    arm = FakeArm(MID_DEG)
    arm.send_teleop_action([np.radians(v) for v in (MID_DEG[0] + 4.0, *MID_DEG[1:])], 0.0)
    with pytest.raises(AssertionError, match="in one tick"):
        assert_motion_ok(arm, RATE_HZ)
    arm = FakeArm(MID_DEG)
    arm.move_joints([np.radians(v) for v in MID_DEG])
    with pytest.raises(AssertionError, match="move_joints"):
        assert_motion_ok(arm, RATE_HZ)
    near_edge = (*MID_DEG[:3], 84.0, MID_DEG[4])           # wrist_flex's envelope ends at 85
    arm = FakeArm(near_edge)
    arm.send_teleop_action([np.radians(v) for v in (*near_edge[:3], 85.5, near_edge[4])], 0.0)
    with pytest.raises(AssertionError, match="envelope"):
        assert_motion_ok(arm, RATE_HZ)


def test_the_motion_check_catches_a_target_past_the_urdf_limits():
    """The SDK's safety check has no SO-101 limits, so ``assert_motion_ok`` checks the URDF ones
    itself: a small step inward that is still past the shoulder's limit fails."""
    lowest = urdf_limits_deg()[1][0]
    arm = FakeArm([0.0, lowest - 6.0, 30.0, 0.0, 0.0])
    arm.send_teleop_action([np.radians(v) for v in (0.0, lowest - 5.5, 30.0, 0.0, 0.0)], 0.0)
    with pytest.raises(AssertionError, match="URDF"):
        assert_motion_ok(arm, RATE_HZ)


def test_the_tests_never_see_the_real_home(tmp_path):
    """CONTRACTS 0.5: HOME and MONOROBOT_HOME both point under the test's temp folder."""
    assert Path.home().resolve().is_relative_to(tmp_path.resolve())
    assert Path(os.environ["MONOROBOT_HOME"]).resolve().is_relative_to(tmp_path.resolve())


# --------------------------------------------------------------------------- simulated follower


def test_one_wave_on_the_simulated_follower(main, sim_follower):
    camera = FakeCamera(square_enters_at(1))
    app = make_app(main, WithDevices(sim_follower, {"camera": camera}), robot_type="sim_so101_follower")
    run_paced(app, 450)                     # one wave is about 10 s; the person then stands still
    assert app.waves == 1
    assert app.phase == "watch"
    home = main.clip_to_envelope(list(REST_DEG))
    assert wait_until_still(sim_follower, home) == pytest.approx(home, abs=2.0)


def test_the_sdk_runs_it_until_it_is_ended(main, tmp_path, monkeypatch, caplog):
    """The SDK's own run(), at the app's own rate, with the camera attached to the arm as ``camera``:
    it waves for someone moving, then the run is ended once the wave is over. Exit code 0, and the
    arm's on_exit folds it to rest and lets go. One wave of one period, to keep it short."""
    caplog.set_level(logging.INFO, logger="wave-when-seen")
    monkeypatch.setattr(main, "WAVES", 1)
    camera = FakeCamera(someone_moving)
    app, code, state = run_under_sdk(main, tmp_path, devices={"camera": camera},
                                     until=lambda a: a.last_wave_end is not None)
    assert code == 0 and app.error is None
    assert app.waves == 1 and app.phase == "watch"
    assert camera.reads > 0 and not camera.connected
    assert state["joints_deg"] == pytest.approx(list(REST_DEG), abs=0.5)
    assert state["torque"] == "off"
    lines = [r.getMessage() for r in caplog.records if r.name == "wave-when-seen"]
    assert lines == ["Watching for someone.", "Someone's here: waving.", "Watching again.", "Waved 1 time."]


# --------------------------------------------------------------------------- the package


def test_built_manifest(tmp_path):
    manifest, names = built_manifest(tmp_path)
    assert manifest["schema"] == "monorobot/1.0"
    assert manifest["name"] == "wave-when-seen"
    assert manifest["version"] == "0.1.0"
    assert manifest["author"] == {"handle": "grand-wizard-create"}
    assert manifest["compat"]["robots"] == ["hf_so101_follower"]
    assert manifest["devices"] == {"required": ["camera"], "optional": []}
    assert manifest["permissions"]["network"] is False
    assert manifest["license"] == "Apache-2.0"
    assert manifest["whats_new"] == "First version."
    assert manifest["description"] == "Waves when someone steps in front of the camera."
    assert "also_uses" not in manifest
    assert sorted(names) == [".gitignore", "README.md", "main.py", "manifest.json", "pyproject.toml", "robot.yaml"]
