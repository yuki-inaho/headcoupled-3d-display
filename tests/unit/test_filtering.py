"""SE(3) manifold EMA pose filter and on-manifold EKF with constant velocity.

Stage-1 filter per docs/plan_se3_pose_filter_and_calibration.md §10 C4: a manifold
EMA on SE(3) plus a smoothed body-frame twist used for a fixed-horizon prediction.
Stage-2 EKF (same plan §10 C4 follow-up): on-manifold EKF with state dimension 12
(SE(3) x R^6 body twist) implementing the same :class:`PoseFilter` contract.
"""

from __future__ import annotations

import time

import numpy as np

from headcoupled_display import lie
from headcoupled_display.filtering import Se3EKF, Se3PoseFilter

_HORIZON_LIMIT_S = 0.1
_DT_S = 1.0 / 30.0


def _identity() -> np.ndarray:
    return np.eye(4)


def _transform(translation: np.ndarray, axis_angle: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = lie.exp_so3(axis_angle)
    transform[:3, 3] = translation
    return transform


def _twist_body(transform: np.ndarray) -> np.ndarray:
    # A constant body-frame twist that moves the transform by exp(xi*dt) per second.
    return lie.log_se3(transform)  # for a static target this is its own tangent


def test_first_observation_initialises_state() -> None:
    filtr = Se3PoseFilter()
    raw = _transform(np.array([0.0, 0.0, 0.67]), np.array([0.0, 0.0, 0.0]))
    filtr.correct(raw, reprojection_rms_px=1.0, timestamp=time.monotonic())
    assert np.allclose(filtr.pose, raw, atol=1e-12)
    assert np.allclose(filtr.velocity_body, np.zeros(6))
    assert filtr.initialised


def test_static_observation_reduces_jitter() -> None:
    rng = np.random.default_rng(0)
    base = _transform(np.array([0.0, 0.0, 0.67]), np.zeros(3))
    filtr = Se3PoseFilter(position_alpha=0.3)
    now = 0.0
    noisy_outputs = []
    for _ in range(200):
        now += 1.0 / 30.0
        jitter = rng.standard_normal(3) * 0.003  # 3 mm noise
        raw = base.copy()
        raw[:3, 3] += jitter
        filtr.correct(raw, reprojection_rms_px=1.0, timestamp=now)
        noisy_outputs.append(filtr.pose[:3, 3].copy())
    noisy_outputs = np.asarray(noisy_outputs)
    output_std = float(noisy_outputs[50:].std(axis=0).mean())
    assert output_std < 0.003  # smoothed std below the 3 mm input std


def test_constant_velocity_tracking_lags_but_follows() -> None:
    filtr = Se3PoseFilter()
    velocity = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])  # 0.1 m/s along body +x
    start = _identity()
    now = 0.0
    filtr.correct(start, reprojection_rms_px=1.0, timestamp=now)
    predicted_errors = []
    for _ in range(60):
        now += 1.0 / 30.0
        raw = lie.oplus(start, velocity * now)
        filtr.correct(raw, reprojection_rms_px=1.0, timestamp=now)
        predicted = filtr.predict_to(0.0)  # pose at the current time
        predicted_errors.append(float(np.linalg.norm(predicted[:3, 3] - raw[:3, 3])))
    # After warmup the filter should track within a few cm despite the EMA lag.
    assert max(predicted_errors[30:]) < 0.03


def test_prediction_advances_eye_along_velocity() -> None:
    filtr = Se3PoseFilter()
    velocity = np.array([0.0, 0.0, 0.0, 0.2, 0.0, 0.0])  # 0.2 m/s +x body
    start = _identity()
    filtr.correct(start, reprojection_rms_px=1.0, timestamp=0.0)
    # Inject a known velocity directly to test predict_to in isolation.
    filtr._velocity_body = velocity.copy()
    predicted = filtr.predict_to(0.1)
    # horizon clamped to 0.1 s; 0.2 m/s * 0.1 s = 0.02 m along body +x.
    assert np.isclose(predicted[0, 3], 0.02, atol=1e-9)


def test_prediction_horizon_is_clamped() -> None:
    filtr = Se3PoseFilter()
    filtr.correct(_identity(), reprojection_rms_px=1.0, timestamp=0.0)
    filtr._velocity_body = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    clamped = filtr.predict_to(1.0)  # requested 1 s but limit is 0.1 s
    assert np.isclose(clamped[0, 3], _HORIZON_LIMIT_S, atol=1e-9)


def test_outlier_rejected_when_velocity_exceeds_gate() -> None:
    filtr = Se3PoseFilter()
    filtr.correct(_transform(np.array([0, 0, 0.6]), np.zeros(3)), 1.0, timestamp=0.0)
    # A 0.5 m jump in a single 1/30 s frame is > 1.5 m/s and must be rejected.
    outlier = _transform(np.array([0.5, 0.0, 0.6]), np.zeros(3))
    accepted = filtr.correct(outlier, reprojection_rms_px=1.0, timestamp=1.0 / 30.0)
    assert not accepted
    # State did not jump to the outlier.
    assert np.allclose(filtr.pose[:3, 3], np.array([0.0, 0.0, 0.6]))


def test_repeated_invalid_keeps_last_good_pose() -> None:
    filtr = Se3PoseFilter()
    filtr.correct(_transform(np.array([0, 0, 0.6]), np.zeros(3)), 1.0, timestamp=0.0)
    for _ in range(10):
        outlier = _transform(np.array([0.5, 0.0, 0.6]), np.zeros(3))
        filtr.correct(outlier, reprojection_rms_px=1.0, timestamp=1.0 / 30.0)
    assert np.allclose(filtr.pose[:3, 3], np.array([0.0, 0.0, 0.6]))


def test_reprojection_rms_weights_gain() -> None:
    # High RMS (low confidence) should make the filter move less toward the raw pose.
    filtr = Se3PoseFilter()
    filtr.correct(_transform(np.array([0, 0, 0.6]), np.zeros(3)), 1.0, timestamp=0.0)
    raw = _transform(np.array([0.01, 0.0, 0.6]), np.zeros(3))
    filtr.correct(raw, reprojection_rms_px=10.0, timestamp=1.0 / 30.0)  # high RMS
    low_gain_shift = abs(filtr.pose[0, 3] - 0.0)

    filtr2 = Se3PoseFilter()
    filtr2.correct(_transform(np.array([0, 0, 0.6]), np.zeros(3)), 1.0, timestamp=0.0)
    filtr2.correct(raw, reprojection_rms_px=0.5, timestamp=1.0 / 30.0)  # low RMS
    high_gain_shift = abs(filtr2.pose[0, 3] - 0.0)
    assert low_gain_shift < high_gain_shift


def test_reset_clears_state() -> None:
    filtr = Se3PoseFilter()
    filtr.correct(_transform(np.array([0, 0, 0.6]), np.zeros(3)), 1.0, timestamp=0.0)
    filtr.reset()
    assert not filtr.initialised
    assert np.allclose(filtr.velocity_body, np.zeros(6))


def test_predict_to_zero_returns_pose() -> None:
    filtr = Se3PoseFilter()
    raw = _transform(np.array([0, 0, 0.6]), np.array([0.1, 0.0, 0.0]))
    filtr.correct(raw, 1.0, timestamp=0.0)
    assert np.allclose(filtr.predict_to(0.0), filtr.pose)


def test_velocity_is_smoothed_not_raw() -> None:
    # A single large (but gated-acceptable) motion should not slam velocity to the
    # full instantaneous value; the EMA on velocity keeps it bounded.
    filtr = Se3PoseFilter()
    filtr.correct(_identity(), 1.0, timestamp=0.0)
    # 2 cm in one frame ~ 0.6 m/s -- under the gate, but velocity EMA limits it.
    raw = _transform(np.array([0.02, 0.0, 0.0]), np.zeros(3))
    filtr.correct(raw, 1.0, timestamp=1.0 / 30.0)
    instant_speed = 0.02 / (1.0 / 30.0)
    filtered_speed = float(np.linalg.norm(filtr.velocity_body[3:]))
    assert filtered_speed < instant_speed


def test_no_motion_keeps_velocity_near_zero() -> None:
    filtr = Se3PoseFilter()
    base = _transform(np.array([0, 0, 0.6]), np.zeros(3))
    now = 0.0
    for _ in range(100):
        now += 1.0 / 30.0
        filtr.correct(base, 1.0, timestamp=now)
    assert float(np.linalg.norm(filtr.velocity_body)) < 1e-3


def test_adaptive_gain_within_bounds() -> None:
    # gain must stay in (0, 1] regardless of RMS.
    filtr = Se3PoseFilter()
    for rms in (0.0, 0.5, 4.0, 100.0):
        gain = filtr._adaptive_gain(rms)
        assert 0.0 < gain <= 1.0


def _ekf() -> Se3EKF:
    return Se3EKF()


def _ekf_track_constant_velocity(
    velocity: np.ndarray,
    steps: int,
    *,
    start: np.ndarray | None = None,
    measurement_noise: float = 0.0,
) -> tuple[Se3EKF, list[float]]:
    """Run an EKF over constant-velocity ground truth, returning position errors."""

    filtr = _ekf()
    if start is None:
        start = _identity()
    now = 0.0
    errors: list[float] = []
    for step in range(steps):
        now += _DT_S
        raw = lie.oplus(start, velocity * now)
        if measurement_noise:
            raw = raw.copy()
            raw[:3, 3] += np.random.default_rng(step).standard_normal(3) * measurement_noise
        filtr.correct(raw, reprojection_rms_px=1.0, timestamp=now)
        predicted = filtr.predict_to(0.0)
        errors.append(float(np.linalg.norm(predicted[:3, 3] - raw[:3, 3])))
    return filtr, errors


def test_ekf_first_observation_initialises_state() -> None:
    filtr = _ekf()
    raw = _transform(np.array([0.0, 0.0, 0.67]), np.array([0.0, 0.0, 0.0]))
    filtr.correct(raw, reprojection_rms_px=1.0, timestamp=time.monotonic())
    assert np.allclose(filtr.pose, raw, atol=1e-12)
    assert np.allclose(filtr.velocity_body, np.zeros(6))
    assert filtr.initialised
    assert filtr.covariance.shape == (12, 12)
    assert np.all(np.diag(filtr.covariance) > 0.0)


def test_ekf_constant_velocity_tracking_is_exact() -> None:
    # A constant-velocity EKF should track a constant-velocity target with only
    # prediction error (the motion model is exact); after warmup < 2 cm is required
    # by the workdoc and achieved well below that.
    velocity = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])  # 0.1 m/s along body +x
    _, errors = _ekf_track_constant_velocity(velocity, 60)
    assert max(errors[30:]) < 0.02


def test_ekf_constant_velocity_with_noise_tracks() -> None:
    # With measurement noise the EKF still tracks the mean within a few cm.
    velocity = np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
    _, errors = _ekf_track_constant_velocity(velocity, 120, measurement_noise=0.005)
    assert max(errors[90:]) < 0.03


def test_ekf_covariance_converges_static() -> None:
    # Stationary target: the EKF covariance should shrink and converge.
    filtr = _ekf()
    base = _transform(np.array([0.0, 0.0, 0.6]), np.zeros(3))
    now = 0.0
    initial_trace = float(np.trace(filtr.covariance)) if filtr.initialised else None
    filtr.correct(base, reprojection_rms_px=1.0, timestamp=now)
    initial_trace = float(np.trace(filtr.covariance))
    traces = [initial_trace]
    for _ in range(100):
        now += _DT_S
        filtr.correct(base.copy(), reprojection_rms_px=1.0, timestamp=now)
        traces.append(float(np.trace(filtr.covariance)))
    assert traces[-1] < traces[0] * 0.5
    # The trace should be decreasing overall (no blow-up).
    assert all(traces[i + 1] <= traces[i] * 1.01 + 1e-12 for i in range(len(traces) - 1))


def test_ekf_outlier_rejected_and_covariance_unchanged() -> None:
    filtr = _ekf()
    filtr.correct(_transform(np.array([0, 0, 0.6]), np.zeros(3)), 1.0, timestamp=0.0)
    cov_before = filtr.covariance.copy()
    outlier = _transform(np.array([0.5, 0.0, 0.6]), np.zeros(3))
    accepted = filtr.correct(outlier, reprojection_rms_px=1.0, timestamp=_DT_S)
    assert not accepted
    assert np.allclose(filtr.pose[:3, 3], np.array([0.0, 0.0, 0.6]))
    assert np.allclose(filtr.covariance, cov_before)


def test_ekf_prediction_advances_eye_along_velocity() -> None:
    filtr = _ekf()
    velocity = np.array([0.0, 0.0, 0.0, 0.2, 0.0, 0.0])
    filtr.correct(_identity(), reprojection_rms_px=1.0, timestamp=0.0)
    filtr._velocity_body = velocity.copy()
    predicted = filtr.predict_to(0.1)
    assert np.isclose(predicted[0, 3], 0.02, atol=1e-9)


def test_ekf_prediction_horizon_is_clamped() -> None:
    filtr = _ekf()
    filtr.correct(_identity(), reprojection_rms_px=1.0, timestamp=0.0)
    filtr._velocity_body = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    clamped = filtr.predict_to(1.0)
    assert np.isclose(clamped[0, 3], _HORIZON_LIMIT_S, atol=1e-9)


def test_ekf_reset_clears_state() -> None:
    filtr = _ekf()
    filtr.correct(_transform(np.array([0, 0, 0.6]), np.zeros(3)), 1.0, timestamp=0.0)
    filtr.reset()
    assert not filtr.initialised
    assert np.allclose(filtr.velocity_body, np.zeros(6))
    assert np.allclose(filtr.covariance, np.zeros((12, 12)))


def test_ekf_rms_scales_update() -> None:
    # High RMS (low confidence) -> smaller pose correction toward the raw pose.
    def shift_with(rms: float) -> float:
        filtr = _ekf()
        filtr.correct(_transform(np.array([0, 0, 0.6]), np.zeros(3)), 1.0, timestamp=0.0)
        raw = _transform(np.array([0.01, 0.0, 0.6]), np.zeros(3))
        filtr.correct(raw, reprojection_rms_px=rms, timestamp=_DT_S)
        return abs(filtr.pose[0, 3] - 0.0)

    low_shift = shift_with(0.5)
    high_shift = shift_with(10.0)
    assert low_shift > high_shift


def test_ekf_rotation_converges() -> None:
    # A static target with a small initial rotation mismatch: the EKF should
    # converge the orientation back toward the measurement.
    filtr = _ekf()
    base = _transform(np.array([0.0, 0.0, 0.6]), np.zeros(3))
    filtr.correct(base, reprojection_rms_px=1.0, timestamp=0.0)
    # Force a rotation error in the state, then feed the true pose.
    filtr._pose = filtr._pose @ lie.exp_se3(np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0]))
    now = 0.0
    for _ in range(50):
        now += _DT_S
        filtr.correct(base, reprojection_rms_px=1.0, timestamp=now)
    rotation_error = float(np.linalg.norm(lie.log_so3(filtr.pose[:3, :3].T @ base[:3, :3])))
    assert rotation_error < 0.01
