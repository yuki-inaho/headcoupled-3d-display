"""Stage-1 SE(3) pose filter and stage-2 on-manifold EKF.

Stage 1 (``Se3PoseFilter``) is the design from
``docs/plan_se3_pose_filter_and_calibration.md`` §10 (review item C4): a
geodesic exponential moving average on SE(3) with a smoothed body-frame twist,
used for a *fixed-horizon* short-term prediction.

Stage 2 (``Se3EKF``) is the on-manifold EKF with state dimension 12
(SE(3) x R^6 body twist) per the same plan's §10 C4 follow-up. Both implement
the same :class:`PoseFilter` contract so :mod:`headcoupled_display.tracking`
can switch between them via ``quality_metrics["pose_filter"]``.

Conventions follow :mod:`headcoupled_display.lie` (right perturbation,
``xi[:3] = phi`` rotation, ``xi[3:] = rho`` translation).
"""

from __future__ import annotations

import time
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from . import lie

FloatArray = NDArray[np.float64]

#: Default prediction horizon. Stage 1 keeps this fixed (review item B2): dynamic
#: horizon requires the still-unresolved <2 ms clock sync and is equivalent to the
#: out-of-scope display-time interpolation (item 6).
DEFAULT_PREDICTION_HORIZON_S = 1.0 / 30.0
HORIZON_LIMIT_S = 0.1

#: Velocity gate. A frame-to-frame motion faster than these is rejected as an
#: outlier rather than clamped, so a bad pose never partially corrupts the state.
DEFAULT_MAX_LINEAR_SPEED_M_S = 1.5
DEFAULT_MAX_ANGULAR_SPEED_RAD_S = 4.0

#: RMS (px) at which the adaptive gain falls to half its base value. Lower RMS
#: (higher confidence) -> higher gain. For the EKF this is the reference at which
#: the measurement covariance keeps its configured base value.
DEFAULT_RMS_REFERENCE_PX = 2.0

#: Base measurement noise std for the EKF: angular (rad) and position (m). Scaled
#: by (rms / rms_reference) so higher reprojection RMS trusts the pose less.
DEFAULT_MEASUREMENT_ROT_STD_RAD = 0.02
DEFAULT_MEASUREMENT_POS_STD_M = 0.005

#: Continuous-time process noise intensities for the EKF random walk: angular and
#: linear velocity noise (per sqrt-second), plus small pose drift terms.
DEFAULT_ANGULAR_VELOCITY_NOISE = 1.5
DEFAULT_LINEAR_VELOCITY_NOISE = 0.5
DEFAULT_POSE_ANGULAR_NOISE = 0.02
DEFAULT_POSE_LINEAR_NOISE = 0.02

#: Initial covariance diagonal for the EKF state (rad/m and rad/s / m/s).
DEFAULT_INITIAL_ROT_STD_RAD = 0.1
DEFAULT_INITIAL_POS_STD_M = 0.05
DEFAULT_INITIAL_ANGULAR_VEL_STD_RAD_S = 2.0
DEFAULT_INITIAL_LINEAR_VEL_STD_M_S = 0.5

#: Lower bound on the RMS scale factor so a very confident frame cannot drive R
#: to zero (which would make the filter discard its prediction entirely).
_MIN_RMS_SCALE = 0.05

_EPS_DT_S = 1e-6


def _orthonormalize(rotation: FloatArray) -> FloatArray:
    """Return the closest proper rotation matrix via polar decomposition."""

    matrix = np.asarray(rotation, dtype=np.float64)
    u, _, vt = np.linalg.svd(matrix)
    ortho = u @ vt
    if np.linalg.det(ortho) < 0.0:
        u[:, -1] *= -1.0
        ortho = u @ vt
    return ortho


class PoseFilter(Protocol):
    """Common contract shared by :class:`Se3PoseFilter` and :class:`Se3EKF`."""

    @property
    def initialised(self) -> bool: ...

    @property
    def pose(self) -> FloatArray: ...

    @property
    def velocity_body(self) -> FloatArray: ...

    def correct(
        self,
        raw_pose: FloatArray,
        reprojection_rms_px: float,
        *,
        timestamp: float | None = None,
    ) -> bool: ...

    def predict_to(self, horizon_s: float | None = None) -> FloatArray: ...

    def reset(self) -> None: ...


class Se3PoseFilter:
    """Manifold EMA + constant-velocity predictor for a head pose ``T_S_H``.

    State: ``pose`` (the smoothed SE(3) transform) and ``velocity_body`` (the
    smoothed body-frame twist ``[phi, rho]``). ``correct`` applies an outlier
    gate, then moves the pose a fraction of the geodesic toward the raw pose;
    ``predict_to`` extrapolates by ``velocity_body * horizon`` (clamped).
    """

    def __init__(
        self,
        *,
        position_alpha: float = 0.5,
        velocity_alpha: float = 0.3,
        prediction_horizon_s: float = DEFAULT_PREDICTION_HORIZON_S,
        horizon_limit_s: float = HORIZON_LIMIT_S,
        max_linear_speed_m_s: float = DEFAULT_MAX_LINEAR_SPEED_M_S,
        max_angular_speed_rad_s: float = DEFAULT_MAX_ANGULAR_SPEED_RAD_S,
        rms_reference_px: float = DEFAULT_RMS_REFERENCE_PX,
    ) -> None:
        if not 0.0 < position_alpha <= 1.0:
            raise ValueError("position_alpha must be in (0, 1]")
        if not 0.0 < velocity_alpha <= 1.0:
            raise ValueError("velocity_alpha must be in (0, 1]")
        if horizon_limit_s <= 0.0:
            raise ValueError("horizon_limit_s must be positive")
        self._position_alpha = position_alpha
        self._velocity_alpha = velocity_alpha
        self._prediction_horizon_s = prediction_horizon_s
        self._horizon_limit_s = horizon_limit_s
        self._max_linear_speed_m_s = max_linear_speed_m_s
        self._max_angular_speed_rad_s = max_angular_speed_rad_s
        self._rms_reference_px = rms_reference_px
        self._pose: FloatArray | None = None
        self._velocity_body = np.zeros(6, dtype=np.float64)
        self._last_timestamp: float | None = None

    @property
    def initialised(self) -> bool:
        return self._pose is not None

    @property
    def pose(self) -> FloatArray:
        if self._pose is None:
            raise RuntimeError("Se3PoseFilter has not been corrected yet")
        return self._pose

    @property
    def velocity_body(self) -> FloatArray:
        return self._velocity_body

    def _adaptive_gain(self, reprojection_rms_px: float) -> float:
        """Lower reprojection RMS -> higher gain, clamped to (0, 1]."""

        rms = max(float(reprojection_rms_px), 0.0)
        gain = self._position_alpha / (1.0 + rms / self._rms_reference_px)
        return min(1.0, max(gain, 1e-3))

    def correct(
        self,
        raw_pose: FloatArray,
        reprojection_rms_px: float,
        *,
        timestamp: float | None = None,
    ) -> bool:
        """Update the filter with a raw pose, returning False if rejected by the gate.

        ``timestamp`` should be monotonic seconds (e.g. ``time.monotonic()``). When
        omitted the wall clock is used; mixing the two across calls is a caller bug.
        """

        raw = np.asarray(raw_pose, dtype=np.float64)
        if raw.shape != (4, 4):
            raise ValueError("raw_pose must be a 4x4 transform")
        now = float(time.monotonic() if timestamp is None else timestamp)

        if self._pose is None or self._last_timestamp is None:
            self._pose = raw.copy()
            self._pose[:3, :3] = _orthonormalize(self._pose[:3, :3])
            self._velocity_body = np.zeros(6, dtype=np.float64)
            self._last_timestamp = now
            return True

        dt = now - self._last_timestamp
        if dt < _EPS_DT_S:
            # No time has passed; keep velocity, still ease the pose toward raw.
            instantaneous_twist = np.zeros(6, dtype=np.float64)
        else:
            instantaneous_twist = lie.ominus(raw, self._pose) / dt
            linear_speed = float(np.linalg.norm(instantaneous_twist[3:]))
            angular_speed = float(np.linalg.norm(instantaneous_twist[:3]))
            if (
                linear_speed > self._max_linear_speed_m_s
                or angular_speed > self._max_angular_speed_rad_s
            ):
                # Outlier: keep the last good pose, velocity, *and* timestamp. The
                # next frame's gate is then measured against the last accepted pose
                # over the real elapsed time, so a persistent glitch stays rejected
                # instead of being accepted once ``dt`` grows enough.
                return False

        gain = self._adaptive_gain(reprojection_rms_px)
        delta = lie.ominus(raw, self._pose)
        self._pose = lie.oplus(self._pose, gain * delta)
        self._pose[:3, :3] = _orthonormalize(self._pose[:3, :3])

        if dt >= _EPS_DT_S:
            self._velocity_body = (
                1.0 - self._velocity_alpha
            ) * self._velocity_body + self._velocity_alpha * instantaneous_twist
        self._last_timestamp = now
        return True

    def predict_to(self, horizon_s: float | None = None) -> FloatArray:
        """Extrapolate the pose by ``velocity_body * horizon`` (clamped to the limit).

        ``horizon_s=None`` uses the configured default. The clamp enforces review
        item B2: stage 1 does not predict beyond a fixed bound.
        """

        if self._pose is None:
            raise RuntimeError("Se3PoseFilter has not been corrected yet")
        horizon = self._prediction_horizon_s if horizon_s is None else float(horizon_s)
        clamped = max(0.0, min(horizon, self._horizon_limit_s))
        return lie.oplus(self._pose, self._velocity_body * clamped)

    def reset(self) -> None:
        self._pose = None
        self._velocity_body = np.zeros(6, dtype=np.float64)
        self._last_timestamp = None


class Se3EKF:
    """On-manifold EKF for the head pose and body twist (state dimension 12).

    State: ``(_pose`` SE(3) transform, ``_velocity_body`` 6-D body twist,
    ``_covariance`` 12x12). ``correct`` gates the incoming pose exactly like
    :class:`Se3PoseFilter`, then runs one constant-velocity predict-update step:

    - predict: ``T_pred = T ⊞ v·dt`` with ``F = [[Ad_{exp(-v·dt)}, dt·J_r(v·dt)^{-1}],
      [0, I]]`` and ``P = F·P·Fᵀ + Q`` (Q: velocity + small pose random walk).
    - update: ``y = z ⊖ T_pred``, ``S = H·P·Hᵀ + R`` (R scaled by
      ``(rms / rms_reference)²``), ``K = P·Hᵀ·S⁻¹``, ``δ = K·y``,
      ``T ⊞= δ[:6]``, ``v += δ[6:]``, ``P = (I - K·H)·P``.

    Measurement model is the identity on the pose block: ``H = [I₆, 0₆ₓ₆]``.
    """

    def __init__(
        self,
        *,
        prediction_horizon_s: float = DEFAULT_PREDICTION_HORIZON_S,
        horizon_limit_s: float = HORIZON_LIMIT_S,
        max_linear_speed_m_s: float = DEFAULT_MAX_LINEAR_SPEED_M_S,
        max_angular_speed_rad_s: float = DEFAULT_MAX_ANGULAR_SPEED_RAD_S,
        rms_reference_px: float = DEFAULT_RMS_REFERENCE_PX,
        measurement_rot_std_rad: float = DEFAULT_MEASUREMENT_ROT_STD_RAD,
        measurement_pos_std_m: float = DEFAULT_MEASUREMENT_POS_STD_M,
        angular_velocity_noise: float = DEFAULT_ANGULAR_VELOCITY_NOISE,
        linear_velocity_noise: float = DEFAULT_LINEAR_VELOCITY_NOISE,
        pose_angular_noise: float = DEFAULT_POSE_ANGULAR_NOISE,
        pose_linear_noise: float = DEFAULT_POSE_LINEAR_NOISE,
        initial_rot_std_rad: float = DEFAULT_INITIAL_ROT_STD_RAD,
        initial_pos_std_m: float = DEFAULT_INITIAL_POS_STD_M,
        initial_angular_vel_std_rad_s: float = DEFAULT_INITIAL_ANGULAR_VEL_STD_RAD_S,
        initial_linear_vel_std_m_s: float = DEFAULT_INITIAL_LINEAR_VEL_STD_M_S,
    ) -> None:
        if horizon_limit_s <= 0.0:
            raise ValueError("horizon_limit_s must be positive")
        if rms_reference_px <= 0.0:
            raise ValueError("rms_reference_px must be positive")
        self._prediction_horizon_s = prediction_horizon_s
        self._horizon_limit_s = horizon_limit_s
        self._max_linear_speed_m_s = max_linear_speed_m_s
        self._max_angular_speed_rad_s = max_angular_speed_rad_s
        self._rms_reference_px = rms_reference_px
        self._measurement_rot_std = measurement_rot_std_rad
        self._measurement_pos_std = measurement_pos_std_m
        self._angular_velocity_noise = angular_velocity_noise
        self._linear_velocity_noise = linear_velocity_noise
        self._pose_angular_noise = pose_angular_noise
        self._pose_linear_noise = pose_linear_noise
        initial_diagonal = np.array(
            [
                *([initial_rot_std_rad**2] * 3),
                *([initial_pos_std_m**2] * 3),
                *([initial_angular_vel_std_rad_s**2] * 3),
                *([initial_linear_vel_std_m_s**2] * 3),
            ],
            dtype=np.float64,
        )
        self._initial_covariance = np.diag(initial_diagonal)
        self._pose: FloatArray | None = None
        self._velocity_body = np.zeros(6, dtype=np.float64)
        self._covariance = np.zeros((12, 12), dtype=np.float64)
        self._last_timestamp: float | None = None

    @property
    def initialised(self) -> bool:
        return self._pose is not None

    @property
    def pose(self) -> FloatArray:
        if self._pose is None:
            raise RuntimeError("Se3EKF has not been corrected yet")
        return self._pose

    @property
    def velocity_body(self) -> FloatArray:
        return self._velocity_body

    @property
    def covariance(self) -> FloatArray:
        """12x12 covariance (rotation, position, angular vel, linear vel)."""

        return self._covariance

    def _process_noise(self, dt: float) -> FloatArray:
        """Discretised random-walk covariance ``Q`` for a ``dt`` step."""

        diagonal = np.array(
            [
                *([self._pose_angular_noise**2] * 3),
                *([self._pose_linear_noise**2] * 3),
                *([self._angular_velocity_noise**2] * 3),
                *([self._linear_velocity_noise**2] * 3),
            ],
            dtype=np.float64,
        )
        return np.diag(diagonal) * max(dt, 0.0)

    @staticmethod
    def _motion_jacobian(velocity: FloatArray, dt: float) -> FloatArray:
        """Error-state transition ``F`` of ``T ⊞= v·dt`` (constant velocity)."""

        transition = np.eye(12, dtype=np.float64)
        transition[:6, :6] = lie.adjoint_se3(lie.exp_se3(-velocity * dt))
        transition[:6, 6:] = dt * np.linalg.inv(lie.right_jacobian_se3(velocity * dt))
        return transition

    def _predict(self, dt: float) -> None:
        """Constant-velocity prediction: pose, twist, and covariance."""

        assert self._pose is not None
        transition = self._motion_jacobian(self._velocity_body, dt)
        self._covariance = transition @ self._covariance @ transition.T + self._process_noise(dt)
        self._pose = lie.oplus(self._pose, self._velocity_body * dt)
        self._pose[:3, :3] = _orthonormalize(self._pose[:3, :3])

    def _measurement_noise(self, reprojection_rms_px: float) -> FloatArray:
        """``R`` scaled by the reprojection RMS; lower RMS -> smaller R."""

        rms = max(float(reprojection_rms_px), 0.0)
        scale = max(_MIN_RMS_SCALE, (rms / self._rms_reference_px) ** 2)
        diagonal = np.array(
            [self._measurement_rot_std**2] * 3 + [self._measurement_pos_std**2] * 3,
            dtype=np.float64,
        )
        return np.diag(diagonal) * scale

    def _update(self, raw_pose: FloatArray, reprojection_rms_px: float) -> None:
        """Innovation ``y = z ⊖ T_pred`` and the standard EKF correction."""

        assert self._pose is not None
        innovation = lie.ominus(raw_pose, self._pose)
        measurement_covariance = self._measurement_noise(reprojection_rms_px)
        innovation_covariance = self._covariance[:6, :6] + measurement_covariance
        kalman_gain = self._covariance[:, :6] @ np.linalg.inv(innovation_covariance)
        correction = kalman_gain @ innovation
        self._pose = lie.oplus(self._pose, correction[:6])
        self._pose[:3, :3] = _orthonormalize(self._pose[:3, :3])
        self._velocity_body = self._velocity_body + correction[6:]
        identity = np.eye(12, dtype=np.float64)
        self._covariance = (
            identity - kalman_gain @ np.eye(6, 12, dtype=np.float64)
        ) @ self._covariance

    def correct(
        self,
        raw_pose: FloatArray,
        reprojection_rms_px: float,
        *,
        timestamp: float | None = None,
    ) -> bool:
        """Update the filter with a raw pose, returning False if rejected by the gate."""

        raw = np.asarray(raw_pose, dtype=np.float64)
        if raw.shape != (4, 4):
            raise ValueError("raw_pose must be a 4x4 transform")
        now = float(time.monotonic() if timestamp is None else timestamp)

        if self._pose is None or self._last_timestamp is None:
            self._pose = raw.copy()
            self._pose[:3, :3] = _orthonormalize(self._pose[:3, :3])
            self._velocity_body = np.zeros(6, dtype=np.float64)
            self._covariance = self._initial_covariance.copy()
            self._last_timestamp = now
            return True

        dt = now - self._last_timestamp
        if dt < _EPS_DT_S:
            # No time has passed; still run the update against the current pose.
            self._update(raw, reprojection_rms_px)
            return True

        instantaneous_twist = lie.ominus(raw, self._pose) / dt
        linear_speed = float(np.linalg.norm(instantaneous_twist[3:]))
        angular_speed = float(np.linalg.norm(instantaneous_twist[:3]))
        if (
            linear_speed > self._max_linear_speed_m_s
            or angular_speed > self._max_angular_speed_rad_s
        ):
            return False

        self._predict(dt)
        self._update(raw, reprojection_rms_px)
        self._last_timestamp = now
        return True

    def predict_to(self, horizon_s: float | None = None) -> FloatArray:
        """Extrapolate the pose by ``velocity_body * horizon`` (clamped to the limit)."""

        if self._pose is None:
            raise RuntimeError("Se3EKF has not been corrected yet")
        horizon = self._prediction_horizon_s if horizon_s is None else float(horizon_s)
        clamped = max(0.0, min(horizon, self._horizon_limit_s))
        return lie.oplus(self._pose, self._velocity_body * clamped)

    def reset(self) -> None:
        self._pose = None
        self._velocity_body = np.zeros(6, dtype=np.float64)
        self._covariance = np.zeros((12, 12), dtype=np.float64)
        self._last_timestamp = None
