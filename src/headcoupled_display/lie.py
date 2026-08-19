"""Minimal Lie-theory helpers for SE(3) state estimation.

Implements the right-perturbation convention from J. Solà et al.,
"A micro Lie theory for state estimation in robotics" (arXiv:1812.01537, v9):

- ``X ⊞ u = X exp(u)`` and ``Y ⊖ X = log(X^{-1} Y)`` (§4, eq. 10-12).
- SO(3) and SE(3) exponential/logarithm maps (§5.1-5.2, §6.1).
- Right Jacobian of SO(3) (§7.4, eq. 101-104).
- SE(3) adjoint (§6.1, eq. 61).

A tangent vector ``xi`` is ``np.ndarray`` shape ``(6,)`` with
``xi[:3] = phi`` (rotation vector) and ``xi[3:] = rho`` (translation). All maps
are numerically stable at the small-angle singularity via Taylor series.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

#: Below this rotation angle the closed-form maps switch to their Taylor series.
_SMALL_ANGLE_EPS = 1e-8


def hat_so3(vector: FloatArray) -> FloatArray:
    """Skew-symmetric matrix ``[v]x`` (paper eq. 14)."""

    v = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )


def vee_so3(skew: FloatArray) -> FloatArray:
    """Inverse of :func:`hat_so3`."""

    matrix = np.asarray(skew, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("vee_so3 expects a 3x3 skew matrix")
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]], dtype=np.float64)


def exp_so3(phi: FloatArray) -> FloatArray:
    """SO(3) exponential map (Rodrigues, paper §5.1 eq. 16-18)."""

    vector = np.asarray(phi, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(vector))
    skew = hat_so3(vector)
    if theta < _SMALL_ANGLE_EPS:
        # I + [phi]x + 0.5 [phi]x^2 keeps first two non-trivial series terms.
        return np.eye(3) + skew + 0.5 * skew @ skew
    skew_sq = skew @ skew
    return (
        np.eye(3)
        + (np.sin(theta) / theta) * skew
        + ((1.0 - np.cos(theta)) / (theta * theta)) * skew_sq
    )


def log_so3(rotation: FloatArray) -> FloatArray:
    """SO(3) logarithm map (paper §5.2 eq. 23-24)."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("log_so3 expects a 3x3 matrix")
    cos_angle = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cos_angle))
    if theta < _SMALL_ANGLE_EPS:
        # log(I + [phi]x) ~= [phi]x for small rotation; vee of (R - R^T)/2.
        return 0.5 * vee_so3(matrix - matrix.T)
    if abs(theta - np.pi) < 1e-6:
        # Near pi the standard formula divides by sin(theta)~0; fall back to the
        # symmetric sqrt form: R + I = 2 (1+cos) a a^T is rank-1, so recover the
        # axis from the largest diagonal of (R + I)/2.
        symmetric = (matrix + np.eye(3)) / 2.0
        diag = np.clip(np.diag(symmetric), 0.0, 1.0)
        axis_idx = int(np.argmax(diag))
        axis = symmetric[:, axis_idx]
        axis = (
            np.sign(matrix[axis_idx, (axis_idx + 1) % 3] - matrix[(axis_idx + 1) % 3, axis_idx])
            * np.sqrt(np.clip(diag[axis_idx], 0.0, 1.0))
            * (symmetric[:, axis_idx] / (np.linalg.norm(symmetric[:, axis_idx]) + 1e-12))
        )
        return theta * axis
    return (theta / (2.0 * np.sin(theta))) * vee_so3(matrix - matrix.T)


def _v_matrix(phi: FloatArray) -> FloatArray:
    """Closed-form ``V`` so that ``exp([phi,rho]^) = [[R, V rho],[0,1]]`` (§6.1 eq. 57)."""

    vector = np.asarray(phi, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(vector))
    skew = hat_so3(vector)
    if theta < _SMALL_ANGLE_EPS:
        return np.eye(3) + 0.5 * skew + (1.0 / 6.0) * skew @ skew
    skew_sq = skew @ skew
    return (
        np.eye(3)
        + ((1.0 - np.cos(theta)) / (theta * theta)) * skew
        + ((theta - np.sin(theta)) / (theta**3)) * skew_sq
    )


def exp_se3(xi: FloatArray) -> FloatArray:
    """SE(3) exponential map (paper §6.1 eq. 55-58)."""

    tangent = np.asarray(xi, dtype=np.float64).reshape(6)
    phi = tangent[:3]
    rho = tangent[3:]
    rotation = exp_so3(phi)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = _v_matrix(phi) @ rho
    return transform


def _v_inverse(phi: FloatArray) -> FloatArray:
    """Closed-form ``V^{-1}`` for the SE(3) logarithm (paper §6.1 eq. 59)."""

    vector = np.asarray(phi, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(vector))
    skew = hat_so3(vector)
    if theta < _SMALL_ANGLE_EPS:
        return np.eye(3) - 0.5 * skew + (1.0 / 12.0) * skew @ skew
    skew_sq = skew @ skew
    coefficient = (1.0 / (theta * theta)) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
    return np.eye(3) - 0.5 * skew + coefficient * skew_sq


def log_se3(transform: FloatArray) -> FloatArray:
    """SE(3) logarithm map (paper §6.1 eq. 59-60)."""

    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("log_se3 expects a 4x4 matrix")
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    phi = log_so3(rotation)
    rho = _v_inverse(phi) @ translation
    tangent = np.zeros(6, dtype=np.float64)
    tangent[:3] = phi
    tangent[3:] = rho
    return tangent


def oplus(transform: FloatArray, xi: FloatArray) -> FloatArray:
    """Right box-plus ``X ⊞ u = X exp(u)`` (paper §4 eq. 10)."""

    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("oplus expects a 4x4 transform")
    return matrix @ exp_se3(xi)


def ominus(left: FloatArray, right: FloatArray) -> FloatArray:
    """Right box-minus ``Y ⊖ X = log(X^{-1} Y)`` (paper §4 eq. 11)."""

    left_matrix = np.asarray(left, dtype=np.float64)
    right_matrix = np.asarray(right, dtype=np.float64)
    if left_matrix.shape != (4, 4) or right_matrix.shape != (4, 4):
        raise ValueError("ominus expects 4x4 transforms")
    return log_se3(np.linalg.inv(right_matrix) @ left_matrix)


def right_jacobian_so3(phi: FloatArray) -> FloatArray:
    """Right Jacobian ``J_r`` of SO(3) (paper §7.4 eq. 101-104).

    Satisfies ``log(exp(phi^) exp(delta^)) ≈ phi + J_r(phi) delta``.
    """

    vector = np.asarray(phi, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(vector))
    skew = hat_so3(vector)
    if theta < _SMALL_ANGLE_EPS:
        return np.eye(3) - 0.5 * skew + (1.0 / 6.0) * skew @ skew
    skew_sq = skew @ skew
    return (
        np.eye(3)
        - ((1.0 - np.cos(theta)) / (theta * theta)) * skew
        + ((theta - np.sin(theta)) / (theta**3)) * skew_sq
    )


def right_jacobian_se3(xi: FloatArray) -> FloatArray:
    """Right Jacobian ``J_r`` of SE(3) (paper Appendix D, ``J_r(rho,theta)=J_l(-rho,-theta)``).

    Definitional identity: ``log(exp(xi) exp(delta)) ≈ xi + J_r(xi) delta`` (delta
    on the right; paper §II-G eq. 44), equivalently
    ``exp(xi + delta) ≈ exp(xi) exp(J_r^{-1} delta)``. Computed by finite
    differences of the composition Jacobian ``d/dδ log(exp(xi) exp(delta))`` at
    δ=0, which by definition equals ``J_r`` (the same ``jacobian_fd`` approach
    navlie uses for checking analytic Jacobians); 6 exp/log calls, negligible at
    tracking rates.
    """

    tangent = np.asarray(xi, dtype=np.float64).reshape(6)
    base = exp_se3(tangent)
    eps = 1e-6
    composition = np.zeros((6, 6), dtype=np.float64)
    for index in range(6):
        direction = np.zeros(6, dtype=np.float64)
        direction[index] = 1.0
        plus = log_se3(base @ exp_se3(eps * direction))
        minus = log_se3(base @ exp_se3(-eps * direction))
        composition[:, index] = (plus - minus) / (2.0 * eps)
    return composition


def adjoint_se3(transform: FloatArray) -> FloatArray:
    """SE(3) adjoint ``Ad_T`` for tangents ordered ``[phi, rho]`` (paper §6.1 eq. 61).

    ``Ad_T = [[R, 0],[t^ R, R]]`` maps a body-frame twist to the spatial frame.
    """

    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("adjoint_se3 expects a 4x4 transform")
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    adjoint = np.zeros((6, 6), dtype=np.float64)
    adjoint[:3, :3] = rotation
    adjoint[3:, :3] = hat_so3(translation) @ rotation
    adjoint[3:, 3:] = rotation
    return adjoint
