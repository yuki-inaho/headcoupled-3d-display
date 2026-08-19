"""Lie algebra helpers for SE(3) pose filtering (arXiv:1812.01537).

Right-perturbation convention: X ⊞ u = X exp(u). Tangent vectors are np.ndarray
shape (6,) with ``xi[:3] = phi`` (rotation) and ``xi[3:] = rho`` (translation).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from headcoupled_display import lie

FloatArray = NDArray[np.float64]

_EPS = 1e-8


def _skew(v: FloatArray) -> FloatArray:
    return lie.hat_so3(v)


def _random_rotation(rng: np.random.Generator) -> FloatArray:
    # Random axis-angle with |phi| < pi keeps log unique.
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis) + 1e-12
    angle = rng.uniform(0.0, np.pi - 0.1)
    return lie.exp_so3(axis * angle)


def _random_transform(rng: np.random.Generator) -> FloatArray:
    rotation = _random_rotation(rng)
    translation = rng.standard_normal(3)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def test_hat_vee_roundtrip() -> None:
    vector = np.array([0.1, -0.2, 0.3])
    assert np.allclose(lie.vee_so3(lie.hat_so3(vector)), vector)


def test_skew_is_antisymmetric() -> None:
    skew = _skew(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(skew, -skew.T)


def test_exp_so3_identity_for_zero() -> None:
    assert np.allclose(lie.exp_so3(np.zeros(3)), np.eye(3))


def test_log_so3_identity_for_identity() -> None:
    assert np.allclose(lie.log_so3(np.eye(3)), np.zeros(3))


def test_exp_log_so3_roundtrip() -> None:
    rng = np.random.default_rng(42)
    for _ in range(50):
        rotation = _random_rotation(rng)
        recovered = lie.exp_so3(lie.log_so3(rotation))
        assert np.allclose(recovered, rotation, atol=1e-9)


def test_log_exp_so3_roundtrip() -> None:
    rng = np.random.default_rng(7)
    for _ in range(50):
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis) + 1e-12
        phi = axis * rng.uniform(0.0, np.pi - 0.1)
        assert np.allclose(lie.log_so3(lie.exp_so3(phi)), phi, atol=1e-9)


def test_exp_so3_small_angle_series() -> None:
    # Near-zero rotation must not divide by ~0.
    phi = np.array([1e-10, -2e-10, 3e-10])
    rotation = lie.exp_so3(phi)
    assert np.allclose(rotation, np.eye(3) + _skew(phi), atol=1e-12)


def test_exp_se3_identity_for_zero() -> None:
    assert np.allclose(lie.exp_se3(np.zeros(6)), np.eye(4))


def test_exp_se3_pure_translation() -> None:
    xi = np.array([0.0, 0.0, 0.0, 0.1, -0.2, 0.3])
    transform = lie.exp_se3(xi)
    expected = np.eye(4)
    expected[:3, 3] = np.array([0.1, -0.2, 0.3])
    assert np.allclose(transform, expected)


def test_exp_log_se3_roundtrip() -> None:
    rng = np.random.default_rng(123)
    for _ in range(50):
        transform = _random_transform(rng)
        recovered = lie.exp_se3(lie.log_se3(transform))
        assert np.allclose(recovered, transform, atol=1e-9)


def test_log_exp_se3_roundtrip() -> None:
    rng = np.random.default_rng(99)
    for _ in range(50):
        transform = _random_transform(rng)
        xi = lie.log_se3(transform)
        assert np.allclose(lie.exp_se3(xi), transform, atol=1e-9)


def test_oplus_ominus_inverse() -> None:
    rng = np.random.default_rng(2024)
    x = _random_transform(rng)
    y = _random_transform(rng)
    # oplus(X, ominus(Y, X)) == Y
    delta = lie.ominus(y, x)
    assert np.allclose(lie.oplus(x, delta), y, atol=1e-9)


def test_ominus_oplus_inverse() -> None:
    rng = np.random.default_rng(55)
    x = _random_transform(rng)
    xi = rng.standard_normal(6) * 0.1
    assert np.allclose(lie.ominus(lie.oplus(x, xi), x), xi, atol=1e-9)


def test_oplus_right_perturbation() -> None:
    # X ⊞ u = X @ exp(u): a pure-translation tangent shifts the translation by R@rho.
    x = np.eye(4)
    x[:3, 3] = np.array([1.0, 2.0, 3.0])
    x[:3, :3] = lie.exp_so3(np.array([0.0, 0.0, np.pi / 2]))
    xi = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    result = lie.oplus(x, xi)
    # R rotates +x (display) to +y, so translation becomes [1, 2+1, 3].
    assert np.allclose(result[:3, 3], np.array([1.0, 3.0, 3.0]), atol=1e-9)


def test_right_jacobian_identity_at_zero() -> None:
    assert np.allclose(lie.right_jacobian_so3(np.zeros(3)), np.eye(3))


def test_right_jacobian_matches_numerical() -> None:
    # Definitional identity (Forster/Solà): exp(phi+delta) ≈ exp(phi) exp(J_r^-1 delta).
    rng = np.random.default_rng(314)
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis) + 1e-12
    phi = axis * 1.2
    delta = rng.standard_normal(3) * 1e-6
    left = lie.exp_so3(phi + delta)
    right = lie.exp_so3(phi) @ lie.exp_so3(np.linalg.inv(lie.right_jacobian_so3(phi)) @ delta)
    assert np.allclose(left, right, atol=1e-8)


def test_right_jacobian_left_is_negative() -> None:
    # J_l(phi) = J_r(-phi).
    rng = np.random.default_rng(1)
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis) + 1e-12
    phi = axis * 0.7
    assert np.allclose(lie.right_jacobian_so3(-phi), lie.right_jacobian_so3(phi).T, atol=1e-9)


def test_right_jacobian_se3_identity_at_zero() -> None:
    assert np.allclose(lie.right_jacobian_se3(np.zeros(6)), np.eye(6))


def test_right_jacobian_se3_matches_numerical() -> None:
    # Definitional identity (same style as the SO(3) test):
    #   exp(xi + delta) ≈ exp(xi) @ exp(J_r(xi)^{-1} delta)
    # i.e. log(exp(xi) exp(delta)) ≈ xi + J_r(xi)^{-1} delta  (delta on the right).
    rng = np.random.default_rng(2718)
    for _ in range(30):
        transform = _random_transform(rng)
        xi = lie.log_se3(transform)
        delta = rng.standard_normal(6) * 1e-6
        left = lie.exp_se3(xi + delta)
        right = lie.exp_se3(xi) @ lie.exp_se3(np.linalg.inv(lie.right_jacobian_se3(xi)) @ delta)
        assert np.allclose(left, right, atol=1e-8)


def test_right_jacobian_se3_small_angle_series() -> None:
    # Near-zero rotation must not divide by ~0. The rotation blocks collapse to I;
    # the Q block legitimately depends on rho, so only the rotation blocks are checked.
    xi = np.array([1e-10, -2e-10, 3e-10, 0.02, -0.01, 0.03])
    jacobian = lie.right_jacobian_se3(xi)
    assert np.allclose(jacobian[:3, :3], np.eye(3), atol=1e-9)
    assert np.allclose(jacobian[3:, 3:], np.eye(3), atol=1e-9)


def test_adjoint_homomorphism() -> None:
    rng = np.random.default_rng(2718)
    t1 = _random_transform(rng)
    t2 = _random_transform(rng)
    expected = lie.adjoint_se3(t1) @ lie.adjoint_se3(t2)
    actual = lie.adjoint_se3(t1 @ t2)
    assert np.allclose(actual, expected, atol=1e-8)


def test_adjoint_identity() -> None:
    assert np.allclose(lie.adjoint_se3(np.eye(4)), np.eye(6))


def test_adjoint_transforms_tangent() -> None:
    # Ad_T transforms a body-frame tangent; round-trip via Ad_{T^-1}.
    rng = np.random.default_rng(1234)
    transform = _random_transform(rng)
    xi = rng.standard_normal(6) * 0.05
    moved = lie.adjoint_se3(transform) @ xi
    back = lie.adjoint_se3(np.linalg.inv(transform)) @ moved
    assert np.allclose(back, xi, atol=1e-9)
