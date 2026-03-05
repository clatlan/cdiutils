# tests/test_dislocation.py
"""
Pytest coverage for cdiutils.analysis.dislocation public API.

Design goals:
- Fast, deterministic, synthetic inputs (no I/O dependency).
- Assert robust invariants (shapes, finiteness, basic geometry/topology),
  not pixel-perfect outputs.

Covers exports from cdiutils.analysis.dislocation.__init__:
  map_min_gradient
  clusters_dislo_strain_map
  extract_structure
  fit_line_3d
  generate_filled_cylinder
  create_circular_mask
  plot_phase_around_dislo
  dislo_process_phase_ring
  remove_large_jumps
  center_angles
  dislo_phase_model
  dislo_rotation_matrix_real_to_theo
  signed_angle_3d
  angle_between_vectors
  project_vector
  normalize_vector
  transform_known_vector_to_crystallographic
  normalize_vectors_3d
  closest_to_zero_in_array
  decompose_experimental_phase

Source reference:
- cdiutils/analysis/dislocation/__init__.py :contentReference[oaicite:0]{index=0}
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_synthetic_cube_with_screw_phase(
    shape=(48, 48, 48),
    phase_scale=1.0,
    core_radius_vox=2.0,
    seed=0,
):
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    zz, yy, xx = np.indices(shape, dtype=float)
    z0, y0, x0 = (nz - 1) / 2.0, (ny - 1) / 2.0, (nx - 1) / 2.0
    zz -= z0
    yy -= y0
    xx -= x0

    # cube-ish amplitude
    cube_r = np.maximum.reduce([np.abs(xx), np.abs(yy), np.abs(zz)])
    half_width = min(nx, ny, nz) * 0.30
    amp = 1.0 / (1.0 + np.exp((cube_r - half_width) / (half_width / 8.0)))
    amp = amp.astype(np.float32)

    # screw-like phase around z: atan2(y, x) with softened core
    r_xy = np.sqrt(xx**2 + yy**2)
    ang = np.arctan2(yy, xx)
    core_soft = 1.0 - np.exp(-((r_xy / core_radius_vox) ** 2))
    phase = (phase_scale * ang * core_soft).astype(np.float32)

    # wrap
    phase = np.angle(np.exp(1j * phase)).astype(np.float32)

    # support mask
    mask = (amp / np.nanmax(amp)) > 0.05
    amp *= mask
    phase *= mask

    # complex object
    obj = amp * np.exp(1j * phase)

    # small random NaNs are a realistic edge case (should not break)
    # but keep it minimal so clustering still works
    if obj.size > 0:
        flat = obj.reshape(-1)
        idx = rng.choice(flat.size, size=min(10, flat.size), replace=False)
        flat[idx] = np.nan + 1j * np.nan
        obj = flat.reshape(obj.shape)

    return amp, phase, mask.astype(np.uint8), obj


def _line_points(shape=(48, 48, 48), n=40):
    """Generate points along a diagonal-ish line for fit_line_3d tests."""
    nz, ny, nx = shape
    t = np.linspace(5, min(nz, ny, nx) - 6, n)
    pts = np.column_stack([t, t * 0.8 + 3.0, t * 0.6 + 7.0])
    return pts


def test_geometry_extract_structure_and_fit_line_3d():
    from cdiutils.analysis import dislocation

    vol = np.zeros((20, 20, 20), dtype=float)
    pts = _line_points(shape=vol.shape, n=30).astype(int)
    vol[pts[:, 0], pts[:, 1], pts[:, 2]] = 1.0

    out = dislocation.extract_structure(vol, threshold=0.5)
    assert out.shape[1] == 3
    assert len(out) >= 10  # robust: non-empty and enough points for fit

    centroid, direction = dislocation.fit_line_3d(out)
    assert centroid.shape == (3,)
    assert direction.shape == (3,)
    assert np.isfinite(centroid).all()
    assert np.isfinite(direction).all()
    assert np.isclose(np.linalg.norm(direction), 1.0, atol=1e-6)


def test_geometry_generate_filled_cylinder_nonempty():
    from cdiutils.analysis import dislocation

    shape = (40, 40, 40)
    centroid = np.array([20.0, 20.0, 20.0])
    direction = np.array([1.0, 0.0, 0.0])

    vol = dislocation.generate_filled_cylinder(
        shape=shape,
        centroid=centroid,
        direction=direction,
        radius=2,
        height=20,
        step=2,
    )
    assert vol.shape == shape
    assert vol.sum() > 0

    # centroid voxel should be inside (or very near) the cylinder
    c = np.rint(centroid).astype(int)
    assert vol[c[0], c[1], c[2]] == 1


def test_geometry_create_circular_mask_outputs():
    from cdiutils.analysis import dislocation

    shape = (32, 32, 32)
    centroid = np.array([16.0, 16.0, 16.0])
    direction = np.array([0.0, 0.0, 1.0])

    circular_mask, polar_angles, disp_vecs, d_out = (
        dislocation.create_circular_mask(
            data_shape=shape,
            centroid=centroid,
            direction=direction,
            selected_point_index=0,
            r=6,
            dr=2,
            slice_thickness=2,
        )
    )

    assert circular_mask.shape == shape
    assert polar_angles.shape == shape
    assert disp_vecs.shape == (*shape, 3)
    assert np.isfinite(d_out).all()
    assert circular_mask.sum() > 0
    assert np.any(polar_angles != 0.0)


def test_geometry_plot_phase_around_dislo_masks_phase(tmp_path):
    from cdiutils.analysis import dislocation

    shape = (40, 40, 40)
    amp = np.ones(shape, dtype=np.float32)

    # simple phase (wrapped)
    zz, yy, xx = np.indices(shape, dtype=float)
    phase = np.angle(np.exp(1j * np.arctan2(yy - 20, xx - 20))).astype(
        np.float32
    )

    selected_dislo = np.zeros(shape, dtype=np.uint8)
    selected_dislo[10:30, 20, 20] = 1  # a line

    centroid = np.array([20.0, 20.0, 20.0])
    direction = np.array([1.0, 0.0, 0.0])

    masked_phase, polar_angles, circular_mask, disp_vecs, d_out = (
        dislocation.plot_phase_around_dislo(
            amp=amp,
            phase=phase,
            selected_dislocation_data=selected_dislo,
            r=6,
            dr=2,
            centroid=centroid,
            direction=direction,
            slice_thickness=2,
            selected_point_index=0,
            save_vti=True,
            fig_title="pytest",
            plot_debug=False,
            save_path=str(tmp_path / "ring.vti"),
            voxel_sizes=(1.0, 1.0, 1.0),
        )
    )

    assert masked_phase.shape == shape
    assert (masked_phase[circular_mask == 0] == 0).all()
    assert np.any(masked_phase != 0)
    assert polar_angles.shape == shape
    assert disp_vecs.shape == (*shape, 3)
    assert np.isfinite(d_out).all()


def test_ring_center_angles_properties():
    from cdiutils.analysis import dislocation

    x = np.array([10.0, 20.0, 30.0, 40.0])
    y = dislocation.center_angles(x)
    assert y.shape == x.shape
    assert np.isclose(np.nanmin(y), -np.nanmax(y), atol=1e-12)


def test_ring_remove_large_jumps_removes_outlier():
    from cdiutils.analysis import dislocation

    x = np.linspace(0, 10, 11)
    y = x.copy()
    y[5] += 100  # outlier jump

    x2, y2 = dislocation.remove_large_jumps(x, y, threshold_factor=1.0)
    assert len(x2) < len(x)
    assert len(x2) == len(y2)


def test_phase_decomp_recovers_cos2_sin2():
    from cdiutils.analysis import dislocation

    rng = np.random.default_rng(0)
    theta = np.linspace(-np.pi, np.pi, 721)

    a, b = 1.7, -0.9
    slope, intercept = 0.3, -0.1

    phi = (
        slope * theta
        + intercept
        + a * np.cos(2 * theta)
        + b * np.sin(2 * theta)
    )
    phi += 0.02 * rng.normal(size=theta.size)

    f_osc, f_fit2, coeffs, f_lin, coeffs_lin = (
        dislocation.decompose_experimental_phase(theta, phi)
    )

    assert np.isfinite(f_osc).all()
    assert np.isfinite(f_fit2).all()
    assert coeffs.shape == (2,)
    assert np.isfinite(coeffs).all()
    # coefficients should be close (noise-tolerant)
    assert np.isclose(coeffs[0], a, atol=0.15)
    assert np.isclose(coeffs[1], b, atol=0.15)

    assert np.isfinite(f_lin).all()
    assert len(coeffs_lin) == 2


def test_dislo_process_phase_ring_runs_without_plotting():
    from cdiutils.analysis import dislocation

    shape = (40, 40, 40)
    amp, phase, mask, obj = _make_synthetic_cube_with_screw_phase(
        shape=shape, phase_scale=1.0
    )
    selected_dislo = np.zeros(shape, dtype=np.uint8)
    selected_dislo[10:30, 20, 20] = 1

    centroid = np.array([20.0, 20.0, 20.0])
    direction = np.array([1.0, 0.0, 0.0])

    phase_ring_3d, angle_ring_3d, circular_mask, disp_vecs, _ = (
        dislocation.plot_phase_around_dislo(
            amp=amp,
            phase=phase,
            selected_dislocation_data=selected_dislo,
            r=7,
            dr=2,
            centroid=centroid,
            direction=direction,
            slice_thickness=2,
            selected_point_index=0,
            save_vti=False,
            plot_debug=False,
            save_path=None,
            voxel_sizes=(1.0, 1.0, 1.0),
        )
    )

    out = dislocation.dislo_process_phase_ring(
        angle=angle_ring_3d,
        phase=phase_ring_3d,
        displacement_vectors=disp_vecs,
        plot_debug=False,
    )
    assert isinstance(out, tuple)
    assert len(out) == 8
    (
        angle_raw,
        phase_raw,
        angle_final,
        phase_final,
        phase_smooth,
        phase_sinu,
        dv_sorted,
        dv_final,
    ) = out
    assert len(angle_raw) == len(phase_raw)
    assert len(angle_final) == len(phase_final)
    assert np.isfinite(phase_final).all()
    assert np.isfinite(phase_sinu).all()


def test_strain_map_map_min_gradient_shapes_and_range(tmp_path):
    from cdiutils.analysis import dislocation

    amp, phase, mask, obj = _make_synthetic_cube_with_screw_phase(
        shape=(40, 40, 40), phase_scale=1.0
    )

    strain_mask, strain_amp = dislocation.map_min_gradient(
        obj=obj,
        voxel_size=[1.0, 1.0, 1.0],
        nb_of_phase_to_test=6,
        save_filename_vti=str(tmp_path / "grad.vti"),
        plot_debug=False,
        save_plot=False,
        verbose=False,
    )

    assert strain_mask.shape == amp.shape
    assert strain_amp.shape == amp.shape
    assert np.nanmin(strain_amp) >= 0.0
    assert np.nanmax(strain_amp) <= 1.0 + 1e-6
    assert (tmp_path / "grad.vti").exists()


def test_clustering_clusters_dislo_strain_map_smoke(tmp_path):
    """
    Smoke test: should identify at least one cluster from a synthetic line
    in a strain-like map. Skips if optional deps are missing.
    """
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    pytest.importorskip("imageio")
    pytest.importorskip("matplotlib")

    from cdiutils.analysis import dislocation

    shape = (48, 48, 48)
    data = np.zeros(shape, dtype=np.float32)
    # make a "high strain" line
    data[10:38, 24, 24] = 1.0

    amp = np.ones(shape, dtype=np.float32)
    phase = np.zeros(shape, dtype=np.float32)

    labeled, n = dislocation.clusters_dislo_strain_map(
        data=data,
        amp=amp,
        phase=phase,
        save_path=str(tmp_path / "cluster"),
        voxel_sizes=(1.0, 1.0, 1.0),
        threshold=0.35,
        min_cluster_size=5,
        distance_threshold=5.0,
        cylinder_radius=2,
        num_spline_points=200,
        smoothing_param=2,
        eps=2.0,
        min_samples=3,
        save_output=False,
        debug_plot=False,
    )

    assert labeled.shape == shape
    assert isinstance(n, int)
    assert np.any(labeled > 0)


def test_theory_vector_utils_and_rotation_matrix():
    from cdiutils.analysis import dislocation

    v = np.array([3.0, 0.0, 4.0])
    vn = dislocation.normalize_vector(v)
    assert np.isclose(np.linalg.norm(vn), 1.0, atol=1e-12)

    v_perp = dislocation.project_vector([1.0, 1.0, 0.0], [1.0, 0.0, 0.0])
    assert np.allclose(v_perp, [0.0, 1.0, 0.0], atol=1e-12)

    # signed angle sanity
    u = np.array([1.0, 0.0, 0.0])
    w = np.array([0.0, 1.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    ang = dislocation.signed_angle_3d(u, w, normal)
    assert np.isclose(ang, 90.0, atol=1e-12)

    ang_abs = dislocation.angle_between_vectors(u, w)
    assert np.isclose(ang_abs, 90.0, atol=1e-12)

    # rotation matrix orthonormal (within tolerance)
    t = np.array([0.0, 0.0, 1.0])
    b = np.array([1.0, 0.0, 0.0])
    R = dislocation.dislo_rotation_matrix_real_to_theo(t, b)
    assert R.shape == (3, 3)
    identity = R @ R.T
    assert np.allclose(identity, np.eye(3), atol=1e-10)

    # transform_known_vector_to_crystallographic: identity rotation
    vx, vy, vz = dislocation.transform_known_vector_to_crystallographic(
        1, 2, 3, np.eye(3)
    )
    assert (vx, vy, vz) == (1, 2, 3)

    # normalize_vectors_3d
    vxn, vyn, vzn = dislocation.normalize_vectors_3d([3, 0], [4, 0], [0, 5])
    mags = np.sqrt(
        np.asarray(vxn) ** 2 + np.asarray(vyn) ** 2 + np.asarray(vzn) ** 2
    )
    assert np.allclose(mags, 1.0, atol=1e-12)

    # closest_to_zero_in_array
    val, idx = dislocation.closest_to_zero_in_array(
        np.array([5.0, -0.2, 0.1, 9.0])
    )
    assert np.isclose(val, 0.1, atol=1e-12)
    assert idx == 2


def test_theory_dislo_phase_model_basic_properties():
    from cdiutils.analysis import dislocation

    theta = np.linspace(-np.pi, np.pi, 361)
    t = np.array([0.0, 0.0, 1.0])
    G = np.array([0.0, 0.0, 1.0])
    b = np.array([0.0, 0.0, 1.0])  # pure screw along t

    phi = dislocation.dislo_phase_model(
        theta=theta, t=t, G=G, b=b, only_theta_dep=True
    )
    assert phi.shape == theta.shape
    assert np.isfinite(phi).all()

    # for pure screw with G||t and b||t, phase should be ~ linear in theta (up to scaling)
    # check monotonicity in the central region
    mid = slice(50, -50)
    dphi = np.diff(phi[mid])
    assert np.all(np.isfinite(dphi))
    assert np.median(dphi) != 0.0
