"""European XFEL I/O utilities."""

# TODO:
# Verify mapping between XFEL theta/chi/phi/twotheta
# and cdiutils sample/detector angle conventions.
from __future__ import annotations

from pathlib import Path

import numpy as np

from cdiutils.io.loader import H5TypeLoader

try:
    from extra_data import RunDirectory
except ImportError as exc:
    raise ImportError(
        "XFELLoader requires EXtra-data. "
        "Install/use it in the European XFEL analysis environment."
    ) from exc
try:
    from damnit import Damnit
except ImportError as exc:
    raise ImportError(
        "XFELLoader requires DAMNIT to read processed XFEL variables."
    ) from exc


def xfel_safe_load(func):
    """Safe loader wrapper for XFEL loading."""

    def wrap(self, *args, **kwargs):
        if not self.experiment_file_path.exists():
            raise FileNotFoundError(
                f"XFEL experiment path does not exist: {self.experiment_file_path}"
            )

        if not self.experiment_file_path.is_dir():
            raise NotADirectoryError(
                f"XFEL experiment path should be a directory: {self.experiment_file_path}"
            )

        return func(self, *args, **kwargs)

    return wrap


class XFELLoader(H5TypeLoader):
    """Loader for European XFEL DAMNIT/EXtra-data processed datasets."""

    angle_names = {
        "sample_outofplane_angle": "mu",
        "sample_inplane_angle": "omega",
        "detector_outofplane_angle": "gamma",
        "detector_inplane_angle": "delta",
    }

    authorised_detector_names = ("agipd", "jungfrau", "epix")

    def __init__(
        self,
        experiment_file_path: str,
        scan: int = None,
        sample_name: str = None,
        detector_name: str = None,
        flat_field: np.ndarray | str = None,
        alien_mask: np.ndarray | str = None,
        aliases_file_name: str = "extra-data-aliases.yml",
        data_key: str = "peak_images",
        pulse_dimension: str = "pulseIndex",
        pulse_reduction: str = "mean",
        **kwargs,
    ) -> None:
        """
        Initialise an XFEL loader.

        Args:
            experiment_file_path: Path to the XFEL scratch directory.
            scan: DAMNIT run/scan number.
            sample_name: Optional sample name.
            detector_name: Detector name, e.g. "agipd".
            flat_field: Optional flat-field correction.
            alien_mask: Optional detector mask.
            aliases_file_name: Alias file name inside the run directory.
            data_key: DAMNIT variable containing detector images.
            pulse_dimension: Xarray pulse dimension name.
            pulse_reduction: Reduction over pulse dimension:
                "mean", "sum", or None.
        """
        self.aliases_file_name = aliases_file_name
        self.data_key = data_key
        self.pulse_dimension = pulse_dimension
        self.pulse_reduction = pulse_reduction

        super().__init__(
            experiment_file_path,
            scan,
            sample_name,
            detector_name,
            flat_field,
            alien_mask,
        )

        self.experiment_file_path = Path(self.experiment_file_path)

    def _get_run(self):
        """Return the EXtra-data run object."""
        run_dir_name = self.sample_name or self.run_dir_name
        run_dir = self.experiment_file_path.parent / run_dir_name
        aliases_file = run_dir / self.aliases_file_name

        run = RunDirectory(run_dir)
        if aliases_file.exists():
            run = run.with_aliases(aliases_file)

        return run

    def _get_run_vars(self, scan: int = None):
        """Return DAMNIT variables for one XFEL run/scan."""
        scan, _ = self._check_scan_sample(scan, None)

        damnit_path = self.experiment_file_path
        db = Damnit(damnit_path)

        return db[scan]

    def _read_images(self, scan: int = None):
        """Read the detector image variable from DAMNIT."""
        run_vars = self._get_run_vars(scan)
        available_keys = list(run_vars.keys())
    
        try:
            return run_vars[self.data_key].read()
    
        except KeyError as exc:
    
            # Preferred fallback: xarray dataset containing detector images
            if "peak_dataset" in available_keys:
                ds = run_vars["peak_dataset"].read()
    
                if "images" in ds:
                    print(
                        f"\nDAMNIT variable {self.data_key!r} was not found."
                        "\nUsing fallback dataset: 'peak_dataset/images'"
                    )
                    return ds["images"]
    
            candidates = []
    
            for key in available_keys:
                try:
                    data = run_vars[key].read()
                    shape = np.shape(data)
    
                    if len(shape) in (3, 4):
                        candidates.append((key, shape))
    
                except Exception:
                    continue
    
            if candidates:
                print(
                    f"\nDAMNIT variable {self.data_key!r} was not found.\n"
                    "Available 3D/4D datasets:"
                )
    
                for key, shape in candidates:
                    print(f"  {key:30s} shape={shape}")
    
            raise KeyError(
                f"DAMNIT variable {self.data_key!r} was not found "
                f"for scan/run {scan}."
            ) from exc
    def _reduce_pulses(self, images, pulse_reduction: str = None):
        """Reduce the XFEL pulse dimension if present."""
        reduction = (
            self.pulse_reduction
            if pulse_reduction is None
            else pulse_reduction
        )
    
        if reduction is None:
            return images
    
        if not hasattr(images, "dims"):
            return images
    
        pulse_dim = self.pulse_dimension
    
        if pulse_dim not in images.dims:
            if "pulseId" in images.dims:
                pulse_dim = "pulseId"
            elif "pulseIndex" in images.dims:
                pulse_dim = "pulseIndex"
            else:
                return images
    
        if reduction == "mean":
            return images.mean(pulse_dim)
    
        if reduction == "sum":
            return images.sum(pulse_dim)
    
        raise ValueError("pulse_reduction should be 'mean', 'sum', or None.")

    @xfel_safe_load
    def load_detector_data(
        self,
        scan: int = None,
        sample_name: str = None,
        roi: tuple[slice] = None,
        rocking_angle_binning: int = None,
        binning_method: str = "sum",
        pulse_reduction: str = None,
    ) -> np.ndarray:
        """
        Load XFEL detector data as a cdiutils-compatible 3D array.

        Returns:
            np.ndarray with shape:
                (rocking_position, detector_y, detector_x)
        """
        scan, sample_name = self._check_scan_sample(scan, sample_name)

        images = self._read_images(scan)
        images = self._reduce_pulses(images, pulse_reduction)

        data = np.asarray(images)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        roi = self._check_roi(roi)
        data = data[roi]

        return self.bin_flat_mask(
            data,
            roi,
            self.flat_field,
            self.alien_mask,
            rocking_angle_binning,
            binning_method,
        )

    def _detect_scan_positions(
        self,
        motor_values,
        n_steps=None,
    ):
        """Detect scan-step positions from train-resolved motor values."""
        values = np.asarray(motor_values, dtype=float)
        values = values[np.isfinite(values)]

        if values.size == 0:
            return values

        if n_steps is None:
            raise ValueError(
                "n_steps is required for XFEL scan-position detection."
            )

        chunks = np.array_split(values, n_steps)

        return np.asarray(
            [np.nanmean(chunk) for chunk in chunks if chunk.size > 0]
        )

    @xfel_safe_load
    def load_motor_positions(
        self,
        scan: int = None,
        sample_name: str = None,
        roi: tuple[slice] = None,
        rocking_angle_binning: int = None,
        theta=None,
        chi=None,
        phi=None,
        twotheta=None,
        theta_offset: float = 0.0,
        chi_offset: float = 0.0,
        phi_offset: float = 0.0,
        twotheta_offset: float = 0.0,
    ) -> dict:
        """Load XFEL diffractometer angles."""
        scan, sample_name = self._check_scan_sample(scan, sample_name)

        angles, scanned_motor_name = self._load_geometry_angles(
            theta=theta,
            chi=chi,
            phi=phi,
            twotheta=twotheta,
            theta_offset=theta_offset,
            chi_offset=chi_offset,
            phi_offset=phi_offset,
            twotheta_offset=twotheta_offset,
        )

        self.rocking_angle = {
            "theta": "sample_outofplane_angle",
            "chi": "sample_inplane_angle",
            "phi": "sample_inplane_angle",
            "twotheta": "detector_inplane_angle",
        }[scanned_motor_name]

        # Apply ROI/binning only to the scanned array angle.
        scanned_angle = angles[scanned_motor_name]

        if rocking_angle_binning and np.ndim(scanned_angle) > 0:
            scanned_angle = self.bin_rocking_angle_values(
                scanned_angle,
                rocking_angle_binning,
            )

        if roi is not None and np.ndim(scanned_angle) > 0:
            if isinstance(roi, (tuple, list)) and len(roi) == 3:
                scanned_angle = scanned_angle[roi[0]]
            elif isinstance(roi, slice):
                scanned_angle = scanned_angle[roi]

        angles[scanned_motor_name] = scanned_angle

        return {
            "sample_outofplane_angle": angles["theta"],
            "sample_inplane_angle": angles["chi"],
            "detector_outofplane_angle": angles["phi"],
            "detector_inplane_angle": angles["twotheta"],
        }

    @xfel_safe_load
    def load_det_calib_params(self) -> dict:
        """Load XFEL detector calibration parameters."""
        run = self._get_run()
        sdd = np.nanmedian(run.alias["sdd"].ndarray())

        detector_shape = self.load_detector_shape()
        cch1 = detector_shape[0] // 2
        cch2 = detector_shape[1] // 2

        return {
            "distance": float(sdd),  # mm
            "pwidth1": 0.2,  # mm
            "pwidth2": 0.2,  # mm
            "cch1": cch1,
            "cch2": cch2,
            "detrot": 0,
            "tilt": 0,
            "tiltazimuth": 0,
        }

    @xfel_safe_load
    def load_energy(self, scan: int = None) -> float:
        """Load photon energy in eV."""
        run = self._get_run()
    
        # Direct energy aliases
        for alias in ("energy-kev", "undulator-energy"):
            if alias in run._aliases:
                energy = run.alias[alias].ndarray()
                energy = float(np.nanmean(energy))
    
                # If value looks like keV, convert to eV
                if energy < 100:
                    energy *= 1e3
    
                return energy
    
        # Wavelength fallback
        if "xgm-wavelength" in run._aliases:
            wavelength = run.alias["xgm-wavelength"].ndarray()
            wavelength = float(np.nanmean(wavelength))
    
            # Usually XFEL wavelength may be in m; convert m -> Å
            if wavelength < 1e-6:
                wavelength *= 1e10
    
            # E[eV] = hc / lambda[Å]
            return 12398.419843320026 / wavelength
    
        raise RuntimeError(
            "Could not load photon energy. Tried aliases: "
            "'energy-kev', 'undulator-energy', 'xgm-wavelength'.\n"
            f"Available aliases are:\n{sorted(run._aliases.keys())}"
        )

    @xfel_safe_load
    def load_detector_shape(self, scan: int = None) -> tuple:
        """Return detector image shape after pulse reduction."""
        data = self.load_detector_data(scan=scan)

        if data.ndim != 3:
            raise ValueError(
                f"Expected detector data with 3 dimensions, got shape {data.shape}."
            )

        return data.shape[1:]

    def _infer_scanned_motor_name(self, run, angles):
        """Infer scanned motor from theta/chi/phi/twotheta aliases."""
        candidates = {}

        for name in ("theta", "chi", "phi", "twotheta"):
            if angles[name] is not None:
                continue

            values = np.asarray(run.alias[name].ndarray(), dtype=float)
            diffs = np.diff(values)
            movement = np.nanmax(values) - np.nanmin(values)
            nonzero_steps = np.count_nonzero(np.abs(diffs) > 0)

            candidates[name] = (movement, nonzero_steps)
        if not candidates:
            raise ValueError(
                "Cannot infer scanned motor because all angles "
                "were provided explicitly."
            )
        scanned_motor_name = max(
            candidates, key=lambda key: candidates[key][0]
        )

        if candidates[scanned_motor_name][0] == 0:
            raise RuntimeError(
                "Could not infer scanned motor: theta, chi, phi, and twotheta "
                "appear constant."
            )

        return scanned_motor_name

    def _load_geometry_angles(
        self,
        theta=None,
        chi=None,
        phi=None,
        twotheta=None,
        theta_offset=0.0,
        chi_offset=0.0,
        phi_offset=0.0,
        twotheta_offset=0.0,
    ):
        """Load theta, chi, phi, and twotheta from EXtra-data aliases."""

        run = self._get_run()

        angles = {
            "theta": theta,
            "chi": chi,
            "phi": phi,
            "twotheta": twotheta,
        }

        images = self._read_images()

        if "position" in images.coords and angles["theta"] is None:
            scanned_motor_name = "theta"
            angles["theta"] = np.asarray(images.coords["position"].values)
        else:
            scanned_motor_name = self._infer_scanned_motor_name(run, angles)

            if angles[scanned_motor_name] is None:
                motor_values = run.alias[scanned_motor_name].ndarray()
                n_steps = np.asarray(images).shape[0]
                angles[scanned_motor_name] = self._detect_scan_positions(
                    motor_values,
                    n_steps=n_steps,
                )

        for key, value in angles.items():
            if value is None:
                angles[key] = run.alias[key].as_single_value()

        offsets = {
            "theta": theta_offset,
            "chi": chi_offset,
            "phi": phi_offset,
            "twotheta": twotheta_offset,
        }

        for key in angles:
            angles[key] = np.asarray(angles[key]) + offsets[key]

        return angles, scanned_motor_name


def load_xfel(
    experiment_file_path: str,
    scan: int = None,
    sample_name: str = None,
    detector_name: str = "agipd",
    **kwargs,
) -> np.ndarray:
    """Load XFEL detector data using XFELLoader."""
    loader = XFELLoader(
        experiment_file_path=experiment_file_path,
        scan=scan,
        sample_name=sample_name,
        detector_name=detector_name,
        **kwargs,
    )
    return loader.load_detector_data()
