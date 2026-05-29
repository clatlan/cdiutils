"""European XFEL I/O utilities."""

# TODO:
# Verify mapping between XFEL theta/chi/phi/twotheta
# and cdiutils sample/detector angle conventions.
from __future__ import annotations

from pathlib import Path

import numpy as np

from cdiutils.io.loader import H5TypeLoader


def xfel_safe_load(func):
    """Safe loader wrapper for XFEL directory/DAMNIT-based loading."""

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
        run_dir_name: str = "test_run",
        damnit_dir_name: str = "test_damnit",
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
            run_dir_name: Relative run directory name.
            damnit_dir_name: Relative DAMNIT database directory name.
            aliases_file_name: Alias file name inside the run directory.
            data_key: DAMNIT variable containing detector images.
            pulse_dimension: Xarray pulse dimension name.
            pulse_reduction: Reduction over pulse dimension:
                "mean", "sum", or None.
        """
        self.run_dir_name = run_dir_name
        self.damnit_dir_name = damnit_dir_name
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
        try:
            from extra.data import RunDirectory
        except ImportError as exc:
            raise ImportError(
                "XFELLoader requires EXtra-data. "
                "Install/use it in the European XFEL analysis environment."
            ) from exc

        run_dir = self.experiment_file_path / self.run_dir_name
        aliases_file = run_dir / self.aliases_file_name

        run = RunDirectory(run_dir)
        if aliases_file.exists():
            run = run.with_aliases(aliases_file)

        return run

    def _get_run_vars(self, scan: int = None):
        """Return DAMNIT variables for one XFEL run/scan."""
        try:
            from damnit import Damnit
        except ImportError as exc:
            raise ImportError(
                "XFELLoader requires DAMNIT to read processed XFEL variables."
            ) from exc

        scan, _ = self._check_scan_sample(scan, None)

        damnit_path = self.experiment_file_path / self.damnit_dir_name
        db = Damnit(damnit_path)

        return db[scan]

    def _read_images(self, scan: int = None):
        """Read the detector image variable from DAMNIT."""
        run_vars = self._get_run_vars(scan)

        try:
            return run_vars[self.data_key].read()
        except KeyError as exc:
            raise KeyError(
                f"DAMNIT variable {self.data_key!r} was not found "
                f"for scan/run {scan}."
            ) from exc

    def _reduce_pulses(self, images, pulse_reduction: str = None):
        """Reduce the XFEL pulse dimension of an xarray object."""
        reduction = (
            self.pulse_reduction
            if pulse_reduction is None
            else pulse_reduction
        )

        if reduction is None:
            return images

        if self.pulse_dimension not in images.dims:
            return images

        if reduction == "mean":
            return images.mean(self.pulse_dimension)

        if reduction == "sum":
            return images.sum(self.pulse_dimension)

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
        """Load photon energy from EXtra-data/XGM.

        Returns energy in eV.
        """
        try:
            from extra.components import XGM
        except ImportError as exc:
            raise ImportError(
                "XFELLoader requires EXtra-data components to read XGM energy."
            ) from exc

        run = self._get_run()
        energy = XGM(run).photon_energy()

        try:
            return float(energy.to("eV").magnitude)
        except AttributeError:
            energy = np.asarray(energy)
            value = float(np.nanmean(energy))

            # If value is probably in keV, convert to eV.
            if value < 100:
                value *= 1e3

            return value

    @xfel_safe_load
    def load_detector_shape(self, scan: int = None) -> tuple:
        """Return detector image shape after pulse reduction."""
        data = self.load_detector_data(scan=scan)

        if data.ndim != 3:
            raise ValueError(
                f"Expected detector data with 3 dimensions, got shape {data.shape}."
            )

        return data.shape[1:]

    def _get_scanned_motor(self, run):
        """Return the scanned motor DataCollection entry."""
        try:
            import extra as ex
            from extra.components import Scantool
        except ImportError as exc:
            raise ImportError(
                "XFELLoader requires EXtra-data to identify scanned motors."
            ) from exc

        sc = Scantool(run)

        motors = []
        missing_motors = list(sc.motor_devices.values())

        for motor_name in missing_motors.copy():
            if motor_name in run:
                motors.append(run[motor_name, "actualPosition"])
                missing_motors.remove(motor_name)
            else:
                property_name = (
                    ex.components.detector_motors.mangle_device_id_camelcase(
                        motor_name
                    )
                )
                property_name = f"{property_name}.actualPosition"

                for source_name in run.control_sources:
                    if run[source_name].device_class == "SlowDataSelector":
                        if property_name in run[source_name]:
                            motors.append(run[source_name, property_name])
                            missing_motors.remove(motor_name)
                            break

        if missing_motors:
            raise RuntimeError(
                f"Could not find these motors: {missing_motors}"
            )

        if not motors:
            raise RuntimeError("No scanned motor found.")

        return motors[0]

    def _get_scanned_motor_name(self, run, scanned_motor):
        """Infer scanned motor alias name: theta, chi, phi, or twotheta."""
        scanned_motor_aliases = []

        for alias_name, source_tuple in run._aliases.items():
            if source_tuple == (
                scanned_motor.source,
                scanned_motor.key.rstrip(".value"),
            ):
                scanned_motor_aliases.append(alias_name)

        for name in scanned_motor_aliases:
            if name in ("theta", "chi", "phi", "twotheta"):
                return name

        raise RuntimeError(
            "Could not infer scanned motor name from aliases. "
            f"Found aliases: {scanned_motor_aliases}"
        )

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
        try:
            from extra.components import Scan
        except ImportError as exc:
            raise ImportError(
                "XFELLoader requires EXtra-data components to load scan positions."
            ) from exc

        run = self._get_run()

        angles = {
            "theta": theta,
            "chi": chi,
            "phi": phi,
            "twotheta": twotheta,
        }

        scanned_motor = self._get_scanned_motor(run)
        scanned_motor_name = self._get_scanned_motor_name(run, scanned_motor)

        if angles[scanned_motor_name] is None:
            angles[scanned_motor_name] = Scan(scanned_motor).positions

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
