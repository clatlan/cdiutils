import copy
import numbers
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import center_of_mass
from xrayutilities.analysis.sample_align import area_detector_calib

# xrayutilities imports
from xrayutilities.experiment import HXRD, QConversion
from xrayutilities.gridder3d import FuzzyGridder3D

from cdiutils import __version__
from cdiutils.geometry import Geometry
from cdiutils.utils import energy_to_wavelength


class SpaceConverter:
    """
    BCDI-critical class for coordinate system transformations.

    Handles conversions between detector, reciprocal lab (q-space),
    and direct lab frames using xrayutilities. Essential for BCDI
    strain calculations, where accurate orthogonalisation preserves
    physical displacement fields.

    The class manages three coordinate systems:
    - Detector frame: raw (i,j,k) pixel indices (non-orthogonal)
    - Reciprocal lab: orthogonal q-space (units: Angstroms-1)
    - Direct lab: orthogonal real-space (units: nm)

    See Also:
        user_guide/coordinate_systems.rst for CXI/XU conventions.
    """

    def __init__(
        self,
        geometry: Geometry,
        det_calib_params: dict = None,
        energy: float = None,
        roi: list = None,
        shape: tuple = None,
        q_lab_shift: tuple = None,
        q_lab_matrix: np.ndarray = None,
        direct_lab_matrix: np.ndarray = None,
        direct_lab_voxel_size: tuple = None,
    ):
        """
        Initialise the space converter.

        Args:
            geometry: beamline geometry (goniometer circles,
                detector orientation).
            det_calib_params: detector calibration (cch1, cch2,
                distance, pwidth1, pwidth2) in pixels and mm. Critical
                for accurate q-space mapping.
            energy: X-ray energy in eV.
            roi: region of interest [row_start, row_end, col_start,
                col_end] applied to detector data.
            shape: data shape (nz, ny, nx) in detector frame.
            q_lab_shift: shift applied to centre Bragg peak at origin
                in reciprocal space (Angstroms-1).
            q_lab_matrix: transformation matrix for q-space
                interpolation (internal use).
            direct_lab_matrix: transformation matrix for direct-space
                interpolation (internal use).
            direct_lab_voxel_size: voxel size in direct lab frame (nm).
        """
        self.geometry = geometry
        self.det_calib_params = det_calib_params
        self.energy = energy
        self.roi = roi
        self.angles: dict = None
        self.q_lab_shift = q_lab_shift

        self.q_lab_interpolator: Interpolator3D = None
        self.direct_lab_interpolator: Interpolator3D = None
        if shape is not None and q_lab_matrix is not None:
            self.q_lab_interpolator = Interpolator3D(shape, q_lab_matrix)

        if shape is not None and direct_lab_matrix is not None:
            self.direct_lab_interpolator = Interpolator3D(
                shape, direct_lab_matrix, direct_lab_voxel_size
            )

        # convert the geometry to xrayutilities coordinate system
        if self.geometry.is_cxi:
            self.geometry.cxi_to_xu()

        self.hxrd = None

        # attributes that must be protected
        self._shape = shape
        self._q_space_transitions = None

        self.xu_gridder: FuzzyGridder3D = None

        # particular attribute that is also defined by its setter.
        self.direct_lab_voxel_size = direct_lab_voxel_size

    @property
    def direct_lab_voxel_size(self):
        if self._direct_lab_voxel_size is None:
            return None
        return tuple(float(e) for e in self._direct_lab_voxel_size)

    @direct_lab_voxel_size.setter
    def direct_lab_voxel_size(self, size: tuple | float | int):
        """
        This setter will reinitialise the direct_lab_interpolator if it
        has been initialised beforehand.
        """
        if isinstance(size, (numbers.Number)):
            size = tuple(np.repeat(size, len(self._shape)))

        if (
            self.direct_lab_interpolator is not None
            and not np.array_equal(
                size, self.direct_lab_interpolator.target_voxel_size
            )
            and self._q_space_transitions is not None
        ):
            print("Reinitialising the direct space interpolator")
            self.init_interpolator(size, space="direct")
        self._direct_lab_voxel_size = size

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        if self._q_space_transitions[0].shape != shape:
            raise ValueError(
                "Setting up the shape is not possible. The shape should"
                "only be defined by the input detector data."
                f"({self._q_space_transitions[0].shape = } vs. {shape = })."  # noqa E251
            )

    def to_dict(self) -> dict:
        """Serialise converter state to dictionary."""
        d = {
            k: self.__dict__[k]
            for k in (
                "energy",
                "roi",
                "q_lab_shift",
                "_shape",
                "angles",
                "det_calib_params",
            )
        }
        d["geometry"] = self.geometry.to_dict()
        if self.q_lab_interpolator is not None:
            d["transformation_matrices"] = {
                "q_lab": self.q_lab_interpolator.original_to_target_matrix,
            }
        if self.direct_lab_interpolator is not None:
            d["transformation_matrices"]["direct_lab"] = (
                self.direct_lab_interpolator.original_to_target_matrix
            )
            d["direct_lab_voxel_size"] = self.direct_lab_voxel_size

        return copy.deepcopy(d)

    def to_file(self, dump_path: str) -> None:
        """
        Save converter configuration to HDF5 file.

        Preserves all transformation matrices and calibration
        parameters, allowing reconstruction of converter state without
        recomputing q-space transitions (expensive operation).

        Args:
            dump_path: output HDF5 file path (.h5 extension added if
                missing).
        """
        dump_path += ".h5" if not dump_path.endswith(".h5") else ""

        attributes = self.to_dict()

        with h5py.File(dump_path, "w") as file:
            file.attrs["creator"] = "cdiutils"
            file.attrs["HDF5_Version"] = h5py.version.hdf5_version
            file.attrs["h5py_version"] = h5py.version.version
            file.create_dataset("program_name", data=f"cdiutils {__version__}")
            # Store basic attributes as datasets
            for key in ("energy", "roi", "q_lab_shift", "shape"):
                if attributes.get(key) is None:
                    attributes[key] = np.nan
                file.create_dataset(key, data=attributes[key])

            # Handle dictionaries
            for key in ("det_calib_params", "angles", "geometry"):
                group = file.create_group(key)
                for k, v in attributes[key].items():
                    group.create_dataset(k, data=v)

            # Save additional interpolator parameters if they exist
            if attributes.get("transformation_matrices") is not None:
                file.create_dataset(
                    "direct_lab_voxel_size",
                    data=attributes["direct_lab_voxel_size"],
                )
                matrice_group = file.create_group("transformation_matrices")
                for key in ("q_lab", "direct_lab"):
                    matrice_group.create_dataset(
                        key, data=attributes["transformation_matrices"][key]
                    )

    @classmethod
    def from_file(cls, path: str) -> "SpaceConverter":
        """
        Load converter from HDF5 file.

        Factory method that reconstructs SpaceConverter instance with
        all transformation matrices and calibration parameters,
        avoiding expensive recomputation of q-space transitions.

        Args:
            path: input HDF5 file path.

        Returns:
            SpaceConverter instance with restored state.
        """

        def decode_if_bytes(value):
            if isinstance(value, (np.ndarray, list, tuple)):
                return [
                    decode_if_bytes(e)
                    if isinstance(e, (bytes, np.ndarray, list, tuple))
                    else e
                    for e in value
                ]
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return value

        with h5py.File(path, "r") as file:
            # Load basic attributes
            attributes = {
                k: file[k][()]
                for k in ("energy", "roi", "q_lab_shift", "shape")
            }
            # Handle dictionaries
            attributes.update(
                {
                    d: {k: decode_if_bytes(v[()]) for k, v in file[d].items()}
                    for d in ("det_calib_params", "geometry")
                }
            )
            # load the angles separately
            angles = {k: v[()] for k, v in file["angles"].items()}

            if "q_lab" in file["transformation_matrices"]:
                attributes["q_lab_matrix"] = file[
                    "transformation_matrices/q_lab"
                ][()]

            if "direct_lab" in file["transformation_matrices"]:
                attributes["direct_lab_matrix"] = file[
                    "transformation_matrices/direct_lab"
                ][()]
                attributes["direct_lab_voxel_size"] = file[
                    "direct_lab_voxel_size"
                ][()]

        # build the Geometry instance
        attributes["geometry"] = Geometry.from_dict(attributes["geometry"])

        # Make sure shape is a tuple
        attributes["shape"] = tuple(attributes["shape"])

        # Create a new SpaceConverter instance using loaded data
        instance = cls(**attributes)
        instance.init_q_space(**angles)

        return instance

    def init_q_space(
        self,
        sample_outofplane_angle: float | np.ndarray,
        sample_inplane_angle: float | np.ndarray,
        detector_outofplane_angle: float | np.ndarray,
        detector_inplane_angle: float | np.ndarray,
        det_calib_params: dict = None,
        roi: list = None,
    ) -> None:
        """
        Initialise q-space mapping using xrayutilities.

        Computes q-space coordinates for each detector pixel given
        goniometer angles. This is the foundation for all BCDI
        coordinate transformations. Requires accurate detector
        calibration (cch1, cch2, distance, pixel sizes) to preserve
        physical strain calculations.

        Args:
            sample_outofplane_angle: out-of-plane rotation (e.g. om,
                theta, eta) in degrees. Can be 1D array (rocking
                curve scan).
            sample_inplane_angle: in-plane rotation (e.g. phi, chi,
                gamma) in degrees.
            detector_outofplane_angle: detector out-of-plane angle
                (e.g. delta, tth) in degrees.
            detector_inplane_angle: detector in-plane angle (e.g. nu,
                gamma) in degrees.
            det_calib_params: detector calibration dict with keys
                'cch1', 'cch2', 'distance', 'pwidth1', 'pwidth2'.
            roi: region of interest [row_start, row_end, col_start,
                col_end].
        """
        # Check that det_calib and roi are provided or set in the
        # instance attributes.
        if det_calib_params is None:
            if self.det_calib_params is None:
                raise ValueError(
                    "det_calib_params not provided and no attribute set "
                    f"({self.det_calib_params = })."  # noqa: E251
                )
            det_calib_params = self.det_calib_params
        if "outerangle_offset" in det_calib_params:
            warnings.warn(
                "outerangle_offset is no longer required, will remove it from "
                "det_calib_params.",
                DeprecationWarning,
            )
            det_calib_params.pop("outerangle_offset")
        det_calib_params = det_calib_params.copy()
        if roi is None:
            if self.roi is None:
                raise ValueError(
                    f"roi not provided and no attribute set ({self.roi = })."  # noqa: E251
                )
            roi = self.roi

        #  Check that necessary det_calib_params keys are provided
        if not np.all(
            k in ["cch1", "cch2", "distance", "pwidth1", "pwidth2"]
            for k in det_calib_params
        ):
            raise ValueError(
                "Key missing in det_calib_params, it must contain at least: "
                "'cch1', 'cch2', 'distance', 'pwidth1' and 'pwidth2'."
            )
        qconversion = QConversion(
            sampleAxis=self.geometry.sample_circles,
            detectorAxis=self.geometry.detector_circles,
            r_i=self.geometry.beam_direction,
        )
        self.hxrd = HXRD(
            idir=self.geometry.beam_direction,  # defines the inplane
            # reference direction (idir points into the beam
            # direction at zero angles)
            ndir=[0, 0, 1],  # defines the surface normal of your sample
            # (ndir points along the innermost sample rotation axis)
            en=self.energy,
            qconv=qconversion,
        )

        self.hxrd.Ang2Q.init_area(
            detectorDir1=self.geometry.detector_axis0_orientation,
            detectorDir2=self.geometry.detector_axis1_orientation,
            cch1=det_calib_params.pop("cch1") - roi[0],
            cch2=det_calib_params.pop("cch2") - roi[2],
            Nch1=roi[1] - roi[0],
            Nch2=roi[3] - roi[2],
            **det_calib_params,
        )

        self._q_space_transitions = self.hxrd.Ang2Q.area(
            sample_outofplane_angle,
            sample_inplane_angle,
            detector_inplane_angle,
            detector_outofplane_angle,
        )
        self._q_space_transitions = np.asarray(self._q_space_transitions)

        self._shape = self._q_space_transitions.shape[1:]

        self.angles = {
            "sample_outofplane_angle": sample_outofplane_angle,
            "sample_inplane_angle": sample_inplane_angle,
            "detector_outofplane_angle": detector_outofplane_angle,
            "detector_inplane_angle": detector_inplane_angle,
        }

    def index_det_to_q_lab(self, ijk: tuple) -> tuple:
        """
        Convert detector index to q-space coordinates.

        Args:
            ijk: detector pixel index (i, j, k).

        Returns:
            (qx, qy, qz) coordinates in Å⁻¹.
        """
        if self._q_space_transitions is None:
            raise ValueError(
                "q_space_transitions is None, please set the q space area "
                "with SpaceConverter.set_q_space_area() method."
            )
        return self.do_transition(ijk, self._q_space_transitions)

    def index_cropped_det_to_q_lab(self, ijk: tuple) -> tuple:
        """Convert cropped detector index to q-space coordinates."""
        return self.index_det_to_q_lab(self.index_cropped_det_to_det(ijk))

    def index_det_to_index_of_q_lab(
        self, ijk: tuple, interpolation_method: str = "cdiutils"
    ) -> tuple:
        """
        Transition an index from the full detector frame to the
        index-of-q lab frame
        """
        if interpolation_method in ("xu", "xrayutilities"):
            cubinates = self.get_xu_q_lab_regular_grid(arrangement="cubinates")
        else:
            cubinates = self.get_q_lab_regular_grid(arrangement="cubinates")
        q_pos = self.index_det_to_q_lab(ijk)  # q value
        new_ijk = np.unravel_index(
            np.argmin(np.linalg.norm(cubinates - q_pos, axis=3)),
            cubinates.shape[:-1],
        )
        return new_ijk

    def index_cropped_det_to_index_of_q_lab(self, ijk: tuple) -> tuple:
        """
        Transition an index from the cropped detector frame to the
        index-of-q lab frame
        """
        cubinates = self.get_q_lab_regular_grid(arrangement="cubinates")
        q_pos = self.index_cropped_det_to_q_lab(ijk)  # q value
        new_ijk = np.unravel_index(
            np.argmin(np.linalg.norm(cubinates - q_pos, axis=3)),
            cubinates.shape[:-1],
        )
        return new_ijk

    @staticmethod
    def dspacing(q_lab_coordinates: np.ndarray | list | tuple) -> float:
        """
        Compute d-spacing from q-space coordinates.

        d-spacing is the lattice plane separation: d = 2π / |q|.

        Args:
            q_lab_coordinates: (qx, qy, qz) in Å⁻¹.

        Returns:
            d-spacing in Ångströms.
        """
        return float(2 * np.pi / np.linalg.norm(q_lab_coordinates))

    @classmethod
    def lattice_parameter(
        cls,
        q_lab_coordinates: np.ndarray | list | tuple,
        hkl: np.ndarray | list | tuple,
    ) -> float:
        """
        Compute lattice parameter from Bragg peak position.

        Uses formula: a = d * sqrt(h² + k² + l²) for cubic crystals.

        Args:
            q_lab_coordinates: (qx, qy, qz) at Bragg peak in Å⁻¹.
            hkl: Miller indices of the reflection.

        Returns:
            Lattice parameter in Ångströms.
        """
        return float(
            cls.dspacing(q_lab_coordinates)
            * np.sqrt(hkl[0] ** 2 + hkl[1] ** 2 + hkl[2] ** 2)
        )

    @staticmethod
    def do_transition(
        ijk: np.ndarray | list | tuple, transition_matrix: np.ndarray
    ) -> tuple:
        """
        Transform a (i, j, k) tuple into the corresponding (qx, qy, qz)
        values based the transition matrix
        """
        ijk = tuple(ijk)
        return tuple(float(transition_matrix[i][ijk]) for i in range(3))

    def _centre_shift_q_space_transitions(
        self, shift_voxel: tuple
    ) -> np.ndarray:
        # using the Interpolator3D requires the centring of the q
        # values, here we save the shift in q for later use
        q_lab_shift = np.array(
            [self._q_space_transitions[i][shift_voxel] for i in range(3)]
        )
        # center the q_space_transitions values (not the indexes) so the
        # center of the Bragg peak is (0, 0, 0) A-1
        for i in range(3):
            self._q_space_transitions[i] = (
                self._q_space_transitions[i] - q_lab_shift[i]
            )

        # reshape the grid so rows correspond to x, y and z coordinates,
        # columns correspond to the bins
        q_space_transitions = self._q_space_transitions.reshape(
            3, self._q_space_transitions[0].size
        )
        self.q_lab_shift = q_lab_shift
        return q_space_transitions

    def init_interpolator(
        self,
        direct_lab_voxel_size: tuple | float = None,
        space: str = "direct",
        shift_voxel: tuple = None,
        verbose: bool = False,
    ):
        """
        Initialise interpolators for orthogonalisation.

        Prepares linear transformation matrices for converting
        non-orthogonal detector data to orthogonal grids. For BCDI
        strain analysis, accurate voxel size calibration is essential
        to preserve displacement field magnitudes.

        Args:
            direct_lab_voxel_size: target voxel size in direct space
                (nm). If None, auto-computed from reciprocal extent.
                Can be scalar (isotropic) or 3-tuple (anisotropic).
            space: which interpolator(s) to initialise: 'reciprocal'
                (q-space), 'direct' (real-space), or 'both'.
            shift_voxel: Bragg peak centre in detector frame (i,j,k).
                Used to centre reciprocal space at (0,0,0). Default:
                centre of array.
            verbose: print auto-computed voxel size.
        """
        if space not in (
            "q",
            "rcp",
            "rcp_space",
            "reciprocal",
            "reciprocal_space",
            "both",
            "direct",
            "direct_space",
        ):
            raise ValueError("Invalid space.")

        if shift_voxel is None:
            shift_voxel = tuple(s // 2 for s in self._shape)

        # get the transformation_matrix
        transformation_matrix = self.get_transformation_matrix(shift_voxel)

        if space in (
            "q",
            "rcp",
            "rcp_space",
            "reciprocal",
            "reciprocal_space",
            "both",
        ):
            self.q_lab_interpolator = Interpolator3D(
                original_shape=self._shape,
                original_to_target_matrix=transformation_matrix,
            )

        if space in ("direct", "direct_space", "both"):
            # Compute the linear transformation matrix of the direct space
            direct_lab_transformation_matrix = np.dot(
                np.linalg.inv(transformation_matrix.T),
                np.diag(2 * np.pi / np.array(self._shape)),
            )
            # Unit of the matrix vectors is A, convert it to nm
            direct_lab_transformation_matrix /= 10

            original_direct_lab_voxel_size = np.linalg.norm(
                direct_lab_transformation_matrix, axis=1
            )
            if verbose:
                print(
                    "Voxel size calculated from the extent in the reciprocal "
                    "space is:\n"
                    f"{original_direct_lab_voxel_size} (nm)."
                )

            if direct_lab_voxel_size is None:
                direct_lab_voxel_size = original_direct_lab_voxel_size
            if isinstance(direct_lab_voxel_size, numbers.Number):
                direct_lab_voxel_size = np.repeat(
                    direct_lab_voxel_size, len(self._shape)
                )

            # initialise the interpolator instance
            self.direct_lab_interpolator = Interpolator3D(
                original_shape=self._shape,
                original_to_target_matrix=direct_lab_transformation_matrix,
                target_voxel_size=direct_lab_voxel_size,
            )
            self._direct_lab_voxel_size = direct_lab_voxel_size

    def orthogonalise_to_q_lab(
        self,
        data: np.ndarray,
        method: str = "cdiutils",
        shift_voxel: tuple = None,
    ) -> np.ndarray:
        """
        Orthogonalise detector data to reciprocal lab frame (q-space).

        BCDI-critical transformation that converts non-orthogonal
        detector pixels to regular reciprocal grid. Choice of method
        affects interpolation quality and speed.

        Args:
            data: 3D detector data (nz, ny, nx) with rocking curve
                along first axis.
            method: 'cdiutils' (fast linear interpolation) or 'xu'
                (xrayutilities.FuzzyGridder3D, slower but handles
                irregular grids better).
            shift_voxel: Bragg peak centre (i,j,k) for centring at
                origin. Default: centre of array.

        Returns:
            Orthogonalised 3D array in q lab frame.
        """
        if self.shape != data.shape:
            raise ValueError(
                f"The shape of the data to orthogonalise {data.shape} must "
                "match that of the data used to initialise the q space "
                f"{self.shape})."
            )

        if method in ("xu", "xrayutilities"):
            gridder = FuzzyGridder3D(*data.shape)
            gridder(*self._q_space_transitions, data)
            self.xu_gridder = gridder
            return gridder.data

        if self.q_lab_interpolator is None:
            self.init_interpolator(
                space="reciprocal_space", shift_voxel=shift_voxel
            )
        return self.q_lab_interpolator(data)

    def orthogonalise_to_direct_lab(
        self,
        direct_space_data: np.ndarray,
        direct_lab_voxel_size: tuple | np.ndarray | list | float = None,
    ) -> np.ndarray:
        """
        Orthogonalise reconstructed object to direct lab frame.

        Transformation that converts BCDI reconstruction
        (in non-orthogonal detector frame) to orthogonal real-space.
        Essential for accurate strain calculations, where voxel size
        determines displacement field calibration.

        Args:
            direct_space_data: 3D complex-valued reconstruction
                (amplitude * exp(i*phase)).
            direct_lab_voxel_size: target voxel size (nm). If
                provided, reinitialises interpolator. Can be scalar or
                3-tuple.

        Returns:
            Orthogonalised 3D complex array in direct lab frame.
        """

        if self.direct_lab_interpolator is None:
            self.init_interpolator(
                direct_lab_voxel_size=direct_lab_voxel_size,
                space="direct",
            )
        if direct_lab_voxel_size is not None:
            # This will reinitialise the whole interpolator!
            self.direct_lab_voxel_size = direct_lab_voxel_size

        return self.direct_lab_interpolator(direct_space_data)

    def get_transformation_matrix(
        self,
        shift_voxel: tuple | None = None,
    ) -> np.ndarray:
        """
        Compute transformation matrix for detector to q-space mapping.

        Returns linear transformation converting (i,j,k) detector
        indices to (qx,qy,qz) coordinates.

        Args:
            shift_voxel: Bragg peak centre (i,j,k). Default: array
                centre.

        Returns:
            3x3 transformation matrix.
        """
        if shift_voxel is None:
            shift_voxel = tuple(s // 2 for s in self._shape)

        q_space_transitions = self._centre_shift_q_space_transitions(
            shift_voxel
        )

        # create the 0 centered index grid
        ortho_grid = []
        for i in np.indices(self._shape):
            ortho_grid.append(i - i[shift_voxel])
        ortho_grid = np.array(ortho_grid).reshape(3, np.prod(self._shape))

        return np.dot(
            q_space_transitions,
            np.dot(
                ortho_grid.T,
                np.linalg.inv(np.dot(ortho_grid, ortho_grid.T)),
            ),
        )

    def get_q_space_transitions(self, arrangement: str = "list"):
        """
        Get q-space coordinates from xrayutilities mapping.

        Args:
            arrangement: 'list' (three 1D arrays) or 'cubinates' (4D
                array with shape (nz,ny,nx,3)).

        Returns:
            Q-space coordinate grid.
        """
        if self._q_space_transitions is None:
            raise ValueError(
                "_q_space_transitions are none, use the set_q_space_area "
                "method"
            )
        if arrangement in ("l", "list"):
            return self._q_space_transitions
        elif arrangement in ("c", "cubinates"):
            return np.moveaxis(
                self._q_space_transitions, source=0, destination=3
            )
        else:
            raise ValueError(
                "arrangement should be 'l', 'list', 'c' or 'cubinates'"
            )

    def get_xu_q_lab_regular_grid(self, arrangement: str = "list"):
        """
        Get regular q lab grid from xrayutilities.FuzzyGridder3D.

        Only available after calling orthogonalise_to_q_lab with
        method='xu'.

        Args:
            arrangement: 'list' (three 1D axes) or 'cubinates' (4D
                meshgrid).

        Returns:
            Regular q-space grid.
        """
        if self.xu_gridder is None:
            raise ValueError(
                "No q lab space xu_gridder initialized, cannot provide a "
                "regular grid"
            )
        grid = [
            self.xu_gridder.xaxis,
            self.xu_gridder.yaxis,
            self.xu_gridder.zaxis,
        ]
        if arrangement in ("l", "list"):
            return grid
        if arrangement in ("c", "cubinates"):
            # create a temporary meshgrid
            qx, qy, qz = np.meshgrid(grid[0], grid[1], grid[2], indexing="ij")
            return np.moveaxis(np.array([qx, qy, qz]), source=0, destination=3)
        raise ValueError(
            "arrangement should be 'l', 'list', 'c' or 'cubinates'"
        )

    def get_q_lab_regular_grid(self, arrangement: str = "list"):
        """
        Get regular q lab grid from cdiutils interpolator.

        Only available after init_interpolator with space='reciprocal'
        or 'both'.

        Args:
            arrangement: 'list' (three 1D axes) or 'cubinates' (3D
                meshgrid with shape (nz,ny,nx,3)).

        Returns:
            Regular q-space grid in Angstrom-1.
        """
        if self.q_lab_interpolator is None:
            raise ValueError(
                "No q lab space interpolator initialized, cannot provide a "
                "regular grid."
            )

        grid = [
            self.q_lab_interpolator.target_grid[i] + self.q_lab_shift[i]
            for i in range(3)
        ]
        if arrangement in ("l", "list"):
            return [grid[0][:, 0, 0], grid[1][0, :, 0], grid[2][0, 0, :]]
        if arrangement in ("c", "cubinates"):
            return np.moveaxis(grid, source=0, destination=3)
        raise ValueError(
            "arrangement should be 'l', 'list', 'c' or 'cubinates'."
        )

    def get_direct_lab_regular_grid(self, arrangement: str = "list"):
        """
        Get regular direct lab grid from cdiutils interpolator.

        Only available after init_interpolator with space='direct' or
        'both'.

        Args:
            arrangement: 'list' (three 1D axes) or 'cubinates' (3D
                meshgrid).

        Returns:
            Regular real-space grid in nm.
        """
        if self.direct_lab_interpolator is None:
            raise ValueError(
                "No direct lab space interpolator initialized, cannot provide "
                "a regular grid"
            )
        grid = [self.direct_lab_interpolator.target_grid[i] for i in range(3)]
        if arrangement in ("l", "list"):
            return [grid[0][:, 0, 0], grid[1][0, :, 0], grid[2][0, 0, :]]
        if arrangement in ("c", "cubinates"):
            return np.moveaxis(grid, source=0, destination=3)
        raise ValueError(
            "arrangement should be 'l', 'list', 'c' or 'cubinates'"
        )

    def get_q_norm_histogram(
        self, q_lab_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute radial profile data for q-space histogram.

        Calculates |q| magnitude for each voxel and returns sorted
        (|q|, intensity) pairs for 1D radial averaging.

        Args:
            q_lab_data: 3D intensity array in orthogonalised q lab
                frame.

        Returns:
            Tuple of (sorted_q_norms, sorted_intensities) as 1D
            arrays.
        """
        try:
            grid = self.get_q_lab_regular_grid(arrangement="c")
        except ValueError:
            grid = self.get_xu_q_lab_regular_grid(arrangement="c")

        grid = np.reshape(grid, (grid.size // 3, 3))
        q_norm = np.linalg.norm(grid, axis=1)
        flattend_intensity = np.reshape(q_lab_data, q_lab_data.size)
        sort_order = q_norm.argsort()
        sorted_q_norm = q_norm[sort_order]
        sorted_flat_intens = flattend_intensity[sort_order]

        return sorted_q_norm, sorted_flat_intens

    def support_transfer(
        self,
        support: np.ndarray,
        voxel_size: tuple,
        convert_to_xu: bool = False,
    ) -> np.ndarray:
        """
        Transfer support from direct lab to reconstruction frame.

        Interpolates support defined in orthogonal direct lab frame
        (e.g. from previous analysis) onto non-orthogonal
        reconstruction frame. Used to initialise BCDI phase retrieval
        with prior knowledge.

        Args:
            support: binary mask in direct lab frame.
            voxel_size: voxel size of input support in nm.
            convert_to_xu: convert support from CXI to XU convention
                before interpolation.

        Returns:
            Support array in detector frame.
        """
        if convert_to_xu:
            support = Geometry.swap_convention(support)
            voxel_size = Geometry.swap_convention(voxel_size)
        shape = support.shape
        rgi = RegularGridInterpolator(
            (
                np.arange(-shape[0] // 2, shape[0] // 2, 1) * voxel_size[0],
                np.arange(-shape[1] // 2, shape[1] // 2, 1) * voxel_size[1],
                np.arange(-shape[2] // 2, shape[2] // 2, 1) * voxel_size[2],
            ),
            support,
            method="linear",
            bounds_error=False,
            fill_value=0,
        )

        shift_voxel = tuple(s // 2 for s in self._shape)
        # create the 0 centered index grid
        ortho_grid = []
        for i in np.indices(self._shape):
            ortho_grid.append(i - i[shift_voxel])
        ortho_grid = np.array(ortho_grid).reshape(3, np.prod(self._shape))

        # get the transformation_matrix
        transformation_matrix = self.get_transformation_matrix(shift_voxel)
        direct_lab_transformation_matrix = np.dot(
            np.linalg.inv(transformation_matrix.T),
            np.diag(2 * np.pi / np.array(self._shape)),
        )
        # Unit of the matrix vectors is A, convert it to nm
        direct_lab_transformation_matrix /= 10

        reconstruction_frame_grid = np.dot(
            direct_lab_transformation_matrix, ortho_grid
        )
        reconstruction_frame_grid = reconstruction_frame_grid.reshape(
            (3,) + self._shape
        )
        reconstruction_frame_support = rgi(
            tuple(reconstruction_frame_grid),  # must be a tuple here
            method="linear",
        )
        return reconstruction_frame_support

    @staticmethod
    def run_detector_calibration(
        detector_calibration_frames: np.ndarray,
        detector_outofplane_angle: float,
        detector_inplane_angle: float,
        energy: float,
        xu_detector_circles: list,  # Convention should be xu not cxi
        pixel_size_x: float = 55e-6,
        pixel_size_y: float = 55e-6,
        sdd_estimate: float = None,
        show: bool = True,
        verbose: bool = True,
    ) -> dict:
        """
        Calibrate detector parameters using xrayutilities (BCDI-critical).

        Determines detector calibration (cch1, cch2, distance, tilt,
        etc.) from direct beam scans. Critical for accurate q-space
        mapping in BCDI strain calculations.

        Args:
            detector_calibration_frames: 3D array of direct beam images
                (nangles, ny, nx).
            detector_outofplane_angle: out-of-plane detector angles
                during scan (degrees).
            detector_inplane_angle: in-plane detector angles during
                scan (degrees).
            energy: X-ray energy in eV.
            xu_detector_circles: detector circle names in XU convention
                (e.g. ['z-', 'y-'] for ID01).
            pixel_size_x: detector pixel size along x in metres
                (default: 55 µm for Eiger).
            pixel_size_y: detector pixel size along y in metres
                (default: 55 µm).
            sdd_estimate: initial sample-detector distance in metres.
                If None, auto-estimated from beam motion.
            show: plot calibration diagnostic plots.
            verbose: print calibration parameters.

        Returns:
            Dictionary with keys: cch1, cch2, pwidth1, pwidth2,
            distance, tiltazimuth, tilt, detrot.

        Example:
            >>> # ID01 detector calibration
            >>> params = converter.run_detector_calibration(
            ...     calibration_scan_data,
            ...     delta_angles,
            ...     nu_angles,
            ...     energy=9000,
            ...     xu_detector_circles=['z-', 'y-']
            ... )
        """
        coms = []
        for i in range(detector_calibration_frames.shape[0]):
            coms.append(center_of_mass(detector_calibration_frames[i]))
        coms = np.array(coms)

        if sdd_estimate is None:
            # get the sample to detector distance
            # for that determine how much the the com has moved when the
            # detector has rotated by 1 degree. We may find this value with
            # delta or nu. Here, we do both and calculate the average. The
            # leading coefficient of the function x_com = f(delta) gives
            # how much the x_com has moved when delta has changed by one
            # degree.
            slope = [
                np.polynomial.polynomial.polyfit(a, c, 1)[1]
                for a, c in zip(
                    (detector_outofplane_angle, detector_inplane_angle),
                    (coms[:, 0], coms[:, 1]),
                )
            ]
            sdd_estimate = (
                (1 / 2)
                * (1 / np.tan(np.pi / 180))
                * (slope[0] * pixel_size_y + slope[1] * pixel_size_x)
            )

        if verbose:
            print(f"[INFO] First estimate of sdd: {sdd_estimate} m.")
            print(
                "[INFO] Processing to detector calibration using "
                "xrayutilities.analysis.sample_align.area_detector_calib"
            )
        parameter_list, _ = area_detector_calib(
            detector_inplane_angle,
            detector_outofplane_angle,
            detector_calibration_frames,
            detaxis=xu_detector_circles,
            r_i="x+",
            start=(pixel_size_x, pixel_size_y, sdd_estimate, 0, 0, 0, 0),
            fix=(True, True, False, False, False, False, True),
            wl=energy_to_wavelength(energy) * 1e-10,  # wavelength in A
        )

        parameters = {
            "cch1": parameter_list[0],
            "cch2": parameter_list[1],
            "pwidth1": parameter_list[2],
            "pwidth2": parameter_list[3],
            "distance": parameter_list[4],
            "tiltazimuth": parameter_list[5],
            "tilt": parameter_list[6],
            "detrot": parameter_list[7],
        }

        if verbose:
            print("Computed parameters:\n")
            for key, value in parameters.items():
                print(f"{key} = {value}")
        if show:
            fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
            fig2, axes2 = plt.subplots(1, 2)
            ax1.imshow(np.log10(detector_calibration_frames.sum(axis=0)))
            axes2[0].plot(detector_outofplane_angle, coms[:, 0])
            axes2[0].set_xlabel("detector outofplane angle")
            axes2[0].set_ylabel("COM along axis0")

            axes2[1].plot(detector_inplane_angle, coms[:, 1])
            axes2[1].set_xlabel("detector inplane angle")
            axes2[1].set_ylabel("COM along axis1")
            fig1.tight_layout()
            fig2.tight_layout()

        return parameters

    def __repr__(self) -> str:
        data = self.__dict__.copy()
        data["geometry"] = self.geometry.to_dict()
        msg = ""
        for attr in ("det_calib_params", "energy", "roi", "angles"):
            msg += f"{attr} = {data[attr]}\n"
        msg += self.geometry.__repr__()
        return msg


class Interpolator3D:
    """
    3D linear interpolation for BCDI orthogonalisation.

    Handles coordinate transformation from non-orthogonal detector
    frame to orthogonal grids (q-space or direct space). Uses
    scipy.interpolate.RegularGridInterpolator with automatic extent
    calculation.

    Note:
        This is an internal class used by SpaceConverter. Users
        typically don't instantiate this directly.
    """

    def __init__(
        self,
        original_shape: tuple | np.ndarray | list,
        original_to_target_matrix: np.ndarray,
        target_voxel_size: tuple | np.ndarray | list | float = None,
        verbose: bool = False,
    ):
        """
        Initialise the 3D interpolator.

        Args:
            original_shape: shape of data in detector frame (nz, ny,
                nx).
            original_to_target_matrix: 3x3 linear transformation
                matrix mapping detector indices to target coordinates.
            target_voxel_size: desired voxel size in target space. If
                None, uses norm of transformation matrix columns. Can
                be scalar or 3-tuple.
            verbose: print extent information.
        """
        self.original_shape = original_shape

        if target_voxel_size is None:
            self.target_voxel_size = np.linalg.norm(
                original_to_target_matrix, axis=1
            )
        elif isinstance(target_voxel_size, numbers.Number):
            self.target_voxel_size = np.repeat(
                target_voxel_size, len(original_shape)
            )
        else:
            if len(target_voxel_size) != len(original_shape):
                raise ValueError(
                    f"Lengths of target_voxel_size ({len(target_voxel_size)}) "
                    f"and original_shape ({len(original_shape)}) should be "
                    "equal"
                )
            self.target_voxel_size = target_voxel_size

        self.original_to_target_matrix = original_to_target_matrix

        # invert the provided rotation matrix
        target_to_original_matrix = np.linalg.inv(original_to_target_matrix)

        self.extents = None

        # initialize the grid in the target space
        self.target_grid = None
        self._init_target_grid(verbose)

        # rotate the target space grid to the original space
        self.target_grid_in_original_space = self._rotate_grid_axis(
            target_to_original_matrix, *self.target_grid
        )

    def _init_target_grid(self, verbose: bool = False) -> None:
        """
        Initialize the target space grid by finding the extent of the
        original space grid in the target space.
        """

        grid_axis0, grid_axis1, grid_axis2 = self.zero_centred_meshgrid(
            self.original_shape
        )

        grid_axis0, grid_axis1, grid_axis2 = self._rotate_grid_axis(
            self.original_to_target_matrix, grid_axis0, grid_axis1, grid_axis2
        )

        self._find_extents(grid_axis0, grid_axis1, grid_axis2)
        if verbose:
            print(
                "The extent in the target space of a regular grid "
                f"defined in the original space with a shape of "
                f"{self.original_shape} is {self.extents}"
            )

        # define a regular grid in the target space with the computed extent
        self.target_grid = self.zero_centred_meshgrid(
            shape=self.extents, scale=self.target_voxel_size
        )

    @staticmethod
    def zero_centred_meshgrid(
        shape: np.ndarray | list | tuple,
        scale: np.ndarray | list | tuple = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the a zero-centred meshgrid with the 'ij' indexing."""

        if scale is None:
            scale = [1, 1, 1]

        return np.meshgrid(
            np.arange(-shape[0] // 2, shape[0] // 2, 1) * scale[0],
            np.arange(-shape[1] // 2, shape[1] // 2, 1) * scale[1],
            np.arange(-shape[2] // 2, shape[2] // 2, 1) * scale[2],
            indexing="ij",
        )

    def _rotate_grid_axis(
        self,
        transfer_matrix: np.ndarray,
        grid_axis0: np.ndarray,
        grid_axis1: np.ndarray,
        grid_axis2: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rotate the grid axes to the target space."""

        rotated_grid_axis0 = (
            transfer_matrix[0, 0] * grid_axis0
            + transfer_matrix[0, 1] * grid_axis1
            + transfer_matrix[0, 2] * grid_axis2
        )
        rotated_grid_axis1 = (
            transfer_matrix[1, 0] * grid_axis0
            + transfer_matrix[1, 1] * grid_axis1
            + transfer_matrix[1, 2] * grid_axis2
        )

        rotated_grid_axis2 = (
            transfer_matrix[2, 0] * grid_axis0
            + transfer_matrix[2, 1] * grid_axis1
            + transfer_matrix[2, 2] * grid_axis2
        )

        return rotated_grid_axis0, rotated_grid_axis1, rotated_grid_axis2

    def _find_extents(
        self,
        grid_axis0: np.ndarray,
        grid_axis1: np.ndarray,
        grid_axis2: np.ndarray,
    ) -> None:
        """Find the extents in 3D of a given tuple of grid."""
        extent_axis0 = int(
            np.ceil(
                (grid_axis0.max() - grid_axis0.min())
                / self.target_voxel_size[0]
            )
        )
        extent_axis1 = int(
            np.ceil(
                (grid_axis1.max() - grid_axis1.min())
                / self.target_voxel_size[1]
            )
        )
        extent_axis2 = int(
            np.ceil(
                (grid_axis2.max() - grid_axis2.min())
                / self.target_voxel_size[2]
            )
        )
        self.extents = extent_axis0, extent_axis1, extent_axis2

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Interpolate data from detector frame to target frame.

        Args:
            data: 3D array in detector frame.

        Returns:
            Interpolated data on regular target grid.
        """
        rgi = RegularGridInterpolator(
            (
                np.arange(-data.shape[0] // 2, data.shape[0] // 2, 1),
                np.arange(-data.shape[1] // 2, data.shape[1] // 2, 1),
                np.arange(-data.shape[2] // 2, data.shape[2] // 2, 1),
            ),
            data,
            method="linear",
            bounds_error=False,
            fill_value=0,
        )

        # find the interpolated value of the grid which was defined in the
        # target space and then rotated to the original space
        interpolated_data = rgi(
            np.concatenate(
                (
                    self.target_grid_in_original_space[0].reshape(
                        (1, self.target_grid_in_original_space[0].size)
                    ),
                    self.target_grid_in_original_space[1].reshape(
                        (1, self.target_grid_in_original_space[1].size)
                    ),
                    self.target_grid_in_original_space[2].reshape(
                        (1, self.target_grid_in_original_space[2].size)
                    ),
                )
            ).transpose()
        )

        # reshape the volume back to its original shape, thus each voxel
        # goes back to its initial position
        interpolated_data = interpolated_data.reshape(
            (self.extents[0], self.extents[1], self.extents[2])
        ).astype(interpolated_data.dtype)

        return interpolated_data
