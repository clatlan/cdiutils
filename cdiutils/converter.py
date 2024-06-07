from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import xrayutilities as xu

from cdiutils.utils import center, crop_at_center
from cdiutils.geometry import Geometry


class SpaceConverter():
    """
    A class to handle the conversions between the different frames and
    spaces.
    """
    def __init__(
            self,
            geometry: Geometry,
            energy: float = None
    ):
        self.geometry = geometry
        # convert the geometry to xrayutilities coordinate system
        if self.geometry.is_cxi:
            self.geometry.cxi_to_xu()

        self.energy = energy
        self.det_calib_parameters = {}
        self.hxrd = None

        self._q_space_transitions = None

        self._reference_voxel = None
        self._cropped_shape = None
        self._full_shape = None

        self.q_space_shift = None

        self.q_lab_interpolator: Interpolator3D = None
        self.direct_lab_interpolator: Interpolator3D = None
        self.xu_gridder: xu.FuzzyGridder3D = None

    @property
    def q_space_transitions(self):
        return self._q_space_transitions

    @q_space_transitions.setter
    def q_space_transitions(self, transitions: np.array):
        self._q_space_transitions = transitions

    @property
    def reference_voxel(self):
        return self._reference_voxel

    @reference_voxel.setter
    def reference_voxel(self, voxel: tuple):
        self._reference_voxel = voxel

    @property
    def cropped_shape(self):
        return self._cropped_shape

    @cropped_shape.setter
    def cropped_shape(self, shape: tuple):
        self._cropped_shape = shape

    @property
    def full_shape(self):
        return self._full_shape

    @full_shape.setter
    def full_shape(self, shape: tuple):
        self._full_shape = shape

    def init_q_space_area(
            self,
            roi: np.ndarray | list | tuple,
            det_calib_parameters: dict = None
    ):
        """
        Initialize the xrayutilites XHRD instance with the detector
        calibration parameters.
        """
        if det_calib_parameters is None:
            det_calib_parameters = self.det_calib_parameters
            if self.det_calib_parameters is None:
                raise ValueError(
                    "Provide det_calib_parameters or run"
                    "the detector calibration"
                )
        else:
            self.det_calib_parameters = det_calib_parameters

        if np.all([
                k in ["cch1", "cch2", "pwidth1", "pwidth2", "distance",
                      "tiltazimuth", "tilt", "detrot", "outerangle_offset"]
                for k in det_calib_parameters.keys()
        ]):
            qconversion = xu.experiment.QConversion(
                sampleAxis=self.geometry.sample_circles,
                detectorAxis=self.geometry.detector_circles,
                r_i=self.geometry.beam_direction
            )
            self.hxrd = xu.HXRD(
                idir=self.geometry.beam_direction,  # defines the inplane
                # reference direction (idir points into the beam
                # direction at zero angles)
                ndir=[0, 0, 1],  # defines the surface normal of your sample
                # (ndir points along the innermost sample rotation axis)
                en=self.energy,
                qconv=qconversion
            )

            self.hxrd.Ang2Q.init_area(
                detectorDir1=self.geometry.detector_vertical_orientation,
                detectorDir2=self.geometry.detector_horizontal_orientation,
                cch1=det_calib_parameters["cch1"] - roi[0],
                cch2=det_calib_parameters["cch2"] - roi[2],
                Nch1=roi[1] - roi[0],
                Nch2=roi[3] - roi[2],
                pwidth1=det_calib_parameters["pwidth1"],
                pwidth2=det_calib_parameters["pwidth2"],
                distance=det_calib_parameters["distance"],
                detrot=det_calib_parameters["detrot"],
                tiltazimuth=det_calib_parameters["tiltazimuth"],
                tilt=det_calib_parameters["tilt"],
            )
        else:
            raise ValueError(
                "det_calib_parameters dict requires the "
                "following keys\n"
                "'cch1', 'cch2', 'pwidth1', 'pwidth2', 'distance',"
                "'tiltazimuth', 'tilt', 'detrot', 'outerangle_offset'"
            )

    def set_q_space_area(
            self,
            sample_outofplane_angle: float | np.ndarray,
            sample_inplane_angle: float | np.ndarray,
            detector_outofplane_angle: float | np.ndarray,
            detector_inplane_angle: float | np.ndarray
    ):
        """
        Compute the _q_space_transitions provided by xrayutilities
        """

        qx, qy, qz = self.hxrd.Ang2Q.area(
                sample_outofplane_angle,
                sample_inplane_angle,
                detector_inplane_angle,
                detector_outofplane_angle
        )
        qx = np.array(qx)
        qy = np.array(qy)
        qz = np.array(qz)
        self._q_space_transitions = np.asarray([qx, qy, qz])
        self._full_shape = self._q_space_transitions.shape[1:]
        self._cropped_shape = self._full_shape

    def index_det_to_q_lab(self, ijk:  tuple) -> tuple:
        """
        Transition an index from the detector frame to the reciprocal 
        lab space
        """

        if self._q_space_transitions is None:
            raise ValueError(
                "q_space_transitions is None, please set the q space area "
                "with SpaceConverter.set_q_space_area() method."
            )
        return self.do_transition(
            ijk,
            self._q_space_transitions
        )

    def index_cropped_det_to_det(self, ijk:  tuple) -> tuple:
        """
        Transition an index of the cropped detector frame to the full
        detector frame 
        """
        if self._reference_voxel is None:
            raise ValueError("Set a reference_voxel.")
        return tuple(
            ijk
            + (np.array(self._full_shape) - np.array(self._cropped_shape))//2
            - (np.array(self._full_shape)//2 - self._reference_voxel)
        )

    def index_cropped_det_to_q_lab(self, ijk:  tuple) -> tuple:
        """
        Transition an index from the cropped detector frame to the
        reciprocal lab space frame
        """
        return self.index_det_to_q_lab(self.index_cropped_det_to_det(ijk))

    def index_det_to_index_of_q_lab(
            self,
            ijk: tuple,
            interpolation_method: str = "cdiutils"
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
            np.argmin(
                np.linalg.norm(
                    cubinates - q_pos,
                    axis=3
                )
            ),
            cubinates.shape[:-1]
        )
        return new_ijk

    def index_cropped_det_to_index_of_q_lab(self, ijk:  tuple) -> tuple:
        """
        Transition an index from the cropped detector frame to the
        index-of-q lab frame
        """
        cubinates = self.get_q_lab_regular_grid(arrangement="cubinates")
        q_pos = self.index_cropped_det_to_q_lab(ijk)  # q value
        new_ijk = np.unravel_index(
            np.argmin(
                np.linalg.norm(
                    cubinates - q_pos,
                    axis=3
                )
            ),
            cubinates.shape[:-1]
        )
        return new_ijk

    @staticmethod
    def dspacing(q_lab_coordinates: np.ndarray | list | tuple) -> float:
        """
        Compute the dspacing
        """
        return float(2*np.pi / np.linalg.norm(q_lab_coordinates))

    @classmethod
    def lattice_parameter(
                cls,
                q_lab_coordinates: np.ndarray | list | tuple,
                hkl: np.ndarray | list | tuple
    ) -> float:
        """
        Compute the lattice parameter
        """
        return float(
                cls.dspacing(q_lab_coordinates)
                * np.sqrt(hkl[0]**2 + hkl[1]**2 + hkl[2]**2)
        )

    @staticmethod
    def do_transition(
            ijk: np.ndarray | list | tuple,
            transition_matrix: np.ndarray
    ) -> tuple:
        """
        Transform a (i, j, k) tuple into the corresponding (qx, qy, qz)
        values based the transition matrix
        """
        ijk = tuple(ijk)
        return tuple(float(transition_matrix[i][ijk]) for i in range(3))

    def crop_q_space_transitions(self) -> np.ndarray:
        """
        Crop the _q_space_transitions according to the cropped data
        shape
        """
        if self._cropped_shape == self._full_shape:
            return self._q_space_transitions.copy()

        q_space_transitions = np.empty((3,) + self._cropped_shape)
        for i in range(3):
            q_space_transitions[i] = crop_at_center(
                center(
                    self._q_space_transitions[i],
                    where=self._reference_voxel
                ),
                final_shape=self._cropped_shape
            )
        return q_space_transitions

    def _center_shift_q_space_transitions(
            self,
            q_space_transitions: np.ndarray,
            shift_voxel: tuple
    ) -> np.ndarray:

        # Using the Interpolator3D requires the centering of the q
        # values, here we save the shift in q for later use
        q_space_shift = [
            q_space_transitions[i][shift_voxel]
            for i in range(3)
        ]
        # center the q_space_transitions values (not the indexes) so the
        # center of the Bragg peak is (0, 0, 0) A-1
        for i in range(3):
            q_space_transitions[i] = (
                q_space_transitions[i] - q_space_shift[i]
            )

        # reshape the grid so rows correspond to x, y and z coordinates,
        # columns correspond to the bins
        q_space_transitions = q_space_transitions.reshape(
                3,
                q_space_transitions[0].size
        )
        self.q_space_shift = q_space_shift
        return q_space_transitions

    def orthogonalize_to_q_lab(
            self,
            data: np.ndarray,
            method: str = "cdiutils",
            shift_voxel: tuple = None
    ) -> np.ndarray:
        """
        Orthogonalize detector data of the reciprocal space to the lab
        (xu) frame.
        """

        self._check_shape(data.shape)
        if self._q_space_transitions[0].shape != data.shape:
            q_space_transitions = self.crop_q_space_transitions()
        else:
            q_space_transitions = self._q_space_transitions

        if method in ("xu", "xrayutilities"):
            gridder = xu.FuzzyGridder3D(*data.shape)
            gridder(*q_space_transitions, data)
            self.xu_gridder = gridder
            return gridder.data

        if self.q_lab_interpolator is None:
            self.init_interpolator(
                data,
                space="reciprocal_space",
                shift_voxel=shift_voxel
            )
        return self.q_lab_interpolator(data)

    def _check_shape(self, shape: tuple) -> None:
        """
        Raise an error if the shape is different to the original raw
        data and the _reference voxel was not set.
        """
        if (
                self._q_space_transitions[0].shape != shape
                and self._reference_voxel is None
        ):
            raise ValueError(
                "The shape of the data to orthogonalize should be similar "
                "to that of the raw data "
                f"({self._q_space_transitions[0].shape} vs. {shape})."
                "\n Or you must set the reference_voxel which is the center "
                "of the parsed cropped data in the full raw data frame."
            )

    def init_interpolator(
            self,
            detector_data: np.ndarray,
            direct_space_data_shape: tuple | np.ndarray | list = None,
            direct_space_voxel_size: tuple | np.ndarray | list | float = None,
            space: str = "direct",
            shift_voxel: tuple = None
    ):

        if space not in (
                "q", "rcp", "rcp_space",
                "reciprocal", "reciprocal_space",
                "both", "direct", "direct_space"):
            raise ValueError(
                "Invalid space."
            )

        shape = detector_data.shape
        size = detector_data.size

        self._check_shape(shape)

        # self.crop_q_space_transitions()
        q_space_transitions = self.crop_q_space_transitions()

        if shift_voxel is None:
            shift_voxel = tuple(s // 2 for s in shape)

        q_space_transitions = self._center_shift_q_space_transitions(
            q_space_transitions,
            shift_voxel
        )

        # create the 0 centered index grid
        k_matrix = []
        for i in np.indices(shape):
            k_matrix.append(i - i[shift_voxel])
        k_matrix = np.array(k_matrix).reshape(3, size)

        # get the linear_transformation_matrix
        linear_transformation_matrix = self.linear_transformation_matrix(
            q_space_transitions,
            k_matrix
        )

        if space in ("direct", "direct_space", "both"):
            if direct_space_data_shape is None:
                raise ValueError(
                    "if space is 'direct' direct_space_data_shape must be "
                    "provided."
                )
            direct_space_data_shape = tuple(direct_space_data_shape)
            if shape != direct_space_data_shape:
                raise ValueError(
                    "The cropped_raw_data should have the same shape as the "
                    "reconstructed object."
                )

        if space in (
                "q", "rcp", "rcp_space",
                "reciprocal", "reciprocal_space", "both"):

            self.q_lab_interpolator = Interpolator3D(
                original_shape=shape,
                original_to_target_matrix=linear_transformation_matrix
            )

        if space in ("direct", "direct_space", "both"):

            # Compute the linear transformation matrix of the direct space
            direct_lab_transformation_matrix = np.dot(
                np.linalg.inv(linear_transformation_matrix.T),
                np.diag(2 * np.pi / np.array(shape))
            )
            # Unit of the matrix vectors is A, convert it to nm
            direct_lab_transformation_matrix /= 10

            direct_lab_voxel_size = np.linalg.norm(
                direct_lab_transformation_matrix,
                axis=1
            )
            print(
                "[INFO] Voxel size in the direct lab space due to the "
                f"orthognolization process is:\n{direct_lab_voxel_size} (nm)."
            )

            # initialise the interpolator instance
            self.direct_lab_interpolator = Interpolator3D(
                original_shape=shape,
                original_to_target_matrix=direct_lab_transformation_matrix,
                target_voxel_size=direct_space_voxel_size
            )

    def save_interpolation_parameters(self, output_path: str):
        """
        Save the interpolation parameters

        Args:
            output_path (str): where to save the parameters.
        """
        if self.q_lab_interpolator is None:
            print("[INFO] No interpolation parameters to save.")
        else:
            np.savez(
                output_path,
                q_space_linear_transformation_matrix=(
                    self.q_lab_interpolator.original_to_target_matrix
                ),
                direct_lab_linear_transformation_matrix=(
                    self.direct_lab_interpolator.original_to_target_matrix
                ),
                q_space_transitions=self._q_space_transitions,
                q_space_shift=self.q_space_shift,
                direct_space_voxel_size=(
                    self.direct_lab_interpolator.target_voxel_size
                )
            )

    def load_interpolation_parameters(
            self,
            file_path: str,
            direct_space_voxel_size: tuple | int = None,
            light_loading: bool = False,
    ) -> tuple:
        """
        Load interpolation parameters from a file and initialize
        interpolators.

        The interpolation parameters include transformation matrices,
        voxel sizes, and shape information used for interpolation in the
        q-space and direct spaces.

        Args:
            file_path (str): Path to the file containing the
                interpolation parameters.
            direct_space_voxel_size (tuple or int): Voxel size if
                provided, otherwise will take the one in the file.
            light_loading (bool): whether the to initialise the
                interpolator. Usually, if orthogonalisation was already
                done before, only the voxel size is relevant.

        Returns:
            the voxel size in any case.
        """
        with np.load(file_path) as npzfile:
            if light_loading:
                return npzfile["direct_space_voxel_size"]
            q_space_linear_transformation_matrix = npzfile[
                "q_space_linear_transformation_matrix"
            ]
            direct_lab_linear_transformation_matrix = npzfile[
                "direct_lab_linear_transformation_matrix"
            ]
            self._q_space_transitions = npzfile["q_space_transitions"]
            self.q_space_shift = npzfile["q_space_shift"]
            if direct_space_voxel_size is None:
                direct_space_voxel_size = npzfile["direct_space_voxel_size"]

        self._full_shape = self._q_space_transitions.shape[1:]
        self._cropped_shape = self._full_shape

        # Initialize the interpolators
        self.q_lab_interpolator = Interpolator3D(
            original_shape=self._full_shape,
            original_to_target_matrix=q_space_linear_transformation_matrix
        )

        self.direct_lab_interpolator = Interpolator3D(
            original_shape=self._full_shape,
            original_to_target_matrix=direct_lab_linear_transformation_matrix,
            target_voxel_size=direct_space_voxel_size
        )
        return direct_space_voxel_size

    def orthogonalize_to_direct_lab(
            self,
            direct_space_data: np.ndarray,
            detector_data: np.ndarray = None,
            direct_space_voxel_size: tuple | np.ndarray | list | float = None,
    ) -> np.ndarray:
        """
        Orthogonalize the direct space data (reconstructed object) to
        orthogonalized direct space.
        """

        if self.direct_lab_interpolator is None:
            if detector_data is None:
                raise ValueError(
                    "Provide a detector_data or initialise the interpolator "
                    "using init_interpolator() method"
                )
            self.init_interpolator(
                detector_data,
                direct_space_data.shape,
                space="direct",
                direct_space_voxel_size=direct_space_voxel_size
            )
        return self.direct_lab_interpolator(direct_space_data)

    @staticmethod
    def linear_transformation_matrix(
            grid: np.ndarray,
            index_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Compute the tranformation matrix that convert (i, j, k) integer
        position into the given grid coordinates.
        """
        return np.dot(
            grid,
            np.dot(
                index_matrix.T,
                np.linalg.inv(
                    np.dot(
                        index_matrix,
                        index_matrix.T
                    )
                )
            )
        )

    def get_q_space_transitions(self, arrangement: str = "list"):
        """
        Get the q space transitions calculated by xrayutilities and
        chose the arrangement either a list of three 1d arrays
        or a cube of q coordinates in the q lab space.
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
                self._q_space_transitions,
                source=0,
                destination=3
            )
        else:
            raise ValueError(
                "arrangement should be 'l', 'list', 'c' or 'cubinates'"
            )
    
    def get_xu_q_lab_regular_grid(self, arrangement: str="list"):
        """
        Get the regular grid used for the xu orthogonalization in the q lab
        space and chose the arrangement either a list of three 1d arrays
        or a cube of q coordinates in the q lab space.
        """
        if self.xu_gridder is None:
            raise ValueError(
                "No q lab space xu_gridder initialized, cannot provide a "
                "regular grid"
            )
        grid = [
            self.xu_gridder.xaxis,
            self.xu_gridder.yaxis,
            self.xu_gridder.zaxis
        ]
        if arrangement in ("l", "list"):
            return grid
        if arrangement in ("c", "cubinates"):
            # create a temporary meshgrid
            qx, qy, qz = np.meshgrid(grid[0], grid[1], grid[2], indexing="ij")
            return np.moveaxis(
                np.array([qx, qy, qz]),
                source=0,
                destination=3
            )
        raise ValueError(
            "arrangement should be 'l', 'list', 'c' or 'cubinates'"
        )

    def get_q_lab_regular_grid(self, arrangement: str="list"):
        """
        Get the regular grid used for the orthogonalization in the q lab
        space and chose the arrangement either a list of three 1d array
        or a cube of q coordinates in the q lab space.
        """
        if self.q_lab_interpolator is None:
            raise ValueError(
                "No q lab space interpolator initialized, cannot provide a "
                "regular grid."
            )

        grid = [
            self.q_lab_interpolator.target_grid[i]
            + self.q_space_shift[i]
            for i in range(3)
        ]
        if arrangement in ("l", "list"):
            return [grid[0][:, 0, 0], grid[1][0, :, 0], grid[2][0, 0, :]]
        if arrangement in ("c", "cubinates"):
            return np.moveaxis(grid, source=0, destination=3)
        raise ValueError(
            "arrangement should be 'l', 'list', 'c' or 'cubinates'."
        )

    def get_direct_lab_regular_grid(self, arrangement: str="list"):
        """
        Get the regular grid used for the orthogonalization in the
        direct space and chose the arrangement either a list of three
        1d array or a cube of the coordinates in the direct lab space.
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

    @staticmethod
    def lab_to_cxi_conventions(
            data: np.ndarray | tuple | list
    ) -> np.ndarray | tuple | list:
        """
        Convert the a np.ndarray, a list or a tuple from the lab frame
        system to the cxi frame conventions.
        [
            axis0=Xlab (pointing away from the light source),
            axis1=Ylab (outboard),
            axis2=Zlab (vertical up)
        ]
        will be converted into
        [
            axis0=Zcxi (pointing away from the light source),
            axis1=Ycxi (vertical up),
            axis2=Xcxi (horizontal completing the right handed system)
        ]
        """
        if isinstance(data, (tuple, list)):
            axis0, axis1, axis2 = data
            return type(data)((axis0, axis2, axis1))

        if isinstance(data, np.ndarray):
            if data.shape == (3, ):
                return np.array([data[0], data[2], data[1]])
            return np.swapaxes(data, axis1=1, axis2=2)

    @staticmethod
    def cxi_to_lab_conventions(
            data: np.ndarray | tuple | list
    ) -> np.ndarray | tuple | list:
        """
        Convert the a np.ndarray, a list or a tuple from the cxi frame
        system to the lab frame conventions.
        [
            axis0=Zcxi (pointing away from the light source),
            axis1=Ycxi (vertical up),
            axis2=Xcxi (horizontal completing the right handed system)
        ]
        will be converted into
        [
            axis0=Xlab (pointing away from the light source),
            axis1=Ylab (outboard),
            axis2=Zlab (vertical up)
        ]

        """
        if isinstance(data, (tuple, list)):
            axis0, axis1, axis2 = data
            return type(data)((axis0, axis2, axis1))
        if isinstance(data, np.ndarray):
            if data.shape == (3, ):
                return np.array([data[0], data[2], data[1]])
            return np.swapaxes(data, axis1=1, axis2=2)
        else:
            raise TypeError(
                "data should be a 3D np.ndarray, a list of 3 values, a tuple "
                "of 3 values or a np.ndarray of 3 values."
            )

    def get_q_norm_histogram(
            self,
            q_lab_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        This funtion calculates the magnitude of the Q vector for each
        position in reciprocal space in the lab frame. It returns the Q
        norms as a 1D array as well as the intensities flattened to a 1D
        array to coresspond directly with the Q norms. This can be best
        visualized using a histogram.
        The q_norms and data are sorted.
        """
        try:
            grid = self.get_q_lab_regular_grid(arrangement="c")
        except ValueError:
            grid = self.get_xu_q_lab_regular_grid(arrangement="c")

        grid = np.reshape(grid, (grid.size//3, 3))
        q_norm = np.linalg.norm(grid, axis=1)
        flattend_intensity = np.reshape(q_lab_data, q_lab_data.size)
        sort_order = q_norm.argsort()
        sorted_q_norm = q_norm[sort_order]
        sorted_flat_intens = flattend_intensity[sort_order]

        return sorted_q_norm, sorted_flat_intens


    @staticmethod
    def run_detector_calibration(
            detector_calibration_frames: np.ndarray,
            detector_outofplane_angle: float,
            detector_inplane_angle: float,
            energy: float,
            xu_detector_circles: list, # Convention should be xu not cxi
            pixel_size_x=55e-6,
            pixel_size_y=55e-6,
            sdd_estimate: float=None,
            show=True,
            verbose=True,
    ) -> dict:

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
            # how much the x_com has moved when delta has changed by one degree.
            slope = [
                np.polynomial.polynomial.polyfit(a, c, 1)[1]
                for a, c in zip(
                    (detector_outofplane_angle, detector_inplane_angle),
                    (coms[:, 0], coms[:, 1])
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
        parameter_list, _ = xu.analysis.sample_align.area_detector_calib(
            detector_inplane_angle,
            detector_outofplane_angle,
            detector_calibration_frames,
            detaxis=xu_detector_circles,
            r_i="x+",
            start=(pixel_size_x, pixel_size_y, sdd_estimate, 0, 0, 0, 0),
            fix=(True, True, False, False, False, False, True),
            wl=xu.en2lam(energy),
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
            "outerangle_offset": parameter_list[8],
        }

        if verbose:
            print("Computed parameters:\n")
            for key, value in parameters.items():
                print(
                    f"{key} = {value}"
                )
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


class Interpolator3D:
    """
    A class to handle 3D interpolations using the
    RegularGridInterpolator of scipy.interpolate. This class deals with the 
    shape of the target space based on the shape in the original space and the
    given transfer matrix.
    """
    def __init__(
            self,
            original_shape: tuple | np.ndarray | list,
            original_to_target_matrix: np.ndarray,
            target_voxel_size: tuple | np.ndarray | list | float = None,
            verbose: bool = False
    ):

        self.original_shape = original_shape

        if target_voxel_size is None:
            self.target_voxel_size = np.linalg.norm(
                original_to_target_matrix, axis=1
            )
        elif isinstance(target_voxel_size, (float, int)):
            self.target_voxel_size = np.repeat(
                target_voxel_size,
                len(original_shape)
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

        # rotate the target space grid to the orginal space
        self.target_grid_in_original_space = self._rotate_grid_axis(
            target_to_original_matrix,
            *self.target_grid
        )

    def _init_target_grid(self, verbose: bool = False) -> None:
        """
        Initialize the target space grid by finding the extent of the
        original space grid in the target space.
        """

        grid_axis0, grid_axis1, grid_axis2 = self.zero_centered_meshgrid(
            self.original_shape
        )
        
        grid_axis0, grid_axis1, grid_axis2 = self._rotate_grid_axis(
            self.original_to_target_matrix,
            grid_axis0, grid_axis1, grid_axis2
        )

        self._find_extents(grid_axis0, grid_axis1, grid_axis2)
        if verbose:
            print(
                "[INFO] the extent in the target space of a regular grid "
                f"defined in the original space with a shape of "
                f"{self.original_shape} is {self.extents}"
            )

        # define a regular grid in the target space with the computed extent
        self.target_grid = self.zero_centered_meshgrid(
            shape=self.extents,
            scale=self.target_voxel_size
        )

    @staticmethod
    def zero_centered_meshgrid(
            shape: np.ndarray | list | tuple,
            scale: np.ndarray | list | tuple = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the a zero-centered meshgrid with the 'ij' indexing numpy
        convention.
        """

        if scale is None:
            scale = [1, 1, 1]

        return np.meshgrid(
            np.arange(-shape[0]//2, shape[0]//2, 1) * scale[0],
            np.arange(-shape[1]//2, shape[1]//2, 1) * scale[1],
            np.arange(-shape[2]//2, shape[2]//2, 1) * scale[2],
            indexing="ij"
        )

    def _rotate_grid_axis(
            self,
            transfer_matrix: np.ndarray,
            grid_axis0: np.ndarray,
            grid_axis1: np.ndarray,
            grid_axis2: np.ndarray
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
            grid_axis2: np.ndarray
    ) -> None:
        """Find the extents in the 3D of a given tuple of grid."""
        extent_axis0 = int(
            np.rint(
                (grid_axis0.max() - grid_axis0.min())
                / self.target_voxel_size[0]
            )
        )
        extent_axis1 = int(
            np.rint(
                (grid_axis1.max() - grid_axis1.min())
                / self.target_voxel_size[1])
        )
        extent_axis2 = int(
            np.rint(
                (grid_axis2.max() - grid_axis2.min())
                / self.target_voxel_size[2])
        )
        self.extents = extent_axis0, extent_axis1, extent_axis2

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Wrap function to execute the interpolation as if the instance of
        Interpolator3D was a callable.
        """
        rgi = RegularGridInterpolator(
            (
                np.arange(-data.shape[0]//2, data.shape[0]//2, 1),
                np.arange(-data.shape[1]//2, data.shape[1]//2, 1),
                np.arange(-data.shape[2]//2, data.shape[2]//2, 1),
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
                    )
                )
            ).transpose()
        )

        # reshape the volume back to its original shape, thus each voxel 
        # goes back to its initial position
        interpolated_data = interpolated_data.reshape(
            (self.extents[0], self.extents[1], self.extents[2])
        ).astype(interpolated_data.dtype)

        return interpolated_data
