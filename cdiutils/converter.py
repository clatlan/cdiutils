from typing import Union, Tuple, Optional
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import xrayutilities as xu

from cdiutils.utils import pretty_print, center, crop_at_center
from cdiutils.geometry import Geometry


class SpaceConverter():
    """
    A class to handle the conversions between the different frames and
    spaces.
    """
    def __init__(
            self,
            geometry: Geometry,
            roi: Union[np.ndarray, list, tuple],
            energy: Optional[float]=None
    ):
        self.geometry = geometry
        # convert the geometry to xrayutilities coordinate system
        self.geometry.cxi_to_xu()

        self.energy = energy
        self.roi = roi
        self.det_calib_parameters = {}
        self.hxrd = None

        self._q_space_transitions = None

        self._reference_voxel = None
        self._cropped_shape = None
        self._full_shape = None

        self.q_space_shift = None

        self.q_lab_interpolator: Interpolator3D=None
        self.direct_lab_interpolator: Interpolator3D=None
        self.xu_gridder: xu.FuzzyGridder3D=None
    
    @property
    def q_space_transition(self):
        return self._q_space_transitions
    
    @property
    def reference_voxel(self):
        return self._reference_voxel

    @reference_voxel.setter
    def reference_voxel(self, voxel: Union[tuple, np.ndarray, list]):
        if isinstance(voxel, (list, np.ndarray)):
            self._reference_voxel = tuple(voxel)
        else:
            self._reference_voxel = voxel
    
    @property
    def cropped_shape(self):
        return self._cropped_shape

    @cropped_shape.setter
    def cropped_shape(self, shape: Union[tuple, np.ndarray, list]):
        self._cropped_shape = shape
    
    @property
    def full_shape(self):
        return self._full_shape

    def init_q_space_area(self, det_calib_parameters: dict=None):
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

        if np.all(
                [
                    k in ["cch1", "cch2", "pwidth1", "pwidth2", "distance",
                          "tiltazimuth", "tilt", "detrot", "outerangle_offset"]
                    for k in det_calib_parameters.keys()
                ]
            ):
            qconversion = xu.experiment.QConversion(
                sampleAxis=self.geometry.sample_circles,
                detectorAxis=self.geometry.detector_circles,
                r_i=self.geometry.beam_direction
            )
            self.hxrd = xu.HXRD(
                idir=self.geometry.beam_direction, # defines the inplane
                # reference direction (idir points into the beam
                # direction at zero angles)
                ndir=[0, 0, 1], # defines the surface normal of your sample
                # (ndir points along the innermost sample rotation axis)
                en=self.energy,
                qconv=qconversion
            )
            self.hxrd.Ang2Q.init_area(
                detectorDir1=self.geometry.detector_vertical_orientation,
                detectorDir2=self.geometry.detector_horizontal_orientation,
                cch1=det_calib_parameters["cch1"] - self.roi[0],
                cch2=det_calib_parameters["cch2"] - self.roi[2],
                Nch1=self.roi[1] - self.roi[0],
                Nch2=self.roi[3] - self.roi[2],
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
            sample_outofplane_angle: Union[float, np.ndarray],
            sample_inplane_angle: Union[float, np.ndarray],
            detector_outofplane_angle: Union[float, np.ndarray],
            detector_inplane_angle:  Union[float, np.ndarray]
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
        # self._q_space_transitions = np.empty((3, ) + q_space_transitions[0].shape)
        # print("Hello")
        # iteratable = (q_space_transitions[i] for i in range(3))
        # print("Hello")
        # self._q_space_transitions = np.fromiter(iteratable,  dtype=np.dtype((float, q_space_transitions[0].shape)))
        # print("Hello")

        # for i in range(3):
        #     self._q_space_transitions[i] = q_space_transitions[i]
        self._full_shape = self._q_space_transitions.shape[1:]

    def index_det_to_q_lab(
            self,
            ijk: Union[np.ndarray, list, tuple],
    ) -> tuple:
        """
        Transition an index from the detector frame to the reciprocal 
        lab space
        """

        if self._q_space_transitions is None:
            raise ValueError(
                "q_space_transitions is None, please set the q space area "
                "with SpaceConverter.set_q_space_area() method"
            )
        return self.do_transition(
            ijk,
            self._q_space_transitions
        )

    def index_cropped_det_to_det(
            self,
            ijk: Union[np.ndarray, list, tuple]
    ) -> tuple:
        """
        Transition an index of the cropped detector frame to the full 
        detector frame 
        """
        if (self._cropped_shape is None) or (self._reference_voxel is None):
            raise ValueError("Set a cropped_shape and a reference_voxel")
        return tuple(
            ijk
            + (np.array(self._full_shape) - np.array(self._cropped_shape))//2
            - (np.array(self._full_shape)//2 - self._reference_voxel)
        )

    def index_cropped_det_to_q_lab(
                self,
                ijk: Union[np.ndarray, list, tuple]
    ) -> tuple:
        """
        Transition an index from the cropped detector frame to the 
        reciprocal lab space frame
        """
        return self.index_det_to_q_lab(self.index_cropped_det_to_det(ijk))
    
    def index_cropped_det_to_index_of_q_lab(
                self,
                ijk: Union[np.ndarray, list, tuple]
    ) -> tuple:
        """
        Transition an index from the cropped detector frame to the 
        index-of-q lab frame
        """
        cubinates = self.get_q_lab_regular_grid(arrangement="cubinates")
        ijk = self.index_cropped_det_to_q_lab(ijk) # q value
        ijk = np.unravel_index(
            np.argmin(
                np.linalg.norm(
                    cubinates - ijk,
                    axis=3
                )
            ),
            cubinates.shape[:-1]
        )
        return ijk
    
    def dspacing(
                self,
                q_lab_coordinates: Union[np.ndarray, list, tuple]
    )-> float:
        """
        Compute the dspacing
        """
        return 2*np.pi / np.linalg.norm(q_lab_coordinates)

    def lattice_parameter(
                self,
                q_lab_coordinates: Union[np.ndarray, list, tuple],
                hkl: Union[np.ndarray, list, tuple]
    ) -> float:
        """
        Compute the lattice parameter
        """
        return (
                self.dspacing(q_lab_coordinates)
                * np.sqrt(hkl[0]**2 + hkl[1]**2 + hkl[2]**2)
        )

    @staticmethod
    def do_transition(
            ijk: Union[np.ndarray, list, tuple],
            transition_matrix: np.ndarray
    ) -> tuple:
        """
        Transform a (i, j, k) tuple into the corresponding qx, qy, qz) 
        values based the transition matrix
        """
        new_x = transition_matrix[0][ijk[0], ijk[1], ijk[2]]
        new_y = transition_matrix[1][ijk[0], ijk[1], ijk[2]]
        new_z = transition_matrix[2][ijk[0], ijk[1], ijk[2]]
        
        return float(new_x), float(new_y), float(new_z)

    def crop_q_space_transitions(self) -> None:
        """
        Crop the _q_space_transitions according to the cropped data
        shape
        """
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
            method: str="cdiutils",
            shift_method: str="center"
    ) -> np.ndarray:
        """
        Orthogonalize detector data of the reciprocal space to the lab
        (xu) frame.
        """

        self._check_shape(data.shape)
        if self._q_space_transitions[0].shape != data.shape:

            self.crop_q_space_transitions()
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
                shift_method=shift_method

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
                "to that of the raw data or you must set the "
                "reference_voxel which is the center of the parsed cropped"
                " data in the full raw data frame"
            )

    def init_interpolator(
            self,
            detector_data: np.ndarray,
            direct_space_data_shape: Optional[tuple]=None,
            direct_space_voxel_size: Union[tuple, np.ndarray, list, float]=None,
            space: str="direct",
            shift_method: str="center"
    ):
        shape = detector_data.shape
        size = detector_data.size

        self._check_shape(shape)

        self.crop_q_space_transitions()
        q_space_transitions = self.crop_q_space_transitions()

        if shift_method == "center":
            shift_voxel = tuple(s // 2 for s in shape)
        elif shift_method == "max":
            shift_voxel = np.unravel_index(np.argmax(detector_data), shape)
        
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

        if space in (
                "q", "rcp", "rcp_space",
                "reciprocal", "reciprocal_space", "both"):
            
            if direct_space_data_shape is None:
                raise ValueError(
                    "if space is 'direct' direct_space_data must be provided"
                )

            if shape != direct_space_data_shape:
                raise ValueError(
                    "The cropped_raw_data should have the same shape as the"
                    "reconstructed object"
                )
    
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
                "orthognolization process",
                direct_lab_voxel_size
            )

            # make the interpolator instance
            self.direct_lab_interpolator = Interpolator3D(
                original_shape=shape,
                original_to_target_matrix=direct_lab_transformation_matrix,
                target_voxel_size=direct_space_voxel_size
            )
        
        else:
            raise ValueError(
                "Unknown space"
            )
            
    def orthogonalize_to_direct_lab(
            self,
            direct_space_data: np.ndarray,
            detector_data: Optional[np.ndarray]=None,
            direct_space_voxel_size: Union[tuple, np.ndarray, list, float]=None,
    ) -> np.ndarray:
        """
        Orthogonalize the direct space data (reconstructed object) to
        orthogonalized direct space.
        """

        if self.direct_lab_interpolator is None:
            if detector_data is None:
                raise ValueError(
                    "Provide a detector_data or initiliase the interpolator "
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

    def get_q_space_transitions(self, arrangement: str="list"):
        """
        Get the q space transitions calculated by xrayutilities and 
        chose the arrangement either a list of three 1d array
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
        space and chose the arrangement either a list of three 1d array
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
        elif arrangement in ("c", "cubinates"):
            # create a temporary meshgrid
            qx, qy, qz = np.meshgrid(grid[0], grid[1], grid[2], indexing="ij")
            return np.moveaxis(
                np.array([qx, qy, qz]),
                source=0,
                destination=3
            )
        else:
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
                "regular grid"
            )

        grid = [
            self.q_lab_interpolator.target_grid[i]
            + self.q_space_shift[i]
            for i in range(3)
        ]
        if arrangement in ("l", "list"):
            return [grid[0][:, 0, 0], grid[1][0, :, 0], grid[2][0, 0, :]]
        elif arrangement in ("c", "cubinates"):
            return np.moveaxis(grid, source=0, destination=3)
        else:
            raise ValueError(
                "arrangement should be 'l', 'list', 'c' or 'cubinates'"
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
        elif arrangement in ("c", "cubinates"):
            return np.moveaxis(grid, source=0, destination=3)
        else:
            raise ValueError(
                "arrangement should be 'l', 'list', 'c' or 'cubinates'"
            )

    @staticmethod
    def lab_to_cxi_conventions(
            data: Union[np.ndarray, tuple, list]
    ) -> Union[np.ndarray, tuple, list]:
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
        elif isinstance(data, np.ndarray):
            if data.shape == (3, ):
                return np.array([data[0], data[2], data[1]])
            else:
                return np.swapaxes(data, axis1=1, axis2=2)
        else:
            raise TypeError(
                "data should be a 3D np.ndarray, a list of 3 values, a tuple "
                "of 3 values or a np.ndarray of 3 values."
            )



    @staticmethod
    def run_detector_calibration(
            detector_calibration_frames: np.ndarray,
            delta: float,
            nu: float,
            energy: float,
            pixel_size_x=55e-6,
            pixel_size_y=55e-6,
            sdd_estimate: float=None,
            show=True,
            verbose=True,
    ) -> dict:

        x_com = []
        y_com = []
        for i in range(detector_calibration_frames.shape[0]):
            com = center_of_mass(detector_calibration_frames[i])
            x_com.append(com[0])
            y_com.append(com[1])
        
        # get the sample to detector distance
        # for that determine how much the the com has moved when the
        # detector has rotated by 1 degree. We may find this value with
        # delta or nu. Here, we do both and calculate the average. The
        # leading coefficient of the function x_com = f(delta) gives
        # how much the x_com has moved when delta has changed by one degree.

        x_com_shift = np.polynomial.polynomial.polyfit(delta, x_com, 1)[1]
        y_com_shift = np.polynomial.polynomial.polyfit(nu, y_com, 1)[1]

        angle1, angle2 = nu, delta
        if sdd_estimate is None:
            sdd_estimate = (
                (1 / 2)
                * (1 / np.tan(np.pi / 180))
                * (x_com_shift + y_com_shift)
                * 55e-6
            )

        if verbose:
            print("[INFO] First estimate of sdd: {}\n".format(sdd_estimate))
        pretty_print(
            "[INFO] Processing to detector calibration using area_detector_calib"
        )
        parameter_list, _ = xu.analysis.sample_align.area_detector_calib(
            angle1,
            angle2,
            detector_calibration_frames,
            ["z-", "y-"],
            "x+",
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
            pretty_print("Computed parameters")
            for key, value in parameters.items():
                print(
                    f"{key} = {value}"
                )
        if show:
            fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
            fig2, axes2 = plt.subplots(1, 2)
            ax1.imshow(np.log10(detector_calibration_frames.sum(axis=0)))
            axes2[0].plot(delta, x_com)
            axes2[0].set_xlabel("delta")
            axes2[0].set_ylabel("COM in x")

            axes2[1].plot(nu, y_com)
            axes2[1].set_xlabel("nu")
            axes2[1].set_ylabel("COM in y")
            fig1.tight_layout()
            fig2.tight_layout()
        
        return parameters


class Interpolator3D:
    """
    A class to handle 3D interpolations using the
    RegularGridInterpolator of scipy.interpolate. This class deals with the 
    shape of the target space based on the shape in the original sapce and the
    given transfer matrix.
    """
    def __init__(
            self,
            original_shape: Union[tuple, np.ndarray, list],
            original_to_target_matrix: np.ndarray,
            target_voxel_size: Union[tuple, np.ndarray, list, float]=None
    ):

        self.original_shape = original_shape

        if target_voxel_size is None:
            self.target_voxel_size = np.linalg.norm(
                original_to_target_matrix, axis=1
            )
        elif isinstance(target_voxel_size, (float, int)):
            self.target_voxel_size = np.repeat(target_voxel_size, 3)
        else:
            self.target_voxel_size = target_voxel_size

        self.original_to_target_matrix = original_to_target_matrix
        
        # invert the provided rotation matrix
        target_to_original_matrix = np.linalg.inv(original_to_target_matrix)

        self.extents = None

        # initialize the grid in the target space
        self.target_grid = None
        self._init_target_grid()

        # rotate the target space grid to the orginal space
        self.target_grid_in_original_space = self._rotate_grid_axis(
            target_to_original_matrix,
            *self.target_grid
        )

    def _init_target_grid(self) -> None:
        """
        Initialize the target space grid by finding the extent of the 
        original space grid in the target space.
        """

        grid_axis0, grid_axis1, grid_axis2 = self._zero_centered_meshgrid(
            self.original_shape
        )
        
        grid_axis0, grid_axis1, grid_axis2 = self._rotate_grid_axis(
            self.original_to_target_matrix,
            grid_axis0, grid_axis1, grid_axis2
        )

        self._find_extents(grid_axis0, grid_axis1, grid_axis2)

        print(
            "[INFO] the extent in the target space of a regular grid defined "
            f"in the original space with a shape of {self.original_shape} is "
            f"{self.extents}"
        )

        # define a regular grid in the target space with the computed extent
        self.target_grid = self._zero_centered_meshgrid(
            shape=self.extents,
            scale=self.target_voxel_size
        )

    def _zero_centered_meshgrid(
            self,
            shape: Union[np.ndarray, list, tuple],
            scale: Optional[Union[np.ndarray, list, tuple]]=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        Override the __call__ function. When called, the class 
        instance runs the interpolation.
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
