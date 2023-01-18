from typing import Union, Tuple, Optional
from scipy.ndimage import center_of_mass
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import xrayutilities as xu

from cdiutils.utils import pretty_print, center, crop_at_center
from cdiutils.geometry import Geometry
from cdiutils.plot.formatting import white_interior_ticks_labels


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

        self.q_space_transitions = None
        self.cropped_q_space_transitions = None

        self._reference_voxel = None
        self._cropped_shape = None
        self._full_shape = None

        self.transition_cubinates = None
        self.orthogonalized_cubinates = None
        self.orthogonalized_q_lab_grid = None

        self.q_lab_interpolator = None
    
    @property
    def reference_voxel(self):
        return self._reference_voxel

    @reference_voxel.setter
    def reference_voxel(self, voxel: Union[tuple, np.ndarray, list]):
        if isinstance(voxel, (list, np.ndarray)):
            self._reference_voxel = tuple(voxel)
        else:
            self._reference_voxel = voxel
        if self._cropped_shape is not None:
            self.crop_q_space_transitions()
    
    @property
    def cropped_shape(self):
        return self._cropped_shape

    @cropped_shape.setter
    def cropped_shape(self, shape: Union[tuple, np.ndarray, list]):
        self._cropped_shape = shape
        if self._reference_voxel is not None:
            self.crop_q_space_transitions()


    def init_q_space_area(self, det_calib_parameters: dict=None):
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

         
    def set_q_space_area(
            self,
            sample_outofplane_angle: Union[float, np.ndarray],
            sample_inplane_angle: Union[float, np.ndarray],
            detector_outofplane_angle: Union[float, np.ndarray],
            detector_inplane_angle:  Union[float, np.ndarray]
    ):
        self.q_space_transitions = np.asarray(
            self.hxrd.Ang2Q.area(
                sample_outofplane_angle,
                sample_inplane_angle,
                detector_inplane_angle,
                detector_outofplane_angle
            ),
        )
        self.transition_cubinates = np.moveaxis(
            self.q_space_transitions,
            source=0,
            destination=3
        )
        self._full_shape = self.q_space_transitions.shape[1:]
    
    def det_to_lab(
            self,
            ijk: Union[np.ndarray, list, tuple],
    ) -> tuple:

        if self.q_space_transitions is None:
            raise ValueError(
                "q_space_transitions is None, please set the q space area "
                "with SpaceConverter.set_q_space_area() method"
            )
        return self.do_transition(
            ijk,
            self.q_space_transitions
        )
    
    def croppped_det_to_det(self, ijk: Union[np.ndarray, list, tuple]):
        if (self._cropped_shape is None) or (self._reference_voxel is None):
            raise ValueError("Set a cropped_shape and a reference_voxel")
        return  (
            ijk 
            + (np.array(self._full_shape) - np.array(self._cropped_shape))//2
            - (np.array(self._full_shape)//2 - self._reference_voxel)
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


    def crop_q_space_transitions(self):
        q_space_transitions = np.empty((3,) + self._cropped_shape)
        for i in range(3):
            q_space_transitions[i] = crop_at_center(
                center(
                    self.q_space_transitions[i],
                    center_coordinates=self._reference_voxel
                ),
                final_shape=self._cropped_shape
            )
        self.cropped_q_space_transitions = q_space_transitions


    def orthogonalize_to_q_lab(
            self,
            data: np.ndarray,
            # reference_voxel: Optional[Union[np.ndarray, list, tuple]]=None,
            method: str="cdiutils",
            shift_method: str="center"
    ) -> np.ndarray:
        """
        Orthogonalize detector data of the reciprocal space to the lab
        (xu) frame.
        """

        shape = data.shape
        size = data.size

        if self.q_space_transitions[0].shape != shape:
            if self._reference_voxel is None:
                raise ValueError(
                    "The shape of the data to orthogonalize should be similar "
                    "to that of the raw data or you must set the "
                    "reference_voxel which is the center of the parsed cropped "
                    "data in the full raw data frame"
                )
            self.crop_q_space_transitions()
            q_space_transitions = self.cropped_q_space_transitions
        else:
            q_space_transitions = self.q_space_transitions


        if method in ("xu", "xrayutilities"):
            gridder = xu.FuzzyGridder3D(*shape)
            gridder(*q_space_transitions, data)
            self.orthogonalized_q_lab_grid = [
                gridder.xaxis,
                gridder.yaxis,
                gridder.zaxis
            ]
            # create a temporary meshgrid
            qx, qy, qz = np.meshgrid(
                gridder.xaxis,
                gridder.yaxis,
                gridder.zaxis,
                indexing="ij"
            )
            # each (i, j, k) coordinates corresponds to a 3 value vector
            # of the q lab space
            self.orthogonalized_cubinates = np.moveaxis(
                np.array([qx, qy, qz]),
                source=0,
                destination=3
            )
            return gridder.data


        if shift_method == "center":
            shift_voxel = tuple(s // 2 for s in shape)
        elif shift_method == "max":
            shift_voxel = np.unravel_index(np.argmax(data), shape)

        # Using the Interpolator3D requires the centering of the q
        # values, here we save the shift in q for later use
        q_space_shift = [
            q_space_transitions[i].item(shift_voxel)
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
        q_space_transitions = q_space_transitions.reshape(3, size)

        # create the index nd.array whose shift_voxel must
        # be the origin
        k_matrix = []
        for i in np.indices(shape):
            k_matrix.append(i - i[shift_voxel])
        k_matrix = np.array(k_matrix).reshape(3, size)

        # get the linear_transform_matrix
        linear_transformation_matrix = self.linear_transformation_matrix(
            q_space_transitions,
            k_matrix
        )

        q_voxel_size = np.linalg.norm(linear_transformation_matrix, axis=1)
        print(
            "[INFO] Voxel size in the xu frame using matrix is",
            q_voxel_size
        )

        # interpolate the diffraction pattern intensity to the
        # orthogonal q lab space
        self.q_lab_interpolator = Interpolator3D(
            original_shape=shape,
            original_to_target_matrix=linear_transformation_matrix
        )
        # get the regular grid used for the interpolation
        temp = [
            self.q_lab_interpolator.target_grid[i]
            + q_space_shift[i]
            for i in range(3)
        ]
        self.orthogonalized_cubinates = np.moveaxis(
                temp,
                source=0,
                destination=3
        )
        self.orthogonalized_q_lab_grid = np.array(
            [
                temp[0][:, 0, 0],
                temp[1][0, :, 0],
                temp[2][0, 0, :],
            ],
        )
        return self.q_lab_interpolator(data)
    

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


    def plot_orthogonalization_process(
            self,
            raw_data: np.ndarray,
            interpolated_data: np.ndarray,
            plot_at: Optional[Union[np.ndarray, list, tuple]]=None,

    ) -> matplotlib.figure.Figure:
        """
        Plot the intensity in the detector frame, index-of-q lab frame
        and q lab frame.
        """

        if plot_at is None:
            print(
                "plot_at parameter not provided, will plot the data at the "
                "center"
            )
            plot_at = tuple(e // 2 for e in raw_data.shape)

        figure, axes = plt.subplots(3, 3, figsize=(12, 8))

        axes[0, 0].matshow(np.log(raw_data[plot_at[0]]+1))
        axes[0, 0].plot(
            plot_at[2], plot_at[1], color="w", marker="x")

        axes[0, 1].matshow(
            np.log(raw_data[:, plot_at[1]]+1))
        axes[0, 1].plot(
            plot_at[2], plot_at[0], color="w", marker="x")

        axes[0, 2].matshow(
            np.log(
                np.swapaxes(
                    raw_data[:, :, plot_at[2]],
                    axis1=0,
                    axis2=1
                ) + 1
            ),
        )
        axes[0, 2].plot(
            plot_at[0], plot_at[1], color="w", marker="x")

        axes[0, 0].set_xlabel(r"detector $axis_2$")
        axes[0, 0].set_ylabel(r"detector $axis_1$")
        axes[0, 1].set_xlabel(r"detector $axis_2$")
        axes[0, 1].set_ylabel(r"detector $axis_0$")
        axes[0, 2].set_xlabel(r"detector $axis_0$")
        axes[0, 2].set_ylabel(r"detector $axis_1$")

        if raw_data.shape != self._full_shape:
            plot_at = self.croppped_det_to_det(plot_at)

        q_plot_at = self.det_to_lab(plot_at)

        index_of_q_plot_at = np.unravel_index(
            np.argmin(
                np.linalg.norm(
                    self.orthogonalized_cubinates - q_plot_at,
                    axis=3
                )
            ),
            self.orthogonalized_cubinates.shape[:-1]
        )

        axes[1, 0].matshow(
            np.log(
                np.swapaxes(interpolated_data[index_of_q_plot_at[0]],
                axis1=0,
                axis2=1
                )+1 # add 1 to avoid log(0)
            ),
            origin="lower"
        )

        axes[1, 1].matshow(
            np.log(
                np.swapaxes(
                    interpolated_data[:, index_of_q_plot_at[1]],
                    axis1=0,
                    axis2=1
                )+1 # add 1 to avoid log(0)
            ),
            origin="lower"
        )
        
        axes[1, 2].matshow(
            np.log(interpolated_data[:, :, index_of_q_plot_at[2]]+1),
            origin="lower"
        )

        axes[1, 0].set_xlabel(r"$y_{lab}/x_{cxi}$")
        axes[1, 0].set_ylabel(r"$z_{lab}/y_{cxi}$")
        axes[1, 1].set_xlabel(r"$x_{lab}/z_{cxi}$")
        axes[1, 1].set_ylabel(r"$z_{lab}/y_{cxi}$")
        axes[1, 2].set_xlabel(r"$y_{lab}/x_{cxi}$")
        axes[1, 2].set_ylabel(r"$x_{lab}/z_{cxi}$")

        # load the orthogonalized grid values
        x_array = self.orthogonalized_q_lab_grid[0]
        y_array = self.orthogonalized_q_lab_grid[1]
        z_array = self.orthogonalized_q_lab_grid[2]

        # careful here, in contourf it is not the matrix convention !
        axes[2, 0].contourf(
            y_array, # must be the matplotlib xaxis array / numpy axis1
            z_array, # must be the matplotlib yaxis array / numpy axis0
            np.log(
                np.swapaxes(
                    interpolated_data[index_of_q_plot_at[0]]+1,
                    axis1=0,
                    axis2=1
                )
            ),
            levels=100,
        )

        axes[2, 1].contourf(
            x_array, # must be the matplotlib xaxis array / numpy axis1
            z_array, # must be the matplotlib yaxis array / numpy axis0
            np.log(
                np.swapaxes(
                    interpolated_data[:, index_of_q_plot_at[1]]+1,
                    axis1=0,
                    axis2=1
                )
            ),
            levels=100,
        )

        axes[2, 2].contourf(
            y_array, # must be the matplotlib xaxis array / numpy axis1
            x_array, # must be the matplotlib yaxis array / numpy axis0
            np.log(
                interpolated_data[:, :, index_of_q_plot_at[2]]
                +1 # add 1 to avoid log(0)
            ),
            levels=100,
        )

        axes[2, 0].set_xlabel(r"$Q_{y_{lab}}~(\si{\angstrom}^{-1})$")
        axes[2, 0].set_ylabel(r"$Q_{z_{lab}}~(\si{\angstrom}^{-1})$")
        axes[2, 1].set_xlabel(r"$Q_{x_{lab}}~(\si{\angstrom}^{-1})$")
        axes[2, 1].set_ylabel(r"$Q_{z_{lab}}~(\si{\angstrom}^{-1})$")
        axes[2, 2].set_xlabel(r"$Q_{y_{lab}}~(\si{\angstrom}^{-1})$")
        axes[2, 2].set_ylabel(r"$Q_{x_{lab}}~(\si{\angstrom}^{-1})$")


        axes[0, 1].set_title(r"Raw data in \textbf{detector frame}")
        axes[1, 1].set_title(r"Orthogonalized data in \textbf{index-of-q lab frame}")
        axes[2, 1].set_title(r"Orthogonalized data in \textbf{q lab frame}")

        figure.canvas.draw()
        for ax in axes.ravel():
            # ax.tick_params(axis="x", bottom=True, top=False, labeltop=False, labelbottom=True)
            white_interior_ticks_labels(ax)
        for ax in axes[2].ravel():
            ax.set_aspect("equal")

        figure.suptitle(r"From \textbf{detector frame} to \textbf{q lab frame}")
        text = (
            "The white X marker shows the\nreference pixel (max) used for the"
            "\ntransformation"
        )
        figure.text(0.05, 0.92, text, fontsize=12, transform=figure.transFigure)
        figure.tight_layout()

        return figure


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
