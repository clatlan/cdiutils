from typing import Union, Optional
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
import numpy as np
import xrayutilities as xu

from cdiutils.utils import pretty_print
from cdiutils.geometry import Geometry


class SpaceConverter():
    """
    A class to handle the conversions between the different frames and
    spaces.
    """
    def __init__(
            self,
            roi: Union[np.array, list, tuple],
            geometry: Optional[Geometry]=Geometry.from_name("ID01"),
            energy: Optional[float]=None
    ):

        self.geometry = geometry
        # convert the geometry to xrayutilities coordinate system
        self.geometry.cxi2xu()

        self.energy = energy
        self.roi = roi
        self.det_calib_parameters = {}
        self.hxrd = None
        self.Q_space_transitions = None


    def init_Q_space_area(self, det_calib_parameters: dict=None):
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
            qconv = xu.experiment.QConversion(
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
                qconv=qconv
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

        # if energy is None:
        #     if self.energy is None:
        #         raise ValueError(
        #             "No energy given, please provide energy value either in the"
        #             "run_detector_calibration() method or when initiliazing the"
        #             "SpaceConverter instance"
        #         )
        #     else:
        #         energy = self.energy
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

        pix0_x = np.polynomial.polynomial.polyfit(delta, x_com, 1)[0]  # pixel 0, reference of the
        # direct beam
        pix0_y = np.polynomial.polynomial.polyfit(nu, y_com, 1)[0]

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
        
        # self.det_calib_parameters = parameters
        return parameters


            
    def set_Q_space_area(
            self,
            sample_outofplane_angle: Union[float, np.array],
            sample_inplane_angle: Union[float, np.array],
            detector_outofplane_angle: Union[float, np.array],
            detector_inplane_angle:  Union[float, np.array]
    ):
        self.Q_space_transitions = self.hxrd.Ang2Q.area(
            sample_outofplane_angle,
            sample_inplane_angle,
            detector_inplane_angle,
            detector_outofplane_angle
        )
    
    def det2lab(
            self,
            detector_coordinates: Union[np.array, list, tuple],
    ) -> tuple:

        if self.Q_space_transitions is None:
            raise ValueError(
                "Q_space_transitions is None, please set the Q space area "
                "with SpaceConverter.set_Q_space_area() method"
            )
        return self.make_transition(
            detector_coordinates,
            self.Q_space_transitions
        )

    
    @staticmethod
    def make_transition(
            xyz: Union[np.array, list, tuple],
            transition_matrix: tuple
    ) -> tuple:
        new_x = transition_matrix[0][xyz[0], xyz[1], xyz[2]]
        new_y = transition_matrix[1][xyz[0], xyz[1], xyz[2]]
        new_z = transition_matrix[2][xyz[0], xyz[1], xyz[2]]
        
        return float(new_x), float(new_y), float(new_z)