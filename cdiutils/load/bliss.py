from typing import Optional, Union
import numpy as np
import silx.io.h5py_utils
# import hdf5plugin
import xrayutilities as xu


def safe(func):
    def wrap(self, *args, **kwargs):
        with silx.io.h5py_utils.File(self.experiment_file_path) as h5file:
            return func(self, h5file, *args, **kwargs)
    return wrap

class BlissLoader():
    def __init__(
            self,
            experiment_file_path: str,
            detector_name: str="flexible",
            sample_name: Optional[str]=None,
            flatfield: Union[np.ndarray, str]=None
        ):
        self.experiment_file_path = experiment_file_path
        self.detector_name = detector_name
        self.sample_name = sample_name

        if isinstance(flatfield, str) and flatfield.endswith(".npz"):
            self.flatfield = np.load(flatfield)["arr_0"]
        elif isinstance(flatfield, np.ndarray):
            self.flatfield=flatfield
        elif flatfield is None:
            self.flatfield = None
        else:
            raise ValueError(
                "[ERROR] wrong value for flatfield parameter, provide a path, "
                "np.ndarray or leave it to None"
            )

    
    @safe
    def load_detector_data(
            self,
            h5file: silx.io.h5py_utils.File,
            scan: int,
            sample_name: Optional[str]=None
        ):

        if sample_name is None:
            sample_name = self.sample_name
        
        key_path = "_".join((sample_name, str(scan))) + ".1"
        if self.detector_name == "flexible":
            try:
                data = h5file[key_path + "/measurement/mpx1x4"][()]
            except KeyError:
                data = h5file[key_path + "/measurement/mpxgaas"][()]
        else:
            data = h5file[key_path + f"/measurement/{self.detector_name}"][()]
        if not self.flatfield is None: 
            data = data * self.flatfield
        return data
    
    @safe
    def show_scan_attributes(
            self,
            h5file: silx.io.h5py_utils.File,
            scan: int,
            sample_name: Optional[str]=None
        ):
        if sample_name is None:
            sample_name = self.sample_name
        key_path = "_".join((sample_name, str(scan))) + ".1"
        print(h5file[key_path].keys())
    
    @safe
    def load_motor_positions(
            self,
            h5file: silx.io.h5py_utils.File,
            scan: int,
            sample_name: Optional[str]=None,
    ):

        if sample_name is None:
            sample_name = self.sample_name

        key_path = "_".join(
             (sample_name, str(scan))
        ) + ".1/instrument/positioners"
        
        nu = h5file[key_path + "/nu"][()]
        delta = h5file[key_path + "/delta"][()]
        eta = h5file[key_path + "/eta"][()]
        phi = h5file[key_path + "/phi"][()]
        return eta, phi, nu, delta
    
    @safe
    def load_measurement_parameter(
            self,
            h5file,
            sample_name: str,
            scan: int,
            parameter_name: str
    ):
        """
        laod the measurement parameters of the specified scan
        """

        key_path = "_".join(
             (sample_name, str(scan))
        ) + ".1/measurement"
        requested_mes_parameter = h5file[f"{key_path}/{parameter_name}"][()]
        return requested_mes_parameter
    
    @safe
    def load_instrument_parameter(
            self,
            h5file,
            sample_name,
            scan,
            ins_parameter
    ):
        key_path = "_".join(
             (sample_name, str(scan))
             ) + ".1/instrument"
        requested_parameter = h5file[key_path + "/" + ins_parameter][()]
        return requested_parameter

    @safe
    def load_sample_parameter(
            self,
            h5file,
            sample_name,
            scan,
            sam_parameter
    ):
        key_path = "_".join(
             (sample_name, str(scan))
             ) + ".1/sample"
        requested_parameter = h5file[key_path + "/" + sam_parameter][()]
        return requested_parameter
    
    @safe
    def load_plotselect_parameter(
            self,
            h5file,
            sample_name,
            scan,
            plot_parameter
    ):
        key_path = "_".join(
             (sample_name, str(scan))
             ) + ".1/plotselect"
        requested_parameter = h5file[key_path + "/" + plot_parameter][()]
        return requested_parameter
    
    # def load_parameter(self, h5file, sample_name, scan, extra_path: str):
    #     key_path = "_".join(sample_name, str(scan)) + extra_path
    #     requested_parameter = h5file[key_path + "/" + ]
        
   
    
    def load_data_in_Q_space(
            self,
            scan: str,
            sample_name: str,
            hxrd: xu.HXRD
    ):
        data = self.load_detector_data(
            sample_name=sample_name,
            scan=scan,
        )
        eta, phi, nu, delta = self.load_motor_positions(
            sample_name=sample_name,
            scan=scan
        )
        detector_to_Q_space = hxrd.Ang2Q.area(eta, phi, nu, delta)# delta=(0, 0, 0, 0))
        nx, ny, nz = data.shape
        gridder = xu.Gridder3D(nx, ny, nz) # xu.FuzzyGridder
        gridder(
            detector_to_Q_space[0],
            detector_to_Q_space[1],
            detector_to_Q_space[2],
            data
        )
        qx, qy, qz = gridder.xaxis, gridder.yaxis, gridder.zaxis
        intensity = gridder.data
        return intensity, (qx, qy, qz), detector_to_Q_space, data
    
    @staticmethod
    def get_mask(channel: Optional[int]) -> np.array:
        mask = np.zeros(shape=(516, 516))
        mask[:, 255:261] = 1
        mask[255:261, :] = 1
        if channel:
            return np.repeat(mask[np.newaxis, :, :,], channel, axis=0)
        else:
            return mask