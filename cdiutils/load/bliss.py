import silx.io.h5py_utils
import hdf5plugin
import xrayutilities as xu


def safe(func):
    def wrap(self, *args, **kwargs):
        with silx.io.h5py_utils.File(self.experiment_file_path) as h5file:
            return func(self, h5file, *args, **kwargs)
    return wrap

class BlissLoader():
    def __init__(
            self,
            experiment_file_path,
            detector_name="mpx1x4",
            flatfield=None
        ):
        self.experiment_file_path = experiment_file_path
        self.detector_name = detector_name
        self.flatfield=flatfield
    
    @safe
    def load_detector_data(
            self,
            h5file,
            sample_name,
            scan,
        ):
        key_path = "_".join((sample_name, str(scan))) + ".1"
        data = h5file[key_path + f"/measurement/{self.detector_name}"][()]
        if not self.flatfield is None: 
            data = data * self.flatfield
        return data
    
    @safe
    def show_scan_attributes(self, h5file, sample_name, scan):
        key_path = "_".join((sample_name, str(scan))) + ".1"
        print(h5file[key_path].keys())
    
    @safe
    def load_motor_positions(self, h5file, sample_name, scan):
        key_path = "_".join(
             (sample_name, str(scan))
             ) + ".1/instrument/positioners"
        
        nu = h5file[key_path + "/nu"][()]
        delta = h5file[key_path + "/delta"][()]
        eta = h5file[key_path + "/eta"][()]
        phi = h5file[key_path + "/phi"][()]
        return eta, phi, nu, delta
    
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
        area = hxrd.Ang2Q.area(eta, phi, nu, delta)# delta=(0, 0, 0, 0))
        nx, ny, nz = data.shape
        gridder = xu.Gridder3D(nx, ny, nz)
        gridder(area[0], area[1], area[2], data)
        qx, qy, qz = gridder.xaxis, gridder.yaxis, gridder.zaxis
        intensity = gridder.data
        return intensity, (qx, qy, qz)