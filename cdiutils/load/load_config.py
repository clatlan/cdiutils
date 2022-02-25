import silx.io
import hdf5plugin

def load_specfile(path: str):
     """Load the specfile from the given path"""
     return  silx.io.open(path)

def safe(func):
    def wrap(self, *args, **kwargs):
        with silx.io.h5py_utils.File(self.experiment_file_path) as h5file:
            return func(self, h5file, *args, **kwargs)
    return wrap

class BlissLoader():
    def __init__(self, experiment_file_path):
        self.experiment_file_path = experiment_file_path
    
    @safe
    def load_detector_data(self, h5file, sample_name, scan):
        key_path = "_".join((sample_name, str(scan))) + ".1"
        return h5file[key_path + "/measurement/mpx1x4"][()]
    
    @safe
    def show_scan_attributes(self, h5file, sample_name, scan):
        key_path = "_".join((sample_name, str(scan))) + ".1"
        print(h5file[key_path].keys())
    
    @safe
    def load_motor_positions(self, h5file, sample_name, scan):
        key_path = "_".join(
             (sample_name, str(scan))
             ) + ".1/instrument/positioners"
        
        nu = float(h5file[key_path + "/nu"][()])
        delta = float(h5file[key_path + "/delta"][()])
        eta = h5file[key_path + "/eta"][()]
        phi = h5file[key_path + "/phi"][()]
        return eta, phi, nu, delta