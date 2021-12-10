import silx.io

def load_spec_file(path: str):
     """Load the specfile from the given path"""
     return  silx.io.open(path)