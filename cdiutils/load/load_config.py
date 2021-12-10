import silx.io

def load_specfile(path: str):
     """Load the specfile from the given path"""
     return  silx.io.open(path)