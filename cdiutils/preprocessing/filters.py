import numpy as np
import xrayutilities as xu

def basic_filter(data, maplog_min_value=3.5):
    return np.power(xu.maplog(data, maplog_min_value, 0), 10)
