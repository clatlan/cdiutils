import numpy as np
import sys
from vtk.util.numpy_support import vtk_to_numpy

sys.path.append('/data/id01/inhouse/clatlan/pythonies/cdiutils')
from cdiutils.load.load_data import get_data_from_vtk
from cdiutils.facetanalysis.facet_utils import get_miller_indices

def get_facet_data(vtk_data):

    point_data = vtk_data.GetPointData()
    cell_data = vtk_data.GetCellData()
    field_data = vtk_data.GetFieldData()

    cell_facet_ids = vtk_to_numpy(cell_data.GetArray('FacetIds'))
    facet_ids = np.unique(cell_facet_ids)

    facet_normals = {key: value for (key, value) in \
        enumerate(vtk_to_numpy(field_data.GetArray("facetNormals")))}

    disp = {}
    strain = {}
    disp_avg = {}
    strain_avg = {}
    disp_std = {}
    strain_std = {}
    point_coord = {}

    for facet in facet_ids:
        # get the indices of interest
        indices_oi = np.where(cell_facet_ids == facet)[0]
        point_oi_id = []

        for id in indices_oi:
            cell = vtk_data.GetCell(id)
            point_oi_id.append(cell.GetPointId(0))
            point_oi_id.append(cell.GetPointId(1))
            point_oi_id.append(cell.GetPointId(2))

        point_oi_id = np.unique(point_oi_id)

        # finally get the the disp and strain of the point of interest
        disp[facet] = vtk_to_numpy(point_data.GetArray("disp"))[point_oi_id]
        strain[facet] = vtk_to_numpy(point_data.GetArray("strain"))[point_oi_id]
        point_coord[facet] = np.array([vtk_data.GetPoint(i) for i in point_oi_id])
        disp_avg[facet] = np.mean(disp[facet])
        strain_avg[facet] = np.mean(strain[facet])
        disp_std[facet] = np.std(disp[facet])
        strain_std[facet] = np.std(strain[facet])

    return disp, strain, disp_avg, strain_avg, disp_std, strain_std, \
        point_coord, facet_normals
