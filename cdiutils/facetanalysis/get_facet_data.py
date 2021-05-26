import numpy as np
import sys
from vtk.util.numpy_support import vtk_to_numpy

sys.path.append('/data/id01/inhouse/clatlan/pythonies/cdiutils')
from cdiutils.load.load_data import get_data_from_vtk
from cdiutils.facetanalysis.facet_utils import get_miller_indices

def get_facet_data(vtk_data, rotation_matrix=None):

    disp = {}
    strain = {}
    disp_avg = {}
    strain_avg = {}
    disp_std = {}
    strain_std = {}
    point_coord = {}
    miller_indices = {}

    point_data = vtk_data.GetPointData()
    cell_data = vtk_data.GetCellData()
    field_data = vtk_data.GetFieldData()

    cell_facet_ids = vtk_to_numpy(cell_data.GetArray('FacetIds'))
    facet_ids = np.unique(cell_facet_ids)
    # print(facet_ids)
    facet_nb = np.max(cell_facet_ids)
    facet_ids = vtk_to_numpy(field_data.GetArray('FacetIds'))
    # print(facet_ids)

    # if rotation matrix is provided, rotate all the facet normals.
    facet_normals = {}
    if rotation_matrix is not None:
        field_facet_ids = vtk_to_numpy(field_data.GetArray('FacetIds'))
        # print(field_facet_ids)

        for id in field_facet_ids:
            # normal = np.array([field_data.GetArray("facetNormals").GetValue((id-1)*3),
            #                    field_data.GetArray("facetNormals").GetValue((id-1)*3+1),
            #                    field_data.GetArray("facetNormals").GetValue((id-1)*3+2)])
            normal = vtk_to_numpy(field_data.GetArray("facetNormals"))[id-1]
            facet_normals[id] = np.dot(rotation_matrix, normal)
            miller_indices[id] = get_miller_indices(facet_normals[id])
            print(id,
                 normal,
                 miller_indices[id],
                 facet_normals[id],
                  vtk_to_numpy(field_data.GetArray('interplanarAngles'))[id-1])
    else:
        field_facet_ids = vtk_to_numpy(field_data.GetArray('FacetIds'))
        for id in field_facet_ids:
            facet_normals[id] = vtk_to_numpy(
                field_data.GetArray("facetNormals"))[id-1]

    facet_ids = field_facet_ids

    for facet in facet_ids:
        # If rotation matrix is provided, miller_indices may be found. In that
        # case, make the miller_indices dictionary.
        # if rotation_matrix is not None:
        #     miller_indices[facet] = get_miller_indices(facet_normals[facet])

        # get the indices of interest, i.e those corresponding to the current
        # facet.
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
        point_coord[facet] = np.array([vtk_data.GetPoint(i) \
            for i in point_oi_id])
        disp_avg[facet] = np.mean(disp[facet])
        strain_avg[facet] = np.mean(strain[facet])
        disp_std[facet] = np.std(disp[facet])
        strain_std[facet] = np.std(strain[facet])


    return {
        "disp": disp,
        "strain": strain,
        "disp_avg": disp_avg,
        "strain_avg": strain_avg,
        "disp_std": disp_std,
        "strain_std": strain_std,
        "point_coord": point_coord,
        "facet_normals": facet_normals,
        "miller_indices": miller_indices
        }
