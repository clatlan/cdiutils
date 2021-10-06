import numpy as np
import sys
from vtk.util.numpy_support import vtk_to_numpy

sys.path.append('/data/id01/inhouse/clatlan/pythonies/cdiutils')
from cdiutils.facetanalysis.facet_utils import get_miller_indices


def facet_data_from_vtk(
        vtk_data,
        rotation_matrix=None,
        verbose=False):
    """
    Make a dictionary of relevant facet-related data from the
    FacetAnalyzer pluggin in Paraview.

    :param vtk_data: the vtk data from Paraview's FacetAnalyzer.
    :param rotation_matrix: the transform matrix (np.array) to switch
    to another frame of reference. (Default is None).
    :param verbose: whether to print out some comments.

    :return: A dictionary of sub dictionary. Every sub dictionary has
    the same keys, and they correspond to the facet ids. Key of the
    main dictionary correspond to a quantity such as strain, stress etc.
    """

    disp = {}
    strain = {}
    disp_avg = {}
    strain_avg = {}
    disp_std = {}
    strain_std = {}
    point_coord = {}
    miller_indices = {}
    facet_normals = {}

    point_data = vtk_data.GetPointData()
    cell_data = vtk_data.GetCellData()
    field_data = vtk_data.GetFieldData()

    # Notice that the numbers of FacetIds for CellData and FieldData
    # are different. In the CellData, id 0 is taken into account and
    # corresponds to edges and corners.
    cell_facet_ids = vtk_to_numpy(cell_data.GetArray('FacetIds'))
    facet_ids = np.unique(cell_facet_ids)
    field_facet_ids = vtk_to_numpy(field_data.GetArray('FacetIds'))

    for id in facet_ids:
        if id != 0:
            # if rotation matrix is provided, rotate all the facet
            # normals.
            if rotation_matrix is not None:
                normal = vtk_to_numpy(
                    field_data.GetArray("facetNormals"))[id-1]
                facet_normals[id] = np.dot(rotation_matrix, normal)
                miller_indices[id] = get_miller_indices(facet_normals[id])
                if verbose:
                    print(
                        "Facet id: {}".format(id),
                        "Original facet normal: {}".format(normal),
                        "Rotated facet normal: {}".format(facet_normals[id]),
                        "Miller indices: {}".format(miller_indices[id])
                    )
            else:
                facet_normals[id] = vtk_to_numpy(
                    field_data.GetArray("facetNormals"))[id-1]

        # Get the indices of interest, i.e those corresponding to the
        # current facet.
        indices_oi = np.where(cell_facet_ids == id)[0]
        point_oi_id = []

        for ind in indices_oi:
            cell = vtk_data.GetCell(ind)
            point_oi_id.append(cell.GetPointId(0))
            point_oi_id.append(cell.GetPointId(1))
            point_oi_id.append(cell.GetPointId(2))

        point_oi_id = np.unique(point_oi_id)

        # finally get the the disp and strain of the point of interest
        disp[id] = vtk_to_numpy(
            point_data.GetArray("disp"))[point_oi_id]
        strain[id] = vtk_to_numpy(
            point_data.GetArray("strain"))[point_oi_id]
        point_coord[id] = np.array([vtk_data.GetPoint(i)
                                    for i in point_oi_id])
        disp_avg[id] = np.mean(disp[id])
        strain_avg[id] = np.mean(strain[id])
        disp_std[id] = np.std(disp[id])
        strain_std[id] = np.std(strain[id])

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
