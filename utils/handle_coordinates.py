import numpy as np

def get_world_coordinates_3ddata(data3d_dicom):
    """Compute world coordinates of the 3D data (helper function for get_normalised_coordinates_2ddata)."""
    d_axis_3d = data3d_dicom[0].ImagePositionPatient[1] + data3d_dicom[0].SliceThickness * np.arange(len(data3d_dicom))
    h_axis_3d = data3d_dicom[0].ImagePositionPatient[2] - data3d_dicom[0].PixelSpacing[1] * np.arange(
        data3d_dicom[0].Rows)
    w_axis_3d = data3d_dicom[0].ImagePositionPatient[0] + data3d_dicom[0].PixelSpacing[0] * np.arange(data3d_dicom[0].Columns)
    return d_axis_3d, h_axis_3d, w_axis_3d


def get_world_coordinates_2ddata(slice2d_dicom, slice_orientation='frontal'):
    """Compute world coordinates of the 2D data (helper function for get_normalised_coordinates_2ddata)."""
    h_axis_2d = slice2d_dicom.ImagePositionPatient[2] - slice2d_dicom.PixelSpacing[1] * np.arange(slice2d_dicom.Rows)
    if slice_orientation == 'frontal':
        # Depth is single value, width is array
        d_axis_2d = slice2d_dicom.ImagePositionPatient[1]
        w_axis_2d = slice2d_dicom.ImagePositionPatient[0] + slice2d_dicom.PixelSpacing[0] * np.arange(slice2d_dicom.Columns)
    if slice_orientation == 'sagittal':
        # Depth is array, width is single value
        d_axis_2d = slice2d_dicom.ImagePositionPatient[1] + slice2d_dicom.PixelSpacing[0] * np.arange(slice2d_dicom.Columns)
        w_axis_2d = slice2d_dicom.ImagePositionPatient[0]
    return d_axis_2d, h_axis_2d, w_axis_2d


def get_normalised_coordinates_2ddata(data3d_dicom, slice2d_dicom, deep_slice=True):
    """ Compute depth, height and width axes of data in slice2d_dicom w.r.t. data in data3d normalised on [-1,1]^3."""
    # Note: All coordinates correspond to voxels starting in the front top left corner

    # Get orientation of 2d data
    match slice2d_dicom.ImageOrientationPatient:
        case [1, 0, 0, 0, 0, -1]:
            slice_orientation = 'frontal'
        case [0, 1, 0, 0, 0, -1]:
            slice_orientation = 'sagittal'
        case _:
            raise Exception('Implementation only works for frontal or sagittal slices')

    # Get world coordinates of 2d and 3d data
    d_axis_2d, h_axis_2d, w_axis_2d = get_world_coordinates_2ddata(slice2d_dicom, slice_orientation)
    d_axis_3d, h_axis_3d, w_axis_3d = get_world_coordinates_3ddata(data3d_dicom)

    # Optionally extend scalar axis (of slice position) to array respecting the different thicknesses of the slice
    if deep_slice:
        slice2d_thickness = slice2d_dicom.SliceThickness
        match slice_orientation:
            case 'frontal':
                slice3d_thickness = data3d_dicom[0].SliceThickness
                n_points = np.round(slice2d_thickness / slice3d_thickness / 2).astype(int)
                d_axis_2d = d_axis_2d + slice3d_thickness * np.arange(- n_points, n_points + 1)
            case 'sagittal':
                slice3d_thickness = data3d_dicom[0].PixelSpacing[0]
                n_points = np.round(slice2d_thickness / slice3d_thickness / 2).astype(int)
                w_axis_2d = w_axis_2d + slice3d_thickness * np.arange(- n_points, n_points + 1)

    # Compute coordinates of 2d data such that they correspond to coordinates of 3d data normalised to [-1,1]^3
    d_axis_normalised = 2 * (d_axis_2d - d_axis_3d.min()) / (d_axis_3d.max() - d_axis_3d.min()) - 1
    h_axis_normalised = - 2 * (h_axis_2d - h_axis_3d.min()) / (h_axis_3d.max() - h_axis_3d.min()) + 1
    w_axis_normalised = 2 * (w_axis_2d - w_axis_3d.min()) / (w_axis_3d.max() - w_axis_3d.min()) - 1
    return d_axis_normalised, h_axis_normalised, w_axis_normalised
