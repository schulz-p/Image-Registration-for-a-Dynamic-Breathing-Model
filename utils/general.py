import torch
import numpy as np

from utils import handle_coordinates


def fast_bilinear_interpolation(input_array, x_indices, y_indices):
    """Binilinear interpolation.
    (Implementation based on the GitHub repository by Jelmer Wolterink (MIAGroupUT):
    https://github.com/MIAGroupUT/IDIR)"""
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)

    x = x_indices - x0
    y = y_indices - y0

    output = (
        input_array[x0, y0] * (1 - x) * (1 - y)
        + input_array[x1, y0] * x * (1 - y)
        + input_array[x0, y1] * (1 - x) * y
        + input_array[x1, y1] * x * y
    )
    return output


def fast_trilinear_interpolation(input_array, x_indices, y_indices, z_indices):
    """Trilinear interpolation.
    (Implementation from the GitHub repository by Jelmer Wolterink (MIAGroupUT):
    https://github.com/MIAGroupUT/IDIR)"""
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[2] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[2] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[2] - 1)

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    output = (
        input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1] * x * (1 - y) * z
        + input_array[x0, y1, z1] * (1 - x) * y * z
        + input_array[x1, y1, z0] * x * y * (1 - z)
        + input_array[x1, y1, z1] * x * y * z
    )
    return output


def value_averaging(input_array, method="middle", gpu=True):
    """Average the values of the input array in the first dimension with a specified method."""

    num_values = input_array.size(dim=0)

    match method:
        case "middle":
            output_array = input_array[num_values // 2, :]
        case "mean":
            output_array = torch.mean(input_array, dim=0)
        case "gauss-filter":
            gauss_filter = torch.exp(-0.5 * torch.linspace(- (num_values // 2), num_values // 2, num_values) ** 2)
            gauss_filter = gauss_filter / gauss_filter.sum()
            if gpu:
                 gauss_filter = gauss_filter.cuda()
            output_array = torch.sum(gauss_filter[..., None,] * input_array, dim=0)
        case _:
            raise Exception('Method not supported')

    return output_array


def make_coordinate_tensors_slice(image_3D_dicom, image_2D_dicom, mask=None, dims_orig=(28, 28), gpu=True):
    """Make a 3D coordinate tensor with coordinates in slice of 2D image."""

    match image_2D_dicom.ImageOrientationPatient:
        case [1, 0, 0, 0, 0, -1]: view = "frontal"
        case [0, 1, 0, 0, 0, -1]: view = "sagital"
        case _:
            raise Exception('Implementation only works for frontal or sagittal slices')

    # Get axis of 2D image w.r.t. the 3D image coordinates normalised on [-1,1]^3
    d_axis_3D, h_axis_3D, w_axis_3D = handle_coordinates.get_normalised_coordinates_2ddata(image_3D_dicom, image_2D_dicom)

    if view == "frontal":
        h_axis_3D = np.linspace(h_axis_3D.min(), h_axis_3D.max(), dims_orig[0])
        w_axis_3D = np.linspace(w_axis_3D.min(), w_axis_3D.max(), dims_orig[1])
    elif view == "sagital":
        h_axis_3D = np.linspace(h_axis_3D.min(), h_axis_3D.max(), dims_orig[0])
        d_axis_3D = np.linspace(d_axis_3D.min(), d_axis_3D.max(), dims_orig[1])

    # Get axis of 2D image w.r.t. the 2D image coordinates normalised on [-1,1]^2
    h_axis_2D = np.linspace(-1, 1, dims_orig[0])
    w_axis_2D = np.linspace(-1, 1, dims_orig[1])

    # Generate coordinate tensors in 3D and 2D
    coordinate_tensor_3D = [torch.FloatTensor(d_axis_3D),
                            torch.FloatTensor(h_axis_3D),
                            torch.FloatTensor(w_axis_3D)]
    coordinate_tensor_3D = torch.meshgrid(*coordinate_tensor_3D)
    coordinate_tensor_3D = torch.stack(coordinate_tensor_3D, dim=3)

    coordinate_tensor_2D = [torch.FloatTensor(h_axis_2D),
                            torch.FloatTensor(w_axis_2D)]
    coordinate_tensor_2D = torch.meshgrid(*coordinate_tensor_2D)
    coordinate_tensor_2D = torch.stack(coordinate_tensor_2D, dim=2)

    # Restrict to coordinates which are inside the 3D image
    mask_inner_indices = ((-1 <= coordinate_tensor_3D[:,:,:,0]) * (coordinate_tensor_3D[:,:,:,0] <= 1)
                  * (-1 <= coordinate_tensor_3D[:,:,:,1]) * (coordinate_tensor_3D[:,:,:,1] <= 1)
                  * (-1 <= coordinate_tensor_3D[:,:,:,2]) * (coordinate_tensor_3D[:,:,:,2] <= 1))

    # Get dimension of cropped image
    dim_d = torch.sum(mask_inner_indices, 0).max()
    dim_h = torch.sum(mask_inner_indices, 1).max()
    dim_w = torch.sum(mask_inner_indices, 2).max()

    coordinate_tensor_3D = coordinate_tensor_3D[mask_inner_indices]
    coordinate_tensor_3D = torch.reshape(coordinate_tensor_3D, [dim_d, dim_h, dim_w, 3])

    if view == "frontal":
        coordinate_tensor_2D = coordinate_tensor_2D[mask_inner_indices[0,:,:]]
        coordinate_tensor_2D = torch.reshape(coordinate_tensor_2D, [dim_h, dim_w, 2])
        dims = [dim_h, dim_w]
    elif view == "sagital":
        coordinate_tensor_2D = coordinate_tensor_2D[torch.permute(mask_inner_indices[:, :, 0], (1, 0))]
        coordinate_tensor_2D = torch.reshape(coordinate_tensor_2D, [dim_h, dim_d, 2])
        coordinate_tensor_3D = torch.permute(coordinate_tensor_3D, (2, 1, 0, 3))
        dims = [dim_h, dim_d]

    # Reshape coordinates
    coordinate_tensor_3D = torch.reshape(coordinate_tensor_3D,
                                         [coordinate_tensor_3D.size(dim=0), np.prod(coordinate_tensor_3D.shape[1:3]),
                                          3])
    coordinate_tensor_2D = coordinate_tensor_2D.view([np.prod(coordinate_tensor_2D.shape[0:2]), 2])

    if mask == None:
        if gpu:
            coordinate_tensor_3D = coordinate_tensor_3D.cuda()
            coordinate_tensor_2D = coordinate_tensor_2D.cuda()

        return coordinate_tensor_3D, coordinate_tensor_2D, dims

    else:
        # Perhaps restrict to coordinates which are inside the mask
        mask = fast_bilinear_interpolation(mask, coordinate_tensor_2D[:, 0], coordinate_tensor_2D[:, 1]).round()
        mask_indices = (mask == 1)
        coordinate_tensor_2D = coordinate_tensor_2D[mask_indices]
        coordinate_tensor_3D = coordinate_tensor_3D[:,mask_indices]

        if gpu:
            coordinate_tensor_3D = coordinate_tensor_3D.cuda()
            coordinate_tensor_2D = coordinate_tensor_2D.cuda()

        return coordinate_tensor_3D, coordinate_tensor_2D


def make_coordinate_tensor(mask=None, dims=(28, 28, 28), gpu=True):
    """Make a 3D coordinate tensor.
    (Implementation based on the GitHub repository by Jelmer Wolterink (MIAGroupUT):
    https://github.com/MIAGroupUT/IDIR)"""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])
    if mask != None:
        coordinate_tensor = coordinate_tensor[mask.flatten() > 0, :]

    if gpu:
        coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor