import torch
from objectives import regularizer
from utils import general


def ratio_convolutions(input_coords, output):
    """Compute percentage of convolutions on given coordinates."""

    # Compute determinants of Jacobian matrices
    jac = regularizer.compute_jacobian_matrix(input_coords, output)
    det_jac = torch.det(jac)

    num_conv = torch.sum(det_jac <= 0).item()

    percentage_conv = num_conv / input_coords.size(0)

    return percentage_conv


def compute_ratio_convolutions(model, num_input_coords=4E6):
    """Compute percentage of convolution on num_input_coords randomly chosen coordinates."""

    indices = torch.randperm(model.possible_coordinate_tensor.shape[0], device=model.device)[: int(num_input_coords)]
    xc = model.possible_coordinate_tensor[indices, :]
    xc = xc.requires_grad_(True)
    yc = model.network(xc)
    ratio_conv = ratio_convolutions(xc, yc)

    return ratio_conv


def compute_error(model, criterion=torch.nn.MSELoss(), num_input_coords=1E6):
    """Compute MSE/... on num_input_coords randomly chosen coordinates."""

    error = torch.zeros(len(model.moving_image), device=model.device)

    indices = torch.randperm(model.possible_coordinate_tensor.shape[0], device=model.device)[: int(num_input_coords)]
    xc = model.possible_coordinate_tensor[indices, :]
    with torch.no_grad():
        yc = model.network(xc)

    transformed_image = []
    fixed_image = []
    for i in range(len(model.moving_image)):
        transformed_image.append(
            general.fast_trilinear_interpolation(
                model.moving_image[i],
                yc[:, 0],
                yc[:, 1],
                yc[:, 2])
        )
        fixed_image.append(
            general.fast_trilinear_interpolation(
                model.fixed_image[i],
                xc[:, 0],
                xc[:, 1],
                xc[:, 2])
        )

        # Compute the loss
        error[i] = round(criterion(transformed_image[i], fixed_image[i]).item(), 4)

    return error

def compute_error_2D(model, slice2D_indices, criterion=torch.nn.MSELoss(),
                     coordinate_tensor_2D_frontal=None, coordinate_tensor_3D_frontal=None,
                     coordinate_tensor_2D_sagital=None, coordinate_tensor_3D_sagital=None):
    """Compute MSE/... of 2D slice and slice of transformed moving image on given coordinates."""

    if coordinate_tensor_2D_frontal is None:
        coordinate_tensor_2D_frontal = model.possible_coordinate_tensor_slices_2D_frontal
    if coordinate_tensor_3D_frontal is None:
        coordinate_tensor_3D_frontal = model.possible_coordinate_tensor_slices_3D_frontal
    if coordinate_tensor_2D_sagital is None:
        coordinate_tensor_2D_sagital = model.possible_coordinate_tensor_slices_2D_sagital
    if coordinate_tensor_3D_sagital is None:
        coordinate_tensor_3D_sagital = model.possible_coordinate_tensor_slices_3D_sagital

    slice = slice2D_indices[0]
    time = slice2D_indices[1]
    index = slice2D_indices[2]

    match model.slice2D_dicoms[slice][0].ImageOrientationPatient:
        case [1, 0, 0, 0, 0, -1]:  # frontal
            coordinate_tensor_2D = coordinate_tensor_2D_frontal
            coordinate_tensor_3D = coordinate_tensor_3D_frontal
            num_slices_3D = coordinate_tensor_3D.shape[0]
        case [0, 1, 0, 0, 0, -1]:  # sagital
            coordinate_tensor_2D = coordinate_tensor_2D_sagital
            coordinate_tensor_3D = coordinate_tensor_3D_sagital
            num_slices_3D = coordinate_tensor_3D.shape[0]

    slice2D_image = general.fast_bilinear_interpolation(
        model.slice2D_images[slice][index],
        coordinate_tensor_2D[:, 0],
        coordinate_tensor_2D[:, 1]
    )
    # Calculate transformed image values at 2D slice
    coordinate_tensor_3D = torch.reshape(coordinate_tensor_3D, [-1, 3])
    with torch.no_grad():
        output_slice = model.network(coordinate_tensor_3D, steps=time)   # set steps=0 if error before registration should be computed

    loss = 0
    for j in range(len(model.moving_image)):
        transformed_image_slice = general.fast_trilinear_interpolation(
            model.moving_image[j],
            output_slice[:, 0],
            output_slice[:, 1],
            output_slice[:, 2])
        transformed_image_slice = torch.reshape(transformed_image_slice, [num_slices_3D, -1])
        transformed_image_slice = general.value_averaging(transformed_image_slice,
                                                          method=model.method_value_averaging, gpu=model.gpu)
        # Compute 2D error
        loss += model.alpha_3D[j] * criterion(transformed_image_slice, slice2D_image)

    return loss


def compute_dice(model, segment_key, num_input_coords=1E6):
    """Compute dice of segment specified by segment_key on num_input_coords randomly chosen coordinates."""

    indices = torch.randperm(model.possible_coordinate_tensor.shape[0], device=model.device)[: int(num_input_coords)]
    xc = model.possible_coordinate_tensor[indices, :]
    with torch.no_grad():
        yc = model.network(xc)

    # Get images on grids
    fixed_segment = general.fast_trilinear_interpolation(
        model.fixed_segments[segment_key],
        xc[:, 0],
        xc[:, 1],
        xc[:, 2],
    ).round()
    moving_segment = general.fast_trilinear_interpolation(
        model.moving_segments[segment_key],
        xc[:, 0],
        xc[:, 1],
        xc[:, 2],
    ).round()
    transformed_segment = general.fast_trilinear_interpolation(
        model.moving_segments[segment_key],
        yc[:, 0],
        yc[:, 1],
        yc[:, 2],
    ).round()

    dice_before_reg = (2 * (fixed_segment * moving_segment).sum() /
                       (fixed_segment.sum() + moving_segment.sum()))
    dice_after_reg = (2 * (fixed_segment * transformed_segment).sum() /
                      (fixed_segment.sum() + transformed_segment.sum()))


    return dice_before_reg, dice_after_reg
