import torch
import matplotlib.pyplot as plt
from skimage import util
import numpy as np
from math import ceil

from utils import general


def visualize_final_images(model, images_3D_index = 0, resolution=[100, 100, 100], dim=0):
    """Visualize fixed image, transformed moving image and the difference images before and after registration."""
    # Get grids
    xc = general.make_coordinate_tensor(dims=resolution, gpu=model.gpu)
    with torch.no_grad():
        yc = model.network(xc)

    # Get images on grids
    fixed_image = general.fast_trilinear_interpolation(
        model.fixed_image[images_3D_index],
        xc[:, 0],
        xc[:, 1],
        xc[:, 2],
    )
    moving_image = general.fast_trilinear_interpolation(
        model.moving_image[images_3D_index],
        xc[:, 0],
        xc[:, 1],
        xc[:, 2],
    )
    transformed_image = general.fast_trilinear_interpolation(
        model.moving_image[images_3D_index],
        yc[:, 0],
        yc[:, 1],
        yc[:, 2],
    )

    # Reshape and permute images such that correct slices can be extracted
    match dim:
        case 0:
            d = (0, 1, 2)
        case 1:
            d = (1, 0, 2)
        case 2:
            d = (2, 1, 0)
        case _:
            print('d<3 not fulfilled')
            return

    fixed_image = torch.reshape(fixed_image, resolution).cpu()
    moving_image = torch.reshape(moving_image, resolution).cpu()
    transformed_image = torch.reshape(transformed_image, resolution).cpu().detach()
    fixed_image = torch.permute(fixed_image, d)
    moving_image = torch.permute(moving_image, d)
    transformed_image = torch.permute(transformed_image, d)

    # Create image montages
    step_size = ceil(fixed_image.shape[0]/4)
    fixed_montage = util.montage(fixed_image[::step_size], padding_width=4, fill=None)
    moving_montage = util.montage(moving_image[::step_size], padding_width=4, fill=None)
    transformed_montage = util.montage(transformed_image[::step_size], padding_width=4, fill=None)

    # Set minimal and maximal visualization value for images and difference images, select slice
    min_val = 0
    max_val = 1
    min_diff = -1
    max_diff = 1
    slice = int(resolution[dim] / 2)

    # Plot images
    plt.rc('font', size=6)
    fig, ax = plt.subplots(2, 4)
    im1 = ax[0, 0].imshow(fixed_image[slice, :, :], vmin=min_val, vmax=max_val)
    im2 = ax[0, 1].imshow(transformed_image[slice, :, :], vmin=min_val, vmax=max_val)
    fig.colorbar(im1, ax=ax[0, 0])
    fig.colorbar(im2, ax=ax[0, 1])
    ax[1, 0].imshow(fixed_montage, vmin=min_val, vmax=max_val)
    ax[1, 1].imshow(transformed_montage, vmin=min_val, vmax=max_val)
    ax[1, 0].axis('off')
    ax[1, 1].axis('off')

    # Plot difference images
    im_diff1 = ax[0, 2].imshow((fixed_image - moving_image)[slice, :, :], vmin=min_diff, vmax=max_diff)
    im_diff2 = ax[0, 3].imshow((fixed_image - transformed_image)[slice, :, :], vmin=min_diff, vmax=max_diff)
    fig.colorbar(im_diff1, ax=ax[0, 2])
    fig.colorbar(im_diff2, ax=ax[0, 3])
    ax[1, 2].imshow(fixed_montage - moving_montage, vmin=min_diff, vmax=max_diff)
    ax[1, 3].imshow(fixed_montage - transformed_montage, vmin=min_diff, vmax=max_diff)
    ax[1, 2].axis('off')
    ax[1, 3].axis('off')

    # Set titles
    ax[0, 0].set_title('Fixed image')
    ax[0, 1].set_title('Transformed \n moving image')
    ax[0, 2].set_title('Difference \n fixed, moving')
    ax[0, 3].set_title('Difference \n fixed, transformed')

    fig.tight_layout()


def visualize_moving_image_over_time(model, images_3D_index = 0, resolution=[100, 100, 100], dim=0):
    """Visualize a slice of moving image over the time series together with
    - the difference images between the fixed and moving image
    - the difference images between a randomly chosen 2D slice and the corresponding slice of the moving image
    - the difference images between the randomly chosen 2D slice and the corresponding slice of the moving image
      without registration
    - the randomly chosen 2D slice."""

    # Set variables for reshape and permute such that correct slices are visualised
    match dim:
        case 0:
            d = (0, 1, 2)
            resolution2D = resolution[1:]
        case 1:
            d = (1, 0, 2)
            resolution2D = [resolution[0], resolution[2]]
        case 2:
            d = (2, 0, 1)
            resolution2D = resolution[:2]
        case _:
            raise Exception('dim<3 not fulfilled')

    # Get 3D grid
    xc = general.make_coordinate_tensor(dims=resolution, gpu=model.gpu)

    # Get fixed image on 3D grid
    fixed_image = general.fast_trilinear_interpolation(
        model.fixed_image[images_3D_index],
        xc[:, 0],
        xc[:, 1],
        xc[:, 2],
    )

    fixed_image = torch.reshape(fixed_image, resolution).cpu()
    fixed_image = torch.permute(fixed_image, d)

    transformed_image_set = []
    slice2D_image_set = []
    moving_image_slice_set = []
    transformed_image_slice_set = []

    # Go through time steps
    for j in range(model.n_layers + 1):

        # Get transformed moving image
        with torch.no_grad():
            yc = model.network(xc, steps=j)
        transformed_image = general.fast_trilinear_interpolation(
            model.moving_image[images_3D_index],
            yc[:, 0],
            yc[:, 1],
            yc[:, 2],
        )
        transformed_image = torch.reshape(transformed_image, resolution).cpu().detach()
        transformed_image = torch.permute(transformed_image, d)
        transformed_image_set.append(transformed_image)

        # Get random 2D image corresponding to current time step and transformed moving image slice
        np.random.seed(119)
        indices = np.where(model.slice2D_indices[:,1] == j)[0]
        ind = np.random.choice(indices)
        slice = model.slice2D_indices[ind, 0]
        index = model.slice2D_indices[ind, 2]

        xc_slice_3D, xc_slice_2D, resolution2D_cropped = general.make_coordinate_tensors_slice(
            model.fixed_dicom[0], model.slice2D_dicoms[slice][index],
            dims_orig=model.slice2D_images[slice][index].shape, gpu=model.gpu
        )
        num_slices_3D = xc_slice_3D.size(dim=0)

        # 2D image
        slice2D_image = general.fast_bilinear_interpolation(
            model.slice2D_images[slice][index],
            xc_slice_2D[:, 0],
            xc_slice_2D[:, 1]
        )
        slice2D_image_set.append(torch.reshape(slice2D_image, resolution2D_cropped).cpu())

        # Slice of moving image without deformation
        xc_slice_3D = torch.reshape(xc_slice_3D, [-1, 3])
        moving_image_slice = general.fast_trilinear_interpolation(
            model.moving_image[images_3D_index],
            xc_slice_3D[:, 0],
            xc_slice_3D[:, 1],
            xc_slice_3D[:, 2],
        )
        moving_image_slice = torch.reshape(moving_image_slice, [num_slices_3D, -1])
        moving_image_slice = general.value_averaging(moving_image_slice, method=model.method_value_averaging, gpu=model.gpu)
        moving_image_slice_set.append(
            torch.reshape(moving_image_slice, resolution2D_cropped).cpu().detach())

        # Slice of transformed moving image
        with torch.no_grad():
            yc_slice = model.network(xc_slice_3D, steps=j)
        transformed_image_slice = general.fast_trilinear_interpolation(
            model.moving_image[images_3D_index],
            yc_slice[:, 0],
            yc_slice[:, 1],
            yc_slice[:, 2],
        )
        transformed_image_slice = torch.reshape(transformed_image_slice, [num_slices_3D, -1])
        transformed_image_slice = general.value_averaging(transformed_image_slice, method=model.method_value_averaging, gpu=model.gpu)
        transformed_image_slice_set.append(torch.reshape(transformed_image_slice, resolution2D_cropped).cpu().detach())


    # Plot images and difference images
    fig, ax = plt.subplots(5, model.n_layers + 1)
    plt.rc('font', size=6)

    min_val = 0
    max_val = 1
    min_diff = -1
    max_diff = 1
    min_diff_2D = -1
    max_diff_2D = 1

    for j in range(model.n_layers + 1):
        index_middle_slice = int(resolution[dim] / 2)

        ax[0, j].imshow(transformed_image_set[j][index_middle_slice, :, :], vmin=min_val, vmax=max_val)
        ax[1, j].imshow((fixed_image - transformed_image_set[j])[index_middle_slice, :, :], vmin=min_diff, vmax=max_diff)
        ax[2, j].imshow(slice2D_image_set[j] - transformed_image_slice_set[j], vmin=min_diff_2D, vmax=max_diff_2D)
        ax[3, j].imshow(slice2D_image_set[j] - moving_image_slice_set[j], vmin=min_diff_2D, vmax=max_diff_2D)
        ax[4, j].imshow(slice2D_image_set[j], vmin=min_val, vmax=max_val)

        ax[0, j].axis('off')
        ax[1, j].axis('off')
        ax[2, j].axis('off')
        ax[3, j].axis('off')
        ax[4, j].axis('off')

        ax[0, j].set_title('step=' + str(j))
        if j==0:
            ax[0, j].text(-0.5, 0.5, 'Transformed \n moving \n image', horizontalalignment='center',
                          verticalalignment='center', transform=ax[0, j].transAxes)
            ax[1, j].text(-0.5, 0.5, 'Difference \n fixed and \n transformed \n image', horizontalalignment='center',
                          verticalalignment='center', transform=ax[1, j].transAxes)
            ax[2, j].text(-0.5, 0.5, 'Difference \n 2D-slice and \n transformed \n image', horizontalalignment='center',
                          verticalalignment='center', transform=ax[2, j].transAxes)
            ax[3, j].text(-0.5, 0.5, 'Difference \n 2D-slice and \n moving \n image \n before \n transformation', horizontalalignment='center',
                          verticalalignment='center', transform=ax[3, j].transAxes)
            ax[4, j].text(-0.5, 0.5, '2D-slice', horizontalalignment='center',
                          verticalalignment='center', transform=ax[4, j].transAxes)

    fig.tight_layout()


def visualize_segments(model, segment_key, resolution=[100, 100, 100], dim=0):
    """Visualize the alignment of the segments specified by the given segmentation key"""
    # Get grids
    xc = general.make_coordinate_tensor(dims=resolution, gpu=model.gpu)
    with torch.no_grad():
        yc = model.network(xc)

    # Get images on grids
    fixed_segment = general.fast_trilinear_interpolation(
        model.fixed_segments[segment_key],
        xc[:, 0],
        xc[:, 1],
        xc[:, 2],
    )
    moving_segment = general.fast_trilinear_interpolation(
        model.moving_segments[segment_key],
        xc[:, 0],
        xc[:, 1],
        xc[:, 2],
    )
    transformed_segment = general.fast_trilinear_interpolation(
        model.moving_segments[segment_key],
        yc[:, 0],
        yc[:, 1],
        yc[:, 2],
    )

    # Reshape and permute images such that correct slices can be extracted
    match dim:
        case 0:
            d = (0, 1, 2)
        case 1:
            d = (1, 0, 2)
        case 2:
            d = (2, 1, 0)
        case _:
            print('d<3 not fulfilled')
            return

    segments_before_reg = fixed_segment + 2 * moving_segment
    segments_after_reg = fixed_segment + 2 * transformed_segment
    segments_before_reg[segments_before_reg < 0.2] = torch.nan
    segments_after_reg[segments_after_reg < 0.2] = torch.nan

    segments_before_reg = torch.reshape(segments_before_reg, resolution).cpu()
    segments_after_reg = torch.reshape(segments_after_reg, resolution).cpu().detach()
    segments_before_reg = torch.permute(segments_before_reg, d)
    segments_after_reg = torch.permute(segments_after_reg, d)

    # Create image montages
    step_size = ceil(segments_before_reg.shape[0] // 16)
    segments_before_reg = util.montage(segments_before_reg[::step_size], padding_width=4, fill=None)
    segments_after_reg = util.montage(segments_after_reg[::step_size], padding_width=4, fill=None)

    # Plot images
    plt.rc('font', size=6)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(segments_before_reg, cmap='jet', interpolation='nearest', vmin=1, vmax=3)
    ax[1].imshow(segments_after_reg, cmap='jet', interpolation='nearest', vmin=1, vmax=3)

    # Set titles
    fig.suptitle('Segment: ' + segment_key, fontsize=12)
    ax[0].set_title('before registration')
    ax[1].set_title('after registration')
    ax[0].set_xlabel('blue: fixed only, \n green: moving only, \n red: fixed and moving matching')
    ax[1].set_xlabel('blue: fixed only, \n green: transformed moving only, \n red: fixed and transformed moving matching')

    fig.tight_layout()