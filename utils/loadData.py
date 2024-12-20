def load_data():
    """ Load 3D data, 2D data and 3D segmentations:
    - img_insp_dicom is a 2D ndarray of shape (num_weightings, x_size_3D_images)
      containing dicom objects for each 2D yz-slice of the different weightings of the inspiratory image
    - img_insp is a 4D tensor of shape (num_weightings, xyz_size_3D_images)
      containing 3D images representing different weightings of the inspiratory image
    - img_exp_dicom is a 2D ndarray of shape (num_weightings, x_size_3D_images)
      containing dicom objects for each 2D yz-slice of the different weightings of the expiratory image
    - img_exp is a 4D tensor of shape (num_weightings, xyz_size_3D_images)
      containing 3D images representing different weightings of the expiratory image
    - slices_2D_dicom is a 2D ndarray of shape (num_slice_positions, num_slices_per_position)
      containing dicom objects for each 2D image
    - slices_2D is a list of length num_slice_positions
      containing tensors of shape (num_slices_per_position, xy_size_2D_images) with the 2D images
    - segm_insp is a dictionary
      containing 3D segmentation masks of shape xyz_size_3D_images of the inspiratory image
    - segm_exp is a dictionary
      containing 3D segmentation masks of shape xyz_size_3D_images of the expiratory image
    - segm_keys is a 1D ndarray of shape num_segmentation_keys
      containing the names of the segmentations used as keys in the segmentation masks. """

    # ToDo
    img_insp_dicom = None
    img_insp = None
    img_exp_dicom = None
    img_exp = None
    slices_2D_dicom = None
    slices_2D = None
    segm_insp = None
    segm_exp = None
    segm_keys = None

    if img_insp == None:
        raise Exception('Implementation of load_data is missing')

    return (
        img_insp_dicom, img_insp,
        img_exp_dicom, img_exp,
        slices_2D_dicom, slices_2D,
        segm_insp, segm_exp, segm_keys,
    )


def get_2D_indices():
    """ Calculate an array of indices containing information about used 2D images:
    - indices_array is a 2D ndarray of shape (num_slices, 3)
      containing for each 2D slice an entry of the form [index_slice_position, time_step, index_slice], where
        - index_slice_position is the index slice position
          (between 0 and num_slice_positions-1)
        - time_step is the time step the slice is assigned to
        - index_slice is the index of the slice at the specified position
          (between 0 and num_slices_per_position-1)"""

    # ToDo
    indices_array = None

    if indices_array == None:
        raise Exception('Implementation of get_2D_indices is missing')

    return indices_array