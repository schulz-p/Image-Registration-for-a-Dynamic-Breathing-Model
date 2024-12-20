import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from utils import evaluation, loadData
from models import models
from visualization import visualization


'''---------------------------- Initialize paths ----------------------------'''
if torch.cuda.is_available():
    data_dir = r'/data'
    result_dir = r'/results'
else:
    dir = os.getcwd()
    data_dir = dir + r'\data'
    result_dir = dir + r'\results'

result_network_path = result_dir + r"/networks.pth"


'''---------------------------- Load data ----------------------------'''
# ToDo: Implement load_data and get_2D_indices
(
    img_insp_dicom, img_insp,
    img_exp_dicom, img_exp,
    slices_2D_dicom, slices_2D,
    segm_insp, segm_exp, segm_keys
) = loadData.load_data()

slices_2D_indices = loadData.get_2D_indices()


'''---------------------------- Set parameters ----------------------------'''
# ToDo: specify weightings of the 3D images that should be used
indices_weightings = [0, 1, 2, 3]
num_weightings = len(indices_weightings)
# ToDo: specify number of time steps used
num_timesteps = 6

args = {}
args["image_shape"] = img_exp[0].shape
args["alpha_3D"] = [1 / num_weightings] * num_weightings
args["n_layers"] = num_timesteps - 1
args["key_mask_rigid"] = "Rib_Cage"

learn = True


'''---------------------------- Learn or load network ----------------------------'''

# Initialize object handling learning
ImpReg = models.ImplicitRegistrator(img_exp, img_exp_dicom, img_insp, img_insp_dicom,
                                    slices_2D, slices_2D_dicom, slices_2D_indices,
                                    segm_exp, segm_insp, segm_keys, **args)


# Learn and save network or load network
if learn:
    ImpReg.fit()
    torch.save(ImpReg.network.state_dict(), result_network_path)
else:
    ImpReg.network.load_state_dict(torch.load(result_network_path, map_location=torch.device('cpu')))


'''---------------------------- Show results ----------------------------'''

# Visualize loss over epochs
if learn:
    plt.plot([i*ImpReg.log_interval for i in range(len(ImpReg.loss_list))], ImpReg.loss_list, label='loss')
    plt.plot([i*ImpReg.log_interval for i in range(len(ImpReg.data_loss_list))], ImpReg.data_loss_list, label='data loss')
    plt.plot([i*ImpReg.log_interval for i in range(len(ImpReg.data_loss_list_3D))], ImpReg.data_loss_list_3D, label='data loss in 3D')
    plt.plot([i*ImpReg.log_interval for i in range(len(ImpReg.data_loss_list_2D))], ImpReg.data_loss_list_2D, label='data loss in 2D')
    plt.legend(loc='best')
    plt.title("Loss vs epochs")
    plt.show()

# Visualize transformation of moving image over time
visualization.visualize_moving_image_over_time(ImpReg, images_3D_index=0, dim=0)
plt.show()

# Visualize fixed image, transformed moving image and difference images
visualization.visualize_final_images(ImpReg, images_3D_index=0, resolution=[25, 64, 50], dim=0)
plt.show()

# MSE
MSE = evaluation.compute_error(ImpReg, num_input_coords=np.prod(ImpReg.image_shape))
print("MSE in 3D: ", round(torch.mean(MSE).item(), 5), ",     MSE details: ", MSE.cpu().numpy())

# Compute and visualize dice of segments
print("Dice:")
for key in ImpReg.segment_keys:
    dice = evaluation.compute_dice(ImpReg, key, num_input_coords=np.prod(ImpReg.image_shape)) #5E6)
    print("    " + key + ":  before: ", round(dice[0].item(), 4), "   after: ", round(dice[1].item(), 4))
    visualization.visualize_segments(ImpReg, key, resolution=[128,320,250])
    plt.show()

# Compute 2D NCC
NCC_2D = 0
for i in range(slices_2D_indices.shape[0]):
    NCC_2D += evaluation.compute_error_2D(ImpReg, slices_2D_indices[i], criterion=ImpReg.criterion_2D)
NCC_2D = NCC_2D / slices_2D_indices.shape[0]
print('NCC in 2D:', round(NCC_2D.item(), 4))

# Compute percentage of convolutions considering num_input_coords points
ratio_conv = evaluation.compute_ratio_convolutions(ImpReg, num_input_coords=1E5)
print('convolution ratio: ', ratio_conv)