import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from utils import general, evaluation
from networks import networks
from objectives import ncc, regularizer, penalty

"""Implementation based on the GitHub repository by Jelmer Wolterink (MIAGroupUT):
https://github.com/MIAGroupUT/IDIR """

class ImplicitRegistrator:
    """This class handles the registration of the images.
    The class contains the main framework to train the network, which represents the deformation."""

    def __init__(self, moving_image, moving_dicom, fixed_image, fixed_dicom,
                 slice2D_images, slice2D_dicoms, slices_2D_indices,
                 moving_segments, fixed_segments, segment_keys, **args):
        """Initialize parameters of the learning model."""

        # Set all default arguments in a dict: self.args
        self.set_default_arguments()

        # Check if all kwargs keys are valid (this checks for typos)
        assert all(kwarg in self.args.keys() for kwarg in args)

        # Parse important argument from kwargs
        self.epochs = args["epochs"] if "epochs" in args else self.args["epochs"]
        self.log_interval = (
            args["log_interval"]
            if "log_interval" in args
            else self.args["log_interval"]
        )
        self.lr = args["lr"] if "lr" in args else self.args["lr"]

        self.shared_weights = args["shared_weights"] if "shared_weights" in args else self.args["shared_weights"]

        self.alpha_3D = args["alpha_3D"] if "alpha_3D" in args else self.args["alpha_3D"]

        self.use_rigid = args["use_rigid"] if "use_rigid" in args else self.args["use_rigid"]
        self.alpha_rigid = args["alpha_rigid"] if "alpha_rigid" in args else self.args["alpha_rigid"]
        self.key_mask_rigid = args["key_mask_rigid"] if "key_mask_rigid" in args else self.args["key_mask_rigid"]

        self.use_2D = args["use_2D"] if "use_2D" in args else self.args["use_2D"]
        self.alpha_2D = args["alpha_2D"] if "alpha_2D" in args else self.args["alpha_2D"]

        self.momentum = (
            args["momentum"] if "momentum" in args else self.args["momentum"]
        )
        self.optimizer_arg = (
            args["optimizer"] if "optimizer" in args else self.args["optimizer"]
        )
        self.loss_function_arg_3D = (
            args["loss_function_3D"]
            if "loss_function_3D" in args
            else self.args["loss_function_3D"]
        )
        self.loss_function_arg_2D = (
            args["loss_function_2D"]
            if "loss_function_2D" in args
            else self.args["loss_function_2D"]
        )
        self.n_layers = args["n_layers"] if "n_layers" in args else self.args["n_layers"]
        self.hidden_channels = args["hidden_channels"] if "hidden_channels" in args else self.args["hidden_channels"]
        self.gpu = args["gpu"] if "gpu" in args else self.args["gpu"]
        
        # Parse other arguments from kwargs
        self.log = (
            args["log"] if "log" in args else self.args["log"]
        )

        # Make loss list to save losses
        self.loss_list = []
        self.data_loss_list = []
        self.data_loss_list_3D = []
        self.data_loss_list_2D = []

        # Set seed
        torch.manual_seed(self.args["seed"])

        # Load network
        self.network = networks.ResNet(self.n_layers, self.hidden_channels, self.shared_weights)

        # Choose the optimizer
        if self.optimizer_arg.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.network.parameters(), lr=self.lr, momentum=self.momentum
            )

        elif self.optimizer_arg.lower() == "adam":
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        elif self.optimizer_arg.lower() == "adadelta":
            self.optimizer = optim.Adadelta(self.network.parameters(), lr=self.lr)

        else:
            self.optimizer = optim.SGD(
                self.network.parameters(), lr=self.lr, momentum=self.momentum
            )
            print(
                "WARNING: "
                + str(self.optimizer_arg)
                + " not recognized as optimizer, picked SGD instead"
            )
        
        # Move variables to GPU
        if self.gpu:
            self.network.cuda()

        # Choose the loss function
        if self.loss_function_arg_3D.lower() == "mse":
            self.criterion_3D = nn.MSELoss()

        elif self.loss_function_arg_3D.lower() == "l1":
            self.criterion_3D = nn.L1Loss()

        elif self.loss_function_arg_3D.lower() == "ncc":
            self.criterion_3D = ncc.NCC()

        elif self.loss_function_arg_3D.lower() == "smoothl1":
            self.criterion_3D = nn.SmoothL1Loss(beta=0.2)

        elif self.loss_function_arg_3D.lower() == "huber":
            self.criterion_3D = nn.HuberLoss()

        else:
            self.criterion_3D = nn.MSELoss()
            print(
                "WARNING: "
                + str(self.loss_function_arg_3D)
                + " not recognized as loss function, picked MSE instead"
            )
        if self.loss_function_arg_2D.lower() == "mse":
            self.criterion_2D = nn.MSELoss()

        elif self.loss_function_arg_2D.lower() == "l1":
            self.criterion_2D = nn.L1Loss()

        elif self.loss_function_arg_2D.lower() == "ncc":
            self.criterion_2D = ncc.NCC()

        elif self.loss_function_arg_2D.lower() == "smoothl1":
            self.criterion_2D = nn.SmoothL1Loss(beta=0.2)

        elif self.loss_function_arg_2D.lower() == "huber":
            self.criterion_2D = nn.HuberLoss()

        else:
            self.criterion_2D = nn.MSELoss()
            print(
                "WARNING: "
                + str(self.loss_function_arg_2D)
                + " not recognized as loss function, picked MSE instead"
            )

        self.grad_velocity_regularization = (
            args["grad_velocity_regularization"]
            if "grad_velocity_regularization" in args
            else self.args["grad_velocity_regularization"]
        )
        self.alpha_grad_velocity = (
            args["alpha_grad_velocity"]
            if "alpha_grad_velocity" in args
            else self.args["alpha_grad_velocity"]
        )

        self.method_value_averaging = (
            int(args["method_value_averaging"])
            if "method_value_averaging" in args
            else self.args["method_value_averaging"]
        )

        # Set seed
        torch.manual_seed(self.args["seed"])

        # Parse arguments from kwargs
        self.image_shape = (
            args["image_shape"]
            if "image_shape" in args
            else self.args["image_shape"]
        )
        self.batch_size = (
            int(args["batch_size"]) if "batch_size" in args else self.args["batch_size"]
        )
        self.batch_size_2D = (
            args["batch_size_2D"] if "batch_size_2D" in args else self.args["batch_size_2D"]
        )

        # Initialization
        self.moving_image = moving_image
        self.fixed_image = fixed_image
        self.slice2D_images = slice2D_images
        self.moving_dicom = moving_dicom
        self.fixed_dicom = fixed_dicom
        self.slice2D_dicoms = slice2D_dicoms
        self.slice2D_indices = slices_2D_indices
        self.moving_segments = moving_segments
        self.fixed_segments = fixed_segments
        self.segment_keys = segment_keys

        self.device = "cpu"

        self.possible_coordinate_tensor = general.make_coordinate_tensor(
            dims=self.fixed_image[0].shape, gpu=self.gpu
        )

        if self.use_2D:
            self.possible_coordinate_tensor_slices_3D_frontal = []
            self.possible_coordinate_tensor_slices_3D_sagital = []
            for i in range(len(slice2D_images)):
                match slice2D_dicoms[i][0].ImageOrientationPatient:
                    case [1, 0, 0, 0, 0, -1]: # frontal
                        if len(self.possible_coordinate_tensor_slices_3D_frontal)==0:
                            (self.possible_coordinate_tensor_slices_3D_frontal,
                             self.possible_coordinate_tensor_slices_2D_frontal,_) = (
                                general.make_coordinate_tensors_slice(
                                    fixed_dicom[0], slice2D_dicoms[i][0], dims_orig=slice2D_images[i][0].shape, gpu=self.gpu
                                ))
                    case [0, 1, 0, 0, 0, -1]: # sagital
                        if len(self.possible_coordinate_tensor_slices_3D_sagital)==0:
                            (self.possible_coordinate_tensor_slices_3D_sagital,
                             self.possible_coordinate_tensor_slices_2D_sagital,_) = (
                                general.make_coordinate_tensors_slice(
                                    fixed_dicom[0], slice2D_dicoms[i][0], dims_orig=slice2D_images[i][0].shape, gpu=self.gpu
                                ))


        if self.gpu:
            self.moving_image = self.moving_image.cuda()
            self.fixed_image = self.fixed_image.cuda()
            self.slice2D_images = [images.cuda() for images in self.slice2D_images]

            for key in self.segment_keys:
                self.moving_segments[key] = self.moving_segments[key].cuda()
                self.fixed_segments[key] = self.fixed_segments[key].cuda()

            self.device = "cuda"




    def set_default_arguments(self):
        """Set default arguments."""

        # Inherit default arguments from standard learning model
        self.args = {}

        # Define the value of arguments
        self.args["lr"] = 1E-3
        self.args["shared_weights"] = False  # True: velocity field constant in time
        self.args["alpha_3D"] = [1]
        self.args["use_rigid"] = True
        self.args["alpha_rigid"] = 1E-2
        self.args["key_mask_rigid"] = None
        self.args["use_2D"] = True
        self.args["alpha_2D"] = 3E-3
        self.args["batch_size"] = int(1E5)
        self.args["batch_size_2D"] = [int(1E3), 15]
        self.args["n_layers"] = 10
        self.args["hidden_channels"] = 100

        self.args["grad_velocity_regularization"] = True
        self.args["alpha_grad_velocity"] = 1E-1

        self.args["image_shape"] = (200, 200)

        self.args["network"] = None

        self.args["epochs"] = 1E4
        self.args["log_interval"] = 10
        self.args["log"] = True

        self.args["network_type"] = "ResNet"

        self.args["optimizer"] = "Adam"
        self.args["loss_function_3D"] = "mse"
        self.args["loss_function_2D"] = "ncc"
        self.args["momentum"] = 0.5                     # parameter of optimizer 'SGD'
        self.args["method_value_averaging"] = "middle"
        self.args["method_value_averaging"] = "middle"

        self.args["seed"] = 3
        self.args["gpu"] = torch.cuda.is_available()

    def training_iteration(self, epoch):
        """Perform one iteration of training."""

        # Reset the gradient
        self.network.train()

        loss = 0
        loss_2D = 0
        loss_3D = 0

        # 3D loss
        indices = torch.randperm(
            self.possible_coordinate_tensor.shape[0], device=self.device
        )[: self.batch_size]
        coordinate_tensor = self.possible_coordinate_tensor[indices, :]
        coordinate_tensor = coordinate_tensor.requires_grad_(True)

        output, velocity = self.network(coordinate_tensor, get_velocity=True)

        for i in range(len(self.moving_image)):
            transformed_image = general.fast_trilinear_interpolation(
                    self.moving_image[i],
                    output[:, 0],
                    output[:, 1],
                    output[:, 2])
            fixed_image = general.fast_trilinear_interpolation(
                    self.fixed_image[i],
                    coordinate_tensor[:, 0],
                    coordinate_tensor[:, 1],
                    coordinate_tensor[:, 2])
            loss_3D += self.alpha_3D[i] * self.criterion_3D(transformed_image, fixed_image)


        # 2D loss
        if self.use_2D:
            # choose frontal and sagittal coordinates
            indices_frontal = torch.randperm(
                self.possible_coordinate_tensor_slices_2D_frontal.shape[0], device=self.device
            )[: self.batch_size_2D[0]]
            coordinate_tensor_2D_frontal = self.possible_coordinate_tensor_slices_2D_frontal[indices_frontal, :]
            coordinate_tensor_3D_frontal = self.possible_coordinate_tensor_slices_3D_frontal[:, indices_frontal, :]
            coordinate_tensor_3D_frontal = coordinate_tensor_3D_frontal.requires_grad_(True)

            indices_sagital = torch.randperm(
                self.possible_coordinate_tensor_slices_2D_sagital.shape[0], device=self.device
            )[: self.batch_size_2D[0]]
            coordinate_tensor_2D_sagittal = self.possible_coordinate_tensor_slices_2D_sagital[indices_sagital, :]
            coordinate_tensor_3D_sagittal = self.possible_coordinate_tensor_slices_3D_sagital[:, indices_sagital, :]
            coordinate_tensor_3D_sagittal = coordinate_tensor_3D_sagittal.requires_grad_(True)

            indices = torch.randperm(
                self.slice2D_indices.shape[0]
            )[: self.batch_size_2D[1]]
            slice2D_indices = self.slice2D_indices[indices, :]

            for i in range(self.batch_size_2D[1]):
                loss_2D += evaluation.compute_error_2D(
                    self, slice2D_indices[i], criterion=self.criterion_2D,
                    coordinate_tensor_2D_frontal=coordinate_tensor_2D_frontal, coordinate_tensor_3D_frontal=coordinate_tensor_3D_frontal,
                    coordinate_tensor_2D_sagital=coordinate_tensor_2D_sagittal, coordinate_tensor_3D_sagital=coordinate_tensor_3D_sagittal
                )

            loss_2D = self.alpha_2D / self.batch_size_2D[1] * loss_2D


        # Compute the loss
        loss += loss_2D + loss_3D

        # Store the values of the data loss
        if self.log:
            if epoch % self.log_interval == 0:
                self.data_loss_list.append(1/self.log_interval * loss.detach().cpu().numpy())
                self.data_loss_list_3D.append(1/self.log_interval * loss_3D.detach().cpu().numpy())
                if self.use_2D: self.data_loss_list_2D.append(1/self.log_interval * loss_2D.detach().cpu().numpy())
            else:
                self.data_loss_list[epoch//self.log_interval] += 1/self.log_interval * loss.detach().cpu().numpy()
                self.data_loss_list_3D[epoch//self.log_interval] += 1/self.log_interval * loss_3D.detach().cpu().numpy()
                if self.use_2D: self.data_loss_list_2D[epoch//self.log_interval] += 1/self.log_interval * loss_2D.detach().cpu().numpy()

        # Penalty ensuring local rigidity on rigidity mask
        if self.use_rigid:
            mask_rigid = self.fixed_segments[self.key_mask_rigid]
            mask = general.fast_trilinear_interpolation(
                mask_rigid,
                coordinate_tensor[:, 0],
                coordinate_tensor[:, 1],
                coordinate_tensor[:, 2],
            ).round()
            coordinate_tensor_mask = coordinate_tensor[mask == 1]
            coordinate_tensor_mask = coordinate_tensor_mask.requires_grad_(True)
            output_mask = self.network(coordinate_tensor_mask)

            loss += self.alpha_rigid * penalty.compute_rigidity_loss(
                coordinate_tensor_mask, output_mask, batch_size=coordinate_tensor_mask.shape[0]
            )

        # Regularization
        if self.grad_velocity_regularization:
            loss += self.alpha_grad_velocity * regularizer.compute_grad_velocity_norm(
                coordinate_tensor, velocity, batch_size=self.batch_size
            )


        # Perform the backpropagation and update the parameters accordingly
        for param in self.network.parameters():
            param.grad = None
        loss.backward()
        self.optimizer.step()

        # Store the value of the total loss
        if self.log:
            if epoch % self.log_interval == 0:
                self.loss_list.append(1/self.log_interval * loss.detach().cpu().numpy())
            else:
                self.loss_list[epoch//self.log_interval] += 1/self.log_interval * loss.detach().cpu().numpy()


    def fit(self, epochs=None):
        """Train the network."""

        # Determine epochs
        if epochs is None:
            epochs = self.epochs

        # Set seed
        torch.manual_seed(self.args["seed"])

        # Perform training iterations
        for i in tqdm.tqdm(range(epochs)):
            self.training_iteration(i)
