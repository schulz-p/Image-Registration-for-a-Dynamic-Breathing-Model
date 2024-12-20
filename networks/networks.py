import torch
from torch import nn


class ResBlock(nn.Module):
    """ This class contains a residual block of the network."""

    def __init__(self, in_out_channels, hidden_channels):
        """Initialize the residual block."""

        super().__init__()

        self.linear1 = nn.Linear(in_out_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, hidden_channels)
        self.linear3 = nn.Linear(hidden_channels, in_out_channels)

        # Weight initialization
        std1 = 1E0
        std2 = 1E-3
        torch.nn.init.normal_(self.linear1.weight, std=std1)
        torch.nn.init.normal_(self.linear1.bias, std=std1)
        torch.nn.init.normal_(self.linear2.weight, std=std1)
        torch.nn.init.normal_(self.linear2.bias, std=std1)
        torch.nn.init.normal_(self.linear3.weight, std=std2)
        torch.nn.init.normal_(self.linear3.bias, std=std2)


    def forward(self, input):
        """The forward function of the residual block."""

        velocity = self.linear1(input)
        velocity = torch.sin(velocity)
        velocity = self.linear2(velocity)
        velocity = torch.sin(velocity)
        velocity = self.linear3(velocity)
        
        output = input + velocity 

        output = torch.reshape(output,(input.size(dim=0),input.size(dim=1)))
        velocity = torch.reshape(velocity,(input.size(dim=0),input.size(dim=1)))

        return output, velocity


class ResNet(nn.Module):
    """ This class contains the ResNet network."""

    def __init__(self, n_layers, hidden_channels, shared_weights):
        """Initialize the network."""

        super(ResNet, self).__init__()
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        self.shared_weights = shared_weights

        block = ResBlock(3, hidden_channels)

        # Make the layers
        self.layers = []
        for i in range(self.n_layers):
            if self.shared_weights:
                self.layers.append(block)
            else:
                self.layers.append(ResBlock(3, hidden_channels))

        # Combine all layers to one model
        self.layers = nn.Sequential(*self.layers)


    def forward(self, x, steps='all', get_velocity=False):
        """The forward function of the network."""

        if steps=='all':
            steps = self.n_layers

        if get_velocity:
            velocity = []
        # Propagate through layers
        for j in range(steps):
            x, v = self.layers[j](x)
            if get_velocity:
                velocity.append(v)

        # Return the output
        if get_velocity:
            return x, velocity
        
        return x
