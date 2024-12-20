import torch
from utils import general


def compute_grad_velocity_norm(input_coords, velocity, mask=None, batch_size=None):
    """Compute sum of squared Frobenius-norms of the gradients of the velocities."""

    if mask is None:
        mask = 1
    else:
        mask = general.fast_trilinear_interpolation(
            mask,
            input_coords[:, 0],
            input_coords[:, 1],
            input_coords[:, 2],
        ).round().cpu()

    loss = 0
    for v in velocity:
        jac = compute_jacobian_matrix(input_coords, v)
        loss += torch.sum(
            mask * torch.square(torch.linalg.matrix_norm(jac, dim=(1,2)))
        )

    return loss / (batch_size*len(velocity))


def compute_jacobian_matrix(input_coords, output, with_identity=True):
    """Compute the Jacobian matrix of the output wrt the input.
    (Implementation from the GitHub repository by Jelmer Wolterink (MIAGroupUT):
    https://github.com/MIAGroupUT/IDIR)"""

    jacobian_matrix = torch.zeros(input_coords.shape[0], 3, 3)
    for i in range(3):
        jacobian_matrix[:, i, :] = gradient(input_coords, output[:, i])
        if not with_identity:
            jacobian_matrix[:, i, i] -= torch.ones_like(jacobian_matrix[:, i, i])
    return jacobian_matrix


def gradient(input_coords, output, grad_outputs=None):
    """Compute the gradient of the output wrt the input.
    (Implementation from the GitHub repository by Jelmer Wolterink (MIAGroupUT):
    https://github.com/MIAGroupUT/IDIR)"""

    grad_outputs = torch.ones_like(output)
    grad = torch.autograd.grad(
        output, [input_coords], grad_outputs=grad_outputs, create_graph=True, allow_unused=True
    )[0]
    return grad
