import torch
from objectives import regularizer


def compute_bending_energy(input_coords, output, batch_size=None):
    """Compute the bending energy."""

    jacobian_matrix = regularizer.compute_jacobian_matrix(input_coords, output)

    dx_xyz = torch.zeros(input_coords.shape[0], 3, 3)
    dy_xyz = torch.zeros(input_coords.shape[0], 3, 3)
    dz_xyz = torch.zeros(input_coords.shape[0], 3, 3)
    for i in range(3):
        dx_xyz[:, i, :] = regularizer.gradient(input_coords, jacobian_matrix[:, i, 0])
        dy_xyz[:, i, :] = regularizer.gradient(input_coords, jacobian_matrix[:, i, 1])
        dz_xyz[:, i, :] = regularizer.gradient(input_coords, jacobian_matrix[:, i, 2])

    dx_xyz = torch.square(dx_xyz)
    dy_xyz = torch.square(dy_xyz)
    dz_xyz = torch.square(dz_xyz)

    loss = (
        torch.mean(dx_xyz[:, :, 0])
        + torch.mean(dy_xyz[:, :, 1])
        + torch.mean(dz_xyz[:, :, 2])
    )
    loss += (
        2 * torch.mean(dx_xyz[:, :, 1])
        + 2 * torch.mean(dx_xyz[:, :, 2])
        + 2 * torch.mean(dy_xyz[:, :, 2])
    )

    return loss / batch_size


def compute_jacobian_loss(input_coords, output, batch_size=None):
    """Compute the jacobian regularization loss."""

    # Compute Jacobian matrices
    jac = regularizer.compute_jacobian_matrix(input_coords, output)

    # Compute determinants and take norm
    loss = torch.det(jac) - 1
    loss = torch.linalg.norm(loss, 1)

    return loss / batch_size


def compute_orthogonality_loss(input_coords, output, batch_size=None):
    """Compute orthogonality regularization loss."""

    # Compute Jacobian matrices and its transpose
    jac = regularizer.compute_jacobian_matrix(input_coords, output)
    jac_T = torch.transpose(jac, 1, 2)

    loss = jac_T * jac - torch.eye(3)
    loss = torch.sum(torch.square(torch.linalg.matrix_norm(loss, dim=(1,2))))

    return loss / batch_size

def compute_rigidity_loss(input_coords, output, batch_size=None):
    """Compute the rigidity penalty."""

    loss = (compute_bending_energy(input_coords, output, batch_size=batch_size) +
            compute_orthogonality_loss(input_coords, output, batch_size=batch_size) +
            1E-1 * compute_jacobian_loss(input_coords, output, batch_size=batch_size))

    return loss