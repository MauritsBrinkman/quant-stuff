"""Module that contains utility functions."""

import numpy as np
from scipy.linalg import sqrtm

def compute_weights(S):
    """Function that outputs portfolio weights, covariance, predictability, eigenvalues and eigenvectors.

    Keyword arguments:
    - S: ndarray containing asset prices, where column dimension is determined by number of time steps and row
              dimension is the number of assets.
    """

    # Check if function input is correct
    num_rows, num_cols = S.shape
    assert num_cols > num_rows, ("You probably made a mistake, as the number of columns is not longer than the number of"
                                 "rows.")

    # Fit A matrix using least squares (note that '@' is equivalent to using np.matmul)
    A = np.linalg.inv(S[:, :-1] @ S[:, :-1].T) @ (S[:, :-1] @ S[:, 1:].T)

    # Compute covariance matrix and its Cholesky decomposition
    C = np.cov(S)
    C_sqrt = sqrtm(C)

    # Compute B matrix for optimization problem
    C_sqrt_inv = np.linalg.inv(C_sqrt)
    B = C_sqrt_inv @ A.T @ C @ A @ C_sqrt_inv

    # Get eigenvalues and eigenvectors of B
    eig = np.linalg.eig(B)
    eig_values, eig_vectors = eig[0], eig[1]

    index = np.argmin(eig_values)
    z = eig_vectors[:, index]

    # Determine portfolio weights
    x = np.array([C_sqrt_inv @ z]).T
    x /= sum(abs(x[:, 0]))

    return x, A, C, C_sqrt_inv, eig_values, eig_vectors
