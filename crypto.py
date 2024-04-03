import yfinance as yf
import numpy as np
from scipy.linalg import sqrtm

# List of major cryptocurrencies
major_crypto_symbols = ['BTC-EUR', 'ETH-EUR', 'LTC-EUR', 'XRP-EUR']

# Fetch hourly data for each crypto
hourly_data = {}
min_length = np.inf
for symbol in major_crypto_symbols:
    data = yf.download(symbol, interval='1h', period='2y')
    hourly_data[symbol] = data['Close'].values
    min_length = min(min_length, len(hourly_data[symbol]))

# Make sure arrays are of same length
for symbol in major_crypto_symbols:
        hourly_data[symbol] = hourly_data[symbol][:min_length]

# Rescale features by setting mean to zero and obtain asset matrix S_t
S = np.vstack(list(hourly_data.values()))
mean = np.nanmean(S, axis=0)
S -= mean

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




