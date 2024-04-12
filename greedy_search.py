import numpy as np
from itertools import combinations
import yfinance as yf
from scipy.linalg import sqrtm

best_assets = [None, None]
best_pred = np.inf
best_x = None

# Define a list of tickers or symbols for the assets you're interested in
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NFLX', 'NVDA', 'AMD', 'INTC', 'CSCO', 'ADBE', 'PYPL', 'CRM',
           'IBM', 'ORCL']

# Generate all combinations of two assets
assets = yf.download(tickers, start='2022-01-01', end='2024-01-01')['Adj Close']
asset_combinations = combinations(assets, 2)

for asset_pair in asset_combinations:
    asset1, asset2 = asset_pair

    # Extract the adjusted closing prices for the two assets
    cols = [asset1, asset2]
    prices = assets[cols]

    S = np.array(prices).T

    A = np.linalg.inv(S[:, :-1] @ S[:, :-1].T) @ (S[:, :-1] @ S[:, 1:].T)

    C = np.cov(S)
    C_sqrt = sqrtm(C)
    C_sqrt_inv = np.linalg.inv(C_sqrt)

    B = C_sqrt_inv @ A.T @ C @ A @ C_sqrt_inv

    eig = np.linalg.eig(B)
    eig_values, eig_vectors = eig[0], eig[1]

    index = np.argmin(eig_values)
    z = eig_vectors[:, index]

    x = np.array([C_sqrt_inv @ z]).T
    x /= sum(abs(x[:, 0]))

    if max(abs(x)) <= 0.8:
        pred = (x.T @ A.T @ C @ A @ x) / (x.T @ C @ x)

        if pred < best_pred:
            best_pred = pred
            best_assets = [asset1, asset2]
            best_x = x

print('Best assets: ', best_assets)
print('Portfolio weights: ', best_x)
print('Predictability: ', best_pred)

