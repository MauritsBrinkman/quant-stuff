import numpy as np
from itertools import combinations
import yfinance as yf
from utils import compute_weights
from scipy.linalg import sqrtm


def greedy_search(tickers, k, a=5):
    best_assets = [None, None]
    best_pred = np.inf
    best_x = None

    # Generate all combinations of two assets
    assets = yf.download(tickers, start='2022-01-01', end='2024-01-01')['Adj Close']
    asset_combinations = combinations(assets, 2)

    for i in range(2, k+1):
        if i == 2:
            # Find the best two assets
            for asset_pair in asset_combinations:
                asset1, asset2 = asset_pair

                # Extract the adjusted closing prices for the two assets
                cols = [asset1, asset2]
                prices = assets[cols]

                S = np.array(prices).T
                x, A, C, C_sqrt_inv, eig_values, eig_vectors = compute_weights(S)

                if max(abs(x)) <= 0.8:
                    pred = (x.T @ A.T @ C @ A @ x) / (x.T @ C @ x)

                    if pred < best_pred:
                        best_pred = pred
                        best_assets = [asset1, asset2]
                        best_x = x
        else:
            for j in range(len(assets.columns)):
                asset = assets.columns[j]
                if asset not in best_assets:
                    prev_assets = np.array(assets[best_assets[:i]]).T
                    current_asset = np.array(assets[asset]).T

                    S = np.vstack((prev_assets, current_asset))
                    x, A, C, C_sqrt_inv, eig_values, eig_vectors = compute_weights(S)

                    if max(abs(x)) <= 0.8 and min(abs(x)) > 1 / (a * k):
                        pred = (x.T @ A.T @ C @ A @ x) / (x.T @ C @ x)

                        if pred < best_pred:
                            best_pred = pred
                            best_assets = best_assets[:i] + [asset]
                            best_x = x

    return best_assets, best_x, best_pred


# Define a list of tickers or symbols for the assets you're interested in
ticker_names = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NFLX', 'NVDA', 'AMD', 'INTC', 'CSCO', 'ADBE', 'PYPL', 'CRM',
                'IBM', 'ORCL']

# Set number of assets for sparse portfolio and minimum portfolio weight
k, a = 8, 4

best_assets, best_x, predictive_value = greedy_search(tickers=ticker_names, k=k, a=a)
print('Best assets: ', best_assets)
print('Portfolio weights: ', best_x)
print('Predictability: ', predictive_value)
