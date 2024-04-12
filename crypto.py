import yfinance as yf
import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from utils import compute_weights, bring_back_mean
matplotlib.use('TkAgg')
sns.set(style="whitegrid", context="notebook")

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
S_original = np.vstack(list(hourly_data.values()))
mean = np.nanmean(S_original, axis=1)
S = S_original - mean[:, np.newaxis]

x, A, C, C_sqrt_inv, eig_values, eig_vectors = compute_weights(S)

# First eigenvalue is smallest, second eigenvalue is largest
z_mr = eig_vectors[:, 0]  # Mean reversion vector
z_mom = eig_vectors[:, 1]  # Momentum vector

# Compute weights
x_mr = np.array([C_sqrt_inv @ z_mr]).T  # Weights if goal is optimal mean reversion
x_mom = np.array([C_sqrt_inv @ z_mom]).T  # Weights if goal is optimal momentum

# Normalize weights
x_mr /= sum(abs(x_mr[:, 0]))
x_mom /= sum(abs(x_mom[:, 0]))

# Construct the portfolio (still having 0 mean)
P_mr = (x_mr.T @ S)[0]
P_mom = (x_mom.T @ S)[0]

# Re-mean our assets
P_mr_value = bring_back_mean(S, S_original, x_mr)
P_mom_value = bring_back_mean(S, S_original, x_mom)

# Create figure with four subplots
fig, axes = plt.subplots(2, 2)
time_axis = data.T.T.index[:min_length]

axes[0, 0].plot(time_axis, P_mr, color='navy')
axes[0, 0].set_title('Optimal Mean-Reverting Portfolio (Demeaned)', fontweight='bold')

axes[0, 1].plot(time_axis, P_mom, color='navy')
axes[0, 1].set_title('Optimal Predictable Portfolio (Demeaned)', fontweight='bold')

axes[1, 0].plot(time_axis, P_mr_value, color='navy')
axes[1, 0].set_title('Optimal Mean-Reverting Portfolio', fontweight='bold')

axes[1, 1].plot(time_axis, P_mom_value, color='navy')
axes[1, 1].set_title('Optimal Predictable Portfolio', fontweight='bold')

for ax in axes.flatten():
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', va='top')
    ax.set_ylabel('Portfolio Value')

plt.suptitle('Example Crypto Portfolio', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()






