import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from utils.engine import plot_one
import pandas as pd


# 1. Generate Data
np.random.seed(42)
n_samples = 10000
x1 = np.random.uniform(-3, 3, n_samples)
x2 = np.random.uniform(-3, 3, n_samples)

# Define Signal and Noise separately
signal = np.sin(x1) * np.cos(x2)
noise_std = 0  # You can adjust this to see how SNR changes
noise = np.random.normal(0, noise_std, n_samples)
y = signal + noise

# 2. Calculate SNR
# SNR (power) = Var(Signal) / Var(Noise)
# Convert to dB: 10 * log10(SNR_power)
signal_power = np.var(signal)
noise_power = np.var(noise)
snr_linear = signal_power / noise_power
snr_db = 10 * np.log10(snr_linear)

print(f"Signal Power: {signal_power:.4f}")
print(f"Noise Power: {noise_power:.4f}")
print(f"SNR (Linear): {snr_linear:.4f}")
print(f"SNR (dB): {snr_db:.2f} dB")

# 3. Train Tree Model
X = np.stack([x1, x2], axis=1)
model = GradientBoostingRegressor(n_estimators=1000, max_depth=3, random_state=42)

s1 = pd.Series(x1)
s2 = pd.Series(x2)

# Create lagged versions (e.g., lag of 1 step)
# .dropna() is important because shifting creates NaN values at the beginning
df = pd.DataFrame({
    'x1_lag1': s1.shift(1),
    'x2_lag1': s2.shift(1)
}).dropna()

# Prepare your X and y
# Note: You must also drop the first row of y to match the length of X
X_delayed = df.values
y_delayed = y[1:] 

# Train your model
model.fit(X_delayed, y_delayed)

from sklearn.feature_selection import mutual_info_regression

# Calculate MI scores for each feature
# discrete_features=False assumes all inputs are continuous
mi_scores = mutual_info_regression(X_delayed, y_delayed, random_state=42)

# Display results
for i, score in enumerate(mi_scores):
    print(f"Feature {i+1} Mutual Information: {score:.4f}")

# 4. Measure how much 'noise' the model captured
predictions = model.predict(X)
# The residual is the part the model couldn't capture (mostly noise)
residuals = y - predictions
mse = mean_squared_error(y, predictions)
print(f"Model Mean Squared Error (Uncaptured Noise): {mse:.4f}")

plot_one((y))
