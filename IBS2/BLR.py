import numpy as np
import pandas as pd
import math

# Print Array helper
def printarray(array):
    for i in array:
        print(i,end=" ")
    print()

# Loading Dataset
X = pd.read_csv("BBA0171_pairwise_features.csv")
Col = X.columns
print()
print("Total No.of Columns: ",len(Col))

# Removing Non-Numeric Values (Not required)
X = X.select_dtypes(include='number')
Removed_Col = []
for i in Col:
    if i not in X.columns:
        Removed_Col.append(i)
print("Removed : ",Removed_Col)

# Target Variable
Y = X['alignment_coverage']

X = X.drop(columns=['alignment_coverage'])
X = X.drop(columns=['nw_score','sw_score'])

# Normalizing Data (To reduce range of values)
meanX = X.mean()
stdX = X.std()
X = (X-meanX)/stdX

print("After Standardization:",len(X.columns))
# Standardized NaN Reduction Raw BLR (Where Standard Deviation of the Feature = 0)
X = X.drop(columns=X.columns[(X.isna().any())].to_list())

print("After NaN Column Removal:",len(X.columns))

# Y is also ready => Next Step is Bayesian Linear Regression
feature_names = X.columns.to_list()
X = X.to_numpy()
Y = Y.to_numpy()

# Add Bias Term
feature_names = ["bias"] + feature_names
N = X.shape[0]
X = np.hstack([np.ones((N, 1)), X])  # prepend 1s column

# Train / Val / Test Split
idx = np.random.permutation(N)

train_end = int(0.7 * N) # Training Ending index
val_end   = int(0.85 * N) # Validation Ending index

train_idx = idx[:train_end] # Training Index
val_idx   = idx[train_end:val_end] # Validation Index
test_idx  = idx[val_end:] # Test Index

X_train, Y_train = X[train_idx], Y[train_idx] # X and Y Train
X_val,   Y_val   = X[val_idx],   Y[val_idx] # X and Y Validation
X_test,  Y_test  = X[test_idx],  Y[test_idx] # X and Y Test

Y_train = Y_train.reshape(-1, 1)
Y_val   = Y_val.reshape(-1, 1)
Y_test  = Y_test.reshape(-1, 1)

# HyperParameter Tuning
def blr_evidence_maximization(X, y, alpha=1.0, beta=1.0, max_iter=100, tol=1e-6):
    N, D = X.shape

    for it in range(max_iter):
        # Posterior
        S_N_inv = alpha * np.eye(D) + beta * (X.T @ X)
        S_N = np.linalg.inv(S_N_inv)
        m_N = beta * (S_N @ X.T @ y)

        # Effective number of parameters
        eigvals = np.linalg.eigvalsh(beta * (X.T @ X))
        gamma = np.sum(eigvals / (alpha + eigvals))

        # Update alpha and beta
        alpha_new = gamma / (m_N.T @ m_N + 1e-12)
        residual = y - X @ m_N
        beta_new = (N - gamma) / (residual.T @ residual + 1e-12)

        if abs(alpha - alpha_new) < tol and abs(beta - beta_new) < tol:
            break

        alpha, beta = alpha_new.item(), beta_new.item()

    return m_N, S_N, alpha, beta

# Evidence Maximization BLR
m_N, S_N, alpha, beta = blr_evidence_maximization(X_train, Y_train, alpha=1.0, beta=1.0)

print("\nLearned alpha:", alpha)
print("Learned beta :", beta)

# Bayesian Linear Regression
alpha = alpha   # Prior precision
beta  = beta   # Noise precision (1 / sigma^2)

D = X_train.shape[1]
I = np.eye(D)

# Posterior covariance
S_N_inv = alpha * I + beta * (X_train.T @ X_train)
S_N = np.linalg.inv(S_N_inv)

# Posterior mean
m_N = beta * (S_N @ X_train.T @ Y_train)

# Prediction with Uncertainty
def blr_predict(X_new, m_N, S_N, beta):
    mean = X_new @ m_N
    var  = (1 / beta) + np.sum(X_new @ S_N * X_new, axis=1, keepdims=True)
    std  = np.sqrt(var)
    return mean, std

mean_pred, std_pred = blr_predict(X_test, m_N, S_N, beta)

# Evaluation
rmse = np.sqrt(np.mean((mean_pred - Y_test) ** 2))
mae  = np.mean(np.abs(mean_pred - Y_test))

print("\nBLR Test RMSE:", rmse)
print("BLR Test MAE :", mae)

# Show few predictions
print("\nSample Predictions (Mean ± Uncertainty):")
for i in range(min(5, len(Y_test))):
    print(f"True: {Y_test[i][0]:.4f} | Pred: {mean_pred[i][0]:.4f} ± {std_pred[i][0]:.4f}")

importance = np.abs(m_N).flatten()
sorted_idx = np.argsort(importance)[::-1]

print("\nTop 10 Important Features (Posterior Mean Magnitude):")
for i in sorted_idx[:10]:
    i = int(i)
    print(f"{feature_names[i]} : {m_N[i][0]:.5f}")

# 95% Credible Intervals per prediction
lower = mean_pred - 1.96 * std_pred
upper = mean_pred + 1.96 * std_pred

print("\n95% Credible Intervals (first 5):")
for i in range(5):
    print(f"True: {Y_test[i][0]:.4f} | [{lower[i][0]:.4f}, {upper[i][0]:.4f}]")

# Calibration Check
within = np.mean((Y_test >= lower) & (Y_test <= upper)).item()
print("\nFraction of true values inside 95% CI:", within)

