import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    residuals = y_pred - y_true
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    bias = np.sum(residuals) / n
    rmse = np.sqrt(np.sum(residuals**2) / n)
    se = np.sqrt((np.sum(residuals**2) - (np.sum(residuals)**2) / n) / (n - 1)) if n > 1 else np.nan
    return {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'Bias': bias,
        'RMSE': rmse,
        'SE': se
    } 