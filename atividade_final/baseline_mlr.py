import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 1. Caminhos dos arquivos ---
BASE_DIR = Path(__file__).parent
calib_csv = BASE_DIR / 'data' / 'csv' / 'all_data_matlab.csv'
idrc_csv  = BASE_DIR / 'data' / 'csv' / 'all_data_IDRC.csv'

# --- 2. Carregar dados ---
df = pd.read_csv(calib_csv)
idrc_df = pd.read_csv(idrc_csv)

# --- 3. Preparar datasets ---
# Treinamento (Calibration)
mask_calib = df['targetCalibration'].notna()
X_train = pd.DataFrame({
    'wl':    df.loc[mask_calib, 'wl'],
    'input': df.loc[mask_calib, 'inputCalibration']
})
y_train = df.loc[mask_calib, 'targetCalibration']

# Teste
mask_test = df['targetTest'].notna()
X_test = pd.DataFrame({
    'wl':    df.loc[mask_test, 'wl'],
    'input': df.loc[mask_test, 'inputTest']
})
y_test = df.loc[mask_test, 'targetTest']

# Validação: utiliza inputValidation e valores de referência do IDRC
mask_val = df['inputValidation'].notna()
X_val = pd.DataFrame({
    'wl':    df.loc[mask_val, 'wl'],
    'input': df.loc[mask_val, 'inputValidation']
})
y_val = idrc_df['Value (Reference values)']

# --- 4. Treinar modelo ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- 5. Previsões e métricas ---
def compute_metrics(y_true, y_pred):
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2':  r2_score(y_true,   y_pred)
    }

metrics_val  = compute_metrics(y_val,  model.predict(X_val))
metrics_test = compute_metrics(y_test, model.predict(X_test))

# --- 6. Salvar resultados ---
results = pd.DataFrame([
    {'Conjunto': 'Validação', 'MSE': metrics_val['MSE'], 'MAE': metrics_val['MAE'], 'R2': metrics_val['R2']},
    {'Conjunto': 'Teste',      'MSE': metrics_test['MSE'], 'MAE': metrics_test['MAE'], 'R2': metrics_test['R2']},
])

output_csv = BASE_DIR / 'baseline_results.csv'
results.to_csv(output_csv, index=False)

print('\n=== Baseline MLR Results ===')
print(results.to_string(index=False))