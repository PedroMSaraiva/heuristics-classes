import scipy.io
import pandas as pd
import os

# Garante que a pasta de destino existe
os.makedirs('data/csv', exist_ok=True)

# --- 1. ShootOut2012MATLAB.mat ---
mat1 = scipy.io.loadmat('C:/Users/pedro/OneDrive/Documentos/heuristics-classes/atividade_final/data/mat/ShootOut2012MATLAB.mat')

# Calibration
input_cal = pd.DataFrame(mat1['inputCalibration'])
target_cal = pd.DataFrame(mat1['targetCalibration'], columns=['target'])
calibration = pd.concat([input_cal, target_cal], axis=1)
calibration.to_csv('data/csv/calibration.csv', index=False)

# Test
input_test = pd.DataFrame(mat1['inputTest'])
target_test = pd.DataFrame(mat1['targetTest'], columns=['target'])
test = pd.concat([input_test, target_test], axis=1)
test.to_csv('data/csv/test.csv', index=False)

# Validation
input_val = pd.DataFrame(mat1['inputValidation'])
input_val.to_csv('data/csv/validation.csv', index=False)

# Wavelengths (opcional)
wl = pd.DataFrame(mat1['wl'])
wl.to_csv('data/csv/wl.csv', index=False)

# --- 2. IDRC_Validation_set_references_Sheet1.mat ---
mat2 = scipy.io.loadmat('C:/Users/pedro/OneDrive/Documentos/heuristics-classes/atividade_final/data/mat/IDRC_Validation_set_references_Sheet1.mat')

# Extrai os dados
ref_values = pd.DataFrame(mat2['IDRC_Validation_set_references_Sheet1'], columns=['reference'])
obj_labels = pd.DataFrame([''.join(row).strip() for row in mat2['ObjLabels']], columns=['label'])

# Junta os labels e os valores de referência
idrc = pd.concat([obj_labels, ref_values], axis=1)
idrc.to_csv('data/csv/idrc_validation.csv', index=False)