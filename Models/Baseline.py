import pandas as pd
import os

# 1. Configurar la ruta relativa segura
print("Cargando archivo Parquet...")
base_dir = os.path.dirname(os.path.dirname(__file__)) 
file_path = os.path.join(base_dir, 'Data', 'stuff_model_df.parquet')

df = pd.read_parquet(file_path)

# 2. Calcular el modelo nulo (baseline)
print("Calculando el xRunValue promedio por cuenta...")

baseline_df = df.groupby('count')['xRunValue'].mean().reset_index()
baseline_df = baseline_df.sort_values(by='count')

# 3. Imprimir los resultados en la terminal
print("\n--- Baseline Model: xRunValue Promedio por Cuenta ---")
print(baseline_df.to_string(index=False))

# 4. Guardar el resultado en un archivo CSV para tus entregables
output_path = os.path.join(base_dir, 'Data', 'baseline_xRunValue.csv')
baseline_df.to_csv(output_path, index=False)
print(f"\nModelo nulo guardado exitosamente en: {output_path}")