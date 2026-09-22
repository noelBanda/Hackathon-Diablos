import pandas as pd

# 1. Cargar el dataset
print("Cargando archivo Parquet...")
df = pd.read_parquet('/Data/stuff_model_df.parquet') # Asegúrate de que el archivo esté en la misma ruta

# 2. Definir listas de columnas según el diccionario de datos
# Identificadores directos y de agrupación que no deben ser entrenados
id_columns = [
    'PitchUID', 
    'game_anon_id', 
    'pitcher_anon_id', 
    'batter_anon_id', 
    'catcher_anon_id'
]

# Variables marcadas como target_only (Lo que vamos a predecir o resultados de la jugada)
target_columns = [
    'PitchCall', 'KorBB', 'play_result', 'hit_type', 'ExitSpeed', 
    'Angle', 'Direction', 'Distance', 'is_swing', 'is_whiff', 
    'is_contact', 'is_called_strike', 'is_swinging_strike', 
    'is_ball_in_play', 'is_batted', 'is_hit', 'single', 'double', 
    'triple', 'home_run', 'base_on_balls', 'strikeout', 
    'is_hit_by_pitch', 'RunsScored', 'OutsOnPlay', 'swung_outside_strike_zone'
]

# 3. Separar los targets del dataset principal (Guardamos 'X' y 'y')
# Guardamos los targets en un DataFrame independiente (útil para la fase de entrenamiento)
df_targets = df[target_columns].copy()

# 4. Limpiar el dataset de entrenamiento (Features)
# Eliminamos los identificadores y los targets del set de características
df_features = df.drop(columns=id_columns + target_columns)

print(f"Dimensión de Features originales: {df.shape}")
print(f"Dimensión de Features limpios (X): {df_features.shape}")
print(f"Dimensión de Targets (y): {df_targets.shape}")