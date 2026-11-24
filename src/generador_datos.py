
import numpy as np
import pandas as pd
from pathlib import Path

# Generador de datos sinteticos para RedNeuronalAme
rng = np.random.default_rng(42)
N = 1000

def calcular_score_ph(ph):
    if 6.5 <= ph <= 8.5:
        return 100.0
    if ph < 6.5:
        return max(0.0, 100 - (6.5 - ph) * 50)
    return max(0.0, 100 - (ph - 8.5) * 50)

def calcular_score_tds(tds):
    return float(np.clip((2000 - tds) / 1000 * 100, 0, 100))

datos = []

for i in range(1, N + 1):
    # Valores sinteticos del agua antes del filtro
    ph_in = rng.uniform(5.2, 10.0)
    tds_in = rng.uniform(100, 1800)
    temp_in = rng.uniform(15, 35)

    # Comportamiento simulado del filtro
    comportamiento = rng.choice(
        ["mejora", "cambio_minimo", "empeora"],
        p=[0.65, 0.25, 0.10]
    )

    if comportamiento == "mejora":
        ph_out = ph_in + rng.normal(0, 0.65)
        tds_out = tds_in * rng.uniform(0.35, 0.85)
    elif comportamiento == "cambio_minimo":
        ph_out = ph_in + rng.normal(0, 0.12)
        tds_out = tds_in * rng.uniform(0.92, 1.08)
    else:
        ph_out = ph_in + rng.normal(0, 0.8)
        tds_out = tds_in * rng.uniform(1.02, 1.25)

    ph_out = float(np.clip(ph_out, 4.0, 11.0))
    tds_out = float(np.clip(tds_out, 30, 2500))
    temp_out = temp_in + rng.normal(0, 0.5)

    # Conductividad sintetica aproximada.
    # No representa una calibracion real de un sensor.
    ce_in = tds_in * rng.uniform(1.4, 2.0)
    ce_out = tds_out * rng.uniform(1.4, 2.0)

    reduccion = (tds_in - tds_out) / max(tds_in, 1) * 100
    reduccion = float(np.clip(reduccion, 0, 100))

    score_ph = calcular_score_ph(ph_out)
    score_tds = calcular_score_tds(tds_out)
    score_reduccion = reduccion

    # Indice de referencia definido para este proyecto.
    iirf = (
        0.35 * score_ph
        + 0.35 * score_tds
        + 0.30 * score_reduccion
    )

    if iirf >= 80:
        color = "verde"
    elif iirf >= 40:
        color = "amarillo"
    else:
        color = "rojo"

    datos.append({
        "id_muestra": i,
        "origen_datos": "sintetico",
        "comportamiento_simulado": comportamiento,
        "ph_entrada": round(ph_in, 3),
        "ph_salida": round(ph_out, 3),
        "tds_entrada_mg_L": round(tds_in, 2),
        "tds_salida_mg_L": round(tds_out, 2),
        "ce_entrada_uS_cm_sintetica": round(ce_in, 2),
        "ce_salida_uS_cm_sintetica": round(ce_out, 2),
        "temperatura_entrada_C": round(temp_in, 2),
        "temperatura_salida_C": round(temp_out, 2),
        "reduccion_tds_pct": round(reduccion, 2),
        "score_ph": round(score_ph, 2),
        "score_tds": round(score_tds, 2),
        "score_reduccion_tds": round(score_reduccion, 2),
        "iirf_referencia_pct": round(iirf, 2),
        "color_referencia": color
    })

# Guardar el archivo en la carpeta data del proyecto
carpeta_proyecto = Path(__file__).resolve().parent.parent
carpeta_data = carpeta_proyecto / "data"
carpeta_data.mkdir(parents=True, exist_ok=True)

archivo = carpeta_data / "dataset_sintetico.csv"
df = pd.DataFrame(datos)
df.to_csv(archivo, index=False, encoding="utf-8-sig")

print("Dataset sintetico generado correctamente.")
print(f"Archivo: {archivo}")
print(f"Total de muestras: {len(df)}")
print("\nDistribucion por color:")
print(df["color_referencia"].value_counts())
print("\nPrimeras 5 muestras:")
print(df.head().to_string(index=False))
