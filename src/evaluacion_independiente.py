
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

from prediccion import predecir_indice, cargar_modelo


RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ / "resultados" / "evaluacion_independiente.csv"
RUTA_METRICAS = RAIZ / "resultados" / "metricas_independientes.csv"

N_MUESTRAS = 1000
SEMILLA = 20261019


def generar_datos_independientes(n=N_MUESTRAS):
    """Genera datos nuevos sin modificar el dataset de entrenamiento."""
    rng = np.random.default_rng(SEMILLA)
    filas = []

    for i in range(1, n + 1):
        ph_in = rng.uniform(5.2, 10.0)
        tds_in = rng.uniform(100, 1800)
        temp_in = rng.uniform(15, 35)

        comportamiento = rng.choice(
            ["mejora", "cambio_minimo", "empeora"],
            p=[0.65, 0.25, 0.10],
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

        ce_in = tds_in * rng.uniform(1.4, 2.0)
        ce_out = tds_out * rng.uniform(1.4, 2.0)

        score_ph = (
            100.0
            if 6.5 <= ph_out <= 8.5
            else max(
                0.0,
                100.0 - abs(ph_out - (6.5 if ph_out < 6.5 else 8.5)) * 50,
            )
        )

        score_tds = float(
            np.clip((2000 - tds_out) / 1000 * 100, 0, 100)
        )

        reduccion = float(
            np.clip((tds_in - tds_out) / max(tds_in, 1) * 100, 0, 100)
        )

        iirf = (
            0.35 * score_ph
            + 0.35 * score_tds
            + 0.30 * reduccion
        )

        filas.append({
            "id_muestra": i,
            "comportamiento_simulado": comportamiento,
            "ph_entrada": ph_in,
            "ph_salida": ph_out,
            "tds_entrada_mg_L": tds_in,
            "tds_salida_mg_L": tds_out,
            "ce_entrada_uS_cm_sintetica": ce_in,
            "ce_salida_uS_cm_sintetica": ce_out,
            "temperatura_entrada_C": temp_in,
            "temperatura_salida_C": temp_out,
            "iirf_referencia": iirf,
        })

    return pd.DataFrame(filas)


def clasificar(valor):
    if valor < 40:
        return "rojo"
    if valor < 80:
        return "amarillo"
    return "verde"


def evaluar():
    paquete = cargar_modelo()
    datos = generar_datos_independientes()

    predicciones = []

    for _, fila in datos.iterrows():
        mediciones = {
            nombre: float(fila[nombre])
            for nombre in paquete["caracteristicas"]
        }
        resultado = predecir_indice(mediciones, paquete)
        predicciones.append(resultado["indice_estimado"])

    datos["prediccion_red"] = predicciones
    datos["error_absoluto"] = np.abs(
        datos["iirf_referencia"] - datos["prediccion_red"]
    )
    datos["categoria_real"] = datos["iirf_referencia"].apply(clasificar)
    datos["categoria_predicha"] = datos["prediccion_red"].apply(clasificar)

    y_real = datos["iirf_referencia"]
    y_pred = datos["prediccion_red"]

    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)
    exactitud = accuracy_score(
        datos["categoria_real"],
        datos["categoria_predicha"],
    )

    categorias = ["rojo", "amarillo", "verde"]
    matriz = confusion_matrix(
        datos["categoria_real"],
        datos["categoria_predicha"],
        labels=categorias,
    )

    RUTA_DATOS.parent.mkdir(parents=True, exist_ok=True)
    datos.to_csv(RUTA_DATOS, index=False, encoding="utf-8-sig")

    metricas = pd.DataFrame([{
        "muestras": len(datos),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "exactitud_clasificacion": exactitud,
    }])
    metricas.to_csv(RUTA_METRICAS, index=False, encoding="utf-8-sig")

    print("\nEVALUACION INDEPENDIENTE SINTETICA")
    print("=" * 45)
    print(f"Muestras evaluadas: {len(datos)}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"Exactitud de clasificación: {exactitud * 100:.2f}%")

    print("\nMATRIZ DE CONFUSION")
    print(pd.DataFrame(
        matriz,
        index=[f"Real {c}" for c in categorias],
        columns=[f"Predicha {c}" for c in categorias],
    ))

    print("\nREPORTE POR CATEGORIA")
    print(classification_report(
        datos["categoria_real"],
        datos["categoria_predicha"],
        labels=categorias,
        zero_division=0,
    ))

    print("\nArchivos guardados:")
    print(RUTA_DATOS)
    print(RUTA_METRICAS)
    print("\nNota: son datos sintéticos nuevos, no mediciones reales de agua.")


if __name__ == "__main__":
    evaluar()
