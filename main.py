
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ==========================================
# REDNEURONALAME
# Estimacion del indice de rendimiento
# de un filtro de agua mediante una MLP
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "data" / "dataset_sintetico.csv"
CARPETA_MODELOS = BASE_DIR / "modelos"
CARPETA_RESULTADOS = BASE_DIR / "resultados"

CARPETA_MODELOS.mkdir(parents=True, exist_ok=True)
CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

COLUMNAS_ENTRADA = [
    "ph_entrada",
    "ph_salida",
    "tds_entrada_mg_L",
    "tds_salida_mg_L",
    "ce_entrada_uS_cm_sintetica",
    "ce_salida_uS_cm_sintetica",
    "temperatura_entrada_C",
    "temperatura_salida_C",
]

COLUMNA_OBJETIVO = "iirf_referencia_pct"


def cargar_datos():
    """Carga y valida el dataset sintetico."""

    if not DATASET.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset: {DATASET}"
        )

    df = pd.read_csv(DATASET)

    columnas_requeridas = COLUMNAS_ENTRADA + [
        COLUMNA_OBJETIVO
    ]

    faltantes = [
        c for c in columnas_requeridas
        if c not in df.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas en el CSV: {faltantes}"
        )

    df = df[columnas_requeridas].copy()

    # Convertir columnas a valores numericos.
    for columna in columnas_requeridas:
        df[columna] = pd.to_numeric(
            df[columna], errors="coerce"
        )

    # Eliminar registros incompletos o no numericos.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    if len(df) < 20:
        raise ValueError(
            "Se necesitan al menos 20 muestras validas."
        )

    if not df[COLUMNA_OBJETIVO].between(0, 100).all():
        raise ValueError(
            "El indice de referencia debe estar entre 0 y 100."
        )

    return df


def crear_modelo():
    """Construye la red neuronal multicapa."""

    return Pipeline([
        ("escalador", StandardScaler()),
        ("red", MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            tol=0.0001,
            random_state=42,
            verbose=True
        ))
    ])


def clasificar_indice(indice):
    """Asigna un color al indice de referencia estimado."""

    if indice >= 80:
        return "verde"
    elif indice >= 40:
        return "amarillo"
    return "rojo"


def entrenar():
    """Entrena, evalua y guarda el modelo."""

    print("=" * 55)
    print("       REDNEURONALAME")
    print("  Entrenamiento de red neuronal MLP")
    print("=" * 55)

    df = cargar_datos()

    X = df[COLUMNAS_ENTRADA]
    y = df[COLUMNA_OBJETIVO]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    modelo = crear_modelo()

    print(f"\nMuestras validas: {len(df)}")
    print(f"Para entrenamiento: {len(X_train)}")
    print(f"Para prueba: {len(X_test)}")
    print("\nArquitectura: 8 -> 32 -> 16 -> 1")
    print("\nIniciando aprendizaje...\n")

    modelo.fit(X_train, y_train)

    red = modelo.named_steps["red"]

    # Evaluar con datos que no participaron en el entrenamiento.
    predicciones = np.clip(
        modelo.predict(X_test), 0, 100
    )

    mae = mean_absolute_error(y_test, predicciones)
    rmse = np.sqrt(
        mean_squared_error(y_test, predicciones)
    )
    r2 = r2_score(y_test, predicciones)

    print("\n" + "=" * 55)
    print("RESULTADOS DE EVALUACION")
    print("=" * 55)
    print(f"Iteraciones realizadas: {red.n_iter_}")
    print(f"Perdida final: {red.loss_:.6f}")
    print(f"MAE: {mae:.3f} puntos")
    print(f"RMSE: {rmse:.3f} puntos")
    print(f"R2: {r2:.4f}")

    if hasattr(red, "best_validation_score_"):
        print(
            "Mejor R2 de validacion: "
            f"{red.best_validation_score_:.4f}"
        )

    # Guardar modelo, variables y metadatos.
    ruta_modelo = (
        CARPETA_MODELOS / "red_neuronal_filtro.joblib"
    )

    paquete = {
        "modelo": modelo,
        "caracteristicas": COLUMNAS_ENTRADA,
        "objetivo": COLUMNA_OBJETIVO,
        "arquitectura": (8, 32, 16, 1),
        "origen_datos": "sintetico",
        "metricas": {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
        },
    }

    joblib.dump(paquete, ruta_modelo)

    # Guardar predicciones para analizarlas despues.
    evaluacion = pd.DataFrame({
        "indice_referencia": y_test.to_numpy(),
        "indice_estimado": predicciones,
    })

    evaluacion["error_absoluto"] = (
        evaluacion["indice_referencia"]
        - evaluacion["indice_estimado"]
    ).abs()

    evaluacion["color_estimado"] = [
        clasificar_indice(v) for v in predicciones
    ]

    evaluacion.to_csv(
        CARPETA_RESULTADOS / "evaluacion_modelo.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nModelo guardado: {ruta_modelo}")
    print(
        "Evaluacion guardada en: "
        f"{CARPETA_RESULTADOS / 'evaluacion_modelo.csv'}"
    )

    print("\nEntrenamiento terminado.")
    print(
        "Nota: la evaluacion utiliza datos sinteticos "
        "y un indice de referencia definido para el proyecto."
    )


if __name__ == "__main__":
    entrenar()
