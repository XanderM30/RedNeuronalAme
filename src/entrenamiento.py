
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocesamiento import (
    cargar_y_preparar_datos,
    COLUMNAS_ENTRADA,
    COLUMNA_OBJETIVO,
)


# Rutas del proyecto
RUTA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_MODELOS = RUTA_PROYECTO / "modelos"
RUTA_RESULTADOS = RUTA_PROYECTO / "resultados"

RUTA_MODELO = RUTA_MODELOS / "red_neuronal_filtro.joblib"
RUTA_METRICAS = RUTA_RESULTADOS / "evaluacion_modelo.csv"


def entrenar_red_neuronal():
    """Entrena, evalúa y guarda el modelo de regresión."""

    RUTA_MODELOS.mkdir(parents=True, exist_ok=True)
    RUTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    # 1. Cargar y validar el dataset
    X, y = cargar_y_preparar_datos()

    # 2. Separar datos para entrenamiento y evaluación
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )
    )

    # 3. Definir la red neuronal
    modelo = Pipeline([
        ("escalador", StandardScaler()),
        (
            "red",
            MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                max_iter=2000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                tol=0.0001,
                random_state=42,
                verbose=True,
            ),
        ),
    ])

    # 4. Entrenar únicamente con el conjunto de entrenamiento
    print("\nINICIANDO ENTRENAMIENTO")
    print("-" * 40)

    modelo.fit(X_entrenamiento, y_entrenamiento)

    # 5. Evaluar con el conjunto reservado
    predicciones = modelo.predict(X_prueba)

    mae = mean_absolute_error(y_prueba, predicciones)
    rmse = np.sqrt(mean_squared_error(y_prueba, predicciones))
    r2 = r2_score(y_prueba, predicciones)

    print("\nEVALUACIÓN DEL MODELO")
    print("-" * 40)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    # 6. Guardar el modelo y los metadatos
    red = modelo.named_steps["red"]

    paquete_modelo = {
        "modelo": modelo,
        "caracteristicas": COLUMNAS_ENTRADA,
        "objetivo": COLUMNA_OBJETIVO,
        "arquitectura": (8, 32, 16, 1),
        "origen_datos": "sinteticos",
        "metricas": {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
        },
    }

    joblib.dump(paquete_modelo, RUTA_MODELO)

    # 7. Guardar métricas y predicciones de prueba
    resultados = pd.DataFrame({
        "valor_real_sintetico": y_prueba.to_numpy(),
        "prediccion_modelo": predicciones,
        "error_absoluto": np.abs(
            y_prueba.to_numpy() - predicciones
        ),
    })

    resultados.to_csv(RUTA_METRICAS, index=False)

    print("\nENTRENAMIENTO COMPLETADO")
    print(f"Modelo guardado en: {RUTA_MODELO}")
    print(f"Resultados guardados en: {RUTA_METRICAS}")
    print(f"Iteraciones realizadas: {red.n_iter_}")

    return modelo, resultados


if __name__ == "__main__":
    try:
        entrenar_red_neuronal()
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"\nERROR durante el entrenamiento: {error}")
