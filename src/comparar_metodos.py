
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from preprocesamiento import (
    cargar_y_preparar_datos,
    COLUMNAS_ENTRADA,
)
from prediccion import cargar_modelo


RAIZ = Path(__file__).resolve().parent.parent
RUTA_SALIDA = RAIZ / "resultados" / "comparacion_metodos.csv"


def evaluar():
    # Cargar los datos validados
    X, y = cargar_y_preparar_datos()

    # Usar la misma separación que el entrenamiento original
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    # Método 1: red neuronal ya entrenada
    paquete = cargar_modelo()
    modelo = paquete["modelo"]
    pred_red = modelo.predict(X_test)
    pred_red = np.clip(pred_red, 0, 100)

    # Método 2: referencia sencilla que siempre predice
    # el promedio del conjunto de entrenamiento
    referencia = DummyRegressor(strategy="mean")
    referencia.fit(X_train, y_train)
    pred_promedio = referencia.predict(X_test)

    # Método 3: referencia matemática directa.
    # Recalcular el IIRF con las variables de cada muestra.
    ph = X_test["ph_salida"].to_numpy(dtype=float)
    tds_in = X_test["tds_entrada_mg_L"].to_numpy(dtype=float)
    tds_out = X_test["tds_salida_mg_L"].to_numpy(dtype=float)

    score_ph = np.where(
        (ph >= 6.5) & (ph <= 8.5),
        100.0,
        np.where(
            ph < 6.5,
            np.maximum(0, 100 - (6.5 - ph) * 50),
            np.maximum(0, 100 - (ph - 8.5) * 50),
        ),
    )

    score_tds = np.clip((2000 - tds_out) / 1000 * 100, 0, 100)
    reduccion = np.clip(
        (tds_in - tds_out) / np.maximum(tds_in, 1) * 100,
        0,
        100,
    )

    pred_formula = (
        0.35 * score_ph
        + 0.35 * score_tds
        + 0.30 * reduccion
    )

    metodos = {
        "Red neuronal": pred_red,
        "Promedio de entrenamiento": pred_promedio,
        "Formula directa": pred_formula,
    }

    filas = []

    print("\nCOMPARACION DE METODOS")
    print("=" * 76)

    for nombre, pred in metodos.items():
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)

        filas.append({
            "metodo": nombre,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        print(f"\n{nombre}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2:   {r2:.4f}")

    # Guardar resultados detallados por muestra
    detalle = X_test.copy()
    detalle["valor_referencia"] = y_test
    detalle["prediccion_red_neuronal"] = pred_red
    detalle["prediccion_promedio"] = pred_promedio
    detalle["iirf_formula_directa"] = pred_formula
    detalle["error_red"] = np.abs(y_test.to_numpy() - pred_red)
    detalle["error_formula"] = np.abs(y_test.to_numpy() - pred_formula)

    RUTA_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    detalle.to_csv(RUTA_SALIDA, index=False, encoding="utf-8-sig")

    print("\nArchivo de resultados:")
    print(RUTA_SALIDA)

    print("\nNOTA: todos los resultados corresponden a datos sinteticos.")


if __name__ == "__main__":
    evaluar()
