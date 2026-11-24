
from pathlib import Path

import numpy as np
import pandas as pd

from prediccion import predecir_indice
from indice_rendimiento import calcular_iirf


RAIZ = Path(__file__).resolve().parent.parent
RUTA_DATASET = RAIZ / "data" / "dataset_sintetico.csv"
RUTA_RESULTADO = RAIZ / "resultados" / "evaluacion_100_muestras.csv"

COLUMNAS = [
    "ph_entrada",
    "ph_salida",
    "tds_entrada_mg_L",
    "tds_salida_mg_L",
    "ce_entrada_uS_cm_sintetica",
    "ce_salida_uS_cm_sintetica",
    "temperatura_entrada_C",
    "temperatura_salida_C",
]


def evaluar():
    datos = pd.read_csv(RUTA_DATASET)

    muestra = datos.sample(
        n=min(100, len(datos)),
        random_state=123,
    ).copy()

    referencias = []
    predicciones = []

    for _, fila in muestra.iterrows():
        mediciones = {
            columna: float(fila[columna])
            for columna in COLUMNAS
        }

        referencia = calcular_iirf(
            ph_salida=mediciones["ph_salida"],
            tds_entrada=mediciones["tds_entrada_mg_L"],
            tds_salida=mediciones["tds_salida_mg_L"],
        )["iirf_referencia_pct"]

        prediccion = predecir_indice(mediciones)
        estimado = float(prediccion["indice_estimado"])

        referencias.append(referencia)
        predicciones.append(estimado)

    muestra["iirf_calculado"] = referencias
    muestra["prediccion_modelo"] = predicciones
    muestra["error_absoluto"] = np.abs(
        muestra["iirf_calculado"]
        - muestra["prediccion_modelo"]
    )

    muestra.to_csv(RUTA_RESULTADO, index=False)

    mae = muestra["error_absoluto"].mean()
    rmse = np.sqrt(
        np.mean(
            (
                muestra["iirf_calculado"]
                - muestra["prediccion_modelo"]
            ) ** 2
        )
    )

    print("\nEVALUACIÓN DE 100 MUESTRAS")
    print("-" * 40)
    print(f"Muestras evaluadas: {len(muestra)}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(
        "Correlación:",
        f"{muestra['iirf_calculado'].corr(muestra['prediccion_modelo']):.4f}",
    )
    print(f"\nArchivo guardado: {RUTA_RESULTADO}")

    print("\n10 MAYORES ERRORES")
    columnas_salida = [
        "id_muestra",
        "iirf_calculado",
        "prediccion_modelo",
        "error_absoluto",
    ]
    print(
        muestra.nlargest(10, "error_absoluto")[columnas_salida]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    evaluar()
