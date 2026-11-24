
from pathlib import Path

import numpy as np
import pandas as pd


# Rutas del proyecto
RUTA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_DATASET = RUTA_PROYECTO / "data" / "dataset_sintetico.csv"


# Variables que utilizará la red neuronal
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


def cargar_y_preparar_datos(ruta_dataset=None):
    """
    Carga y valida el dataset para el entrenamiento.

    Devuelve:
        X: DataFrame con las ocho variables de entrada.
        y: Serie con el índice de referencia.
    """

    if ruta_dataset is None:
        ruta_dataset = RUTA_DATASET

    ruta_dataset = Path(ruta_dataset)

    if not ruta_dataset.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {ruta_dataset}"
        )

    # Cargar los datos sin modificar el archivo original
    datos = pd.read_csv(ruta_dataset)

    if datos.empty:
        raise ValueError("El dataset no contiene registros.")

    # Comprobar que existan las columnas necesarias
    columnas_requeridas = COLUMNAS_ENTRADA + [
        COLUMNA_OBJETIVO
    ]

    columnas_faltantes = [
        columna
        for columna in columnas_requeridas
        if columna not in datos.columns
    ]

    if columnas_faltantes:
        raise ValueError(
            "Faltan columnas necesarias en el dataset: "
            + ", ".join(columnas_faltantes)
        )

    # Seleccionar únicamente las variables necesarias
    datos_modelo = datos[columnas_requeridas].copy()

    # Convertir las variables a valores numéricos
    for columna in columnas_requeridas:
        datos_modelo[columna] = pd.to_numeric(
            datos_modelo[columna],
            errors="coerce"
        )

    # Detectar valores ausentes o no numéricos
    filas_invalidas = datos_modelo.isna().any(axis=1)

    if filas_invalidas.any():
        cantidad = int(filas_invalidas.sum())

        raise ValueError(
            f"Se encontraron {cantidad} filas con valores "
            "vacíos o no numéricos. Revisa el dataset antes "
            "de entrenar."
        )

    # Detectar valores infinitos
    matriz = datos_modelo.to_numpy(dtype=float)

    if not np.isfinite(matriz).all():
        raise ValueError(
            "El dataset contiene valores infinitos "
            "o no válidos."
        )

    # Separar entradas y objetivo
    X = datos_modelo[COLUMNAS_ENTRADA].copy()
    y = datos_modelo[COLUMNA_OBJETIVO].copy()

    # Verificar el rango del índice de referencia
    if not y.between(0, 100).all():
        raise ValueError(
            "El índice de referencia debe estar entre 0 y 100."
        )

    print("\nPREPROCESAMIENTO COMPLETADO")
    print("-" * 40)
    print(f"Archivo: {ruta_dataset.name}")
    print(f"Registros: {len(datos_modelo)}")
    print(f"Variables de entrada: {len(COLUMNAS_ENTRADA)}")
    print(f"Valores faltantes: {int(datos_modelo.isna().sum().sum())}")
    print(f"Rango del objetivo: {y.min():.2f} a {y.max():.2f}")

    return X, y


if __name__ == "__main__":
    try:
        X, y = cargar_y_preparar_datos()

        print("\nPrimeras cinco filas de entrada:")
        print(X.head())

        print("\nPrimeros cinco índices de referencia:")
        print(y.head())

        print("\nEl dataset está listo para el entrenamiento.")

    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"\nERROR: {error}")
