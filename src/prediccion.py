
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


RUTA_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_MODELO = RUTA_PROYECTO / "modelos" / "red_neuronal_filtro.joblib"


def cargar_modelo():
    """Carga el modelo neuronal previamente entrenado."""
    if not RUTA_MODELO.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo: {RUTA_MODELO}. "
            "Ejecuta primero entrenamiento.py."
        )

    paquete = joblib.load(RUTA_MODELO)

    if not isinstance(paquete, dict) or "modelo" not in paquete:
        raise ValueError(
            "El archivo no contiene el paquete de modelo esperado."
        )

    return paquete


def clasificar_iirf(indice):
    """Clasifica el índice según los límites definidos en el proyecto."""
    if indice >= 80:
        return "verde"
    if indice >= 40:
        return "amarillo"
    return "rojo"


def calcular_iirf_directo(mediciones):
    """
    Calcula el IIRF mediante la fórmula definida en el proyecto.
    Solo necesita pH de salida y TDS de entrada y salida.
    """
    requeridas = [
        "ph_salida",
        "tds_entrada_mg_L",
        "tds_salida_mg_L",
    ]

    faltantes = [nombre for nombre in requeridas if nombre not in mediciones]
    if faltantes:
        raise ValueError(
            "Faltan mediciones para la fórmula directa: "
            + ", ".join(faltantes)
        )

    try:
        ph = float(mediciones["ph_salida"])
        tds_entrada = float(mediciones["tds_entrada_mg_L"])
        tds_salida = float(mediciones["tds_salida_mg_L"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Las mediciones deben ser valores numéricos."
        ) from error

    valores = [ph, tds_entrada, tds_salida]
    if not np.isfinite(valores).all():
        raise ValueError(
            "Las mediciones no pueden contener NaN o infinitos."
        )

    if tds_entrada <= 0 or tds_salida < 0:
        raise ValueError(
            "El TDS de entrada debe ser mayor que cero y el TDS de salida "
            "no puede ser negativo."
        )

    if not 0 <= ph <= 14:
        raise ValueError("El pH debe estar entre 0 y 14.")

    if 6.5 <= ph <= 8.5:
        score_ph = 100.0
    elif ph < 6.5:
        score_ph = max(0.0, 100.0 - (6.5 - ph) * 50)
    else:
        score_ph = max(0.0, 100.0 - (ph - 8.5) * 50)

    score_tds = float(
        np.clip((2000 - tds_salida) / 1000 * 100, 0, 100)
    )

    reduccion_tds = float(
        np.clip(
            (tds_entrada - tds_salida) / max(tds_entrada, 1) * 100,
            0,
            100,
        )
    )

    indice = 0.35 * score_ph + 0.35 * score_tds + 0.30 * reduccion_tds
    indice = float(np.clip(indice, 0, 100))

    return {
        "indice_estimado": round(indice, 2),
        "categoria": clasificar_iirf(indice),
        "metodo": "formula_directa",
        "origen_datos": "mediciones_proporcionadas",
        "interpretacion": (
            "Cálculo según la fórmula experimental del IIRF del proyecto. "
            "No certifica la eficiencia real ni la potabilidad del agua."
        ),
    }


def predecir_indice(mediciones, paquete=None):
    """Estima el IIRF mediante la red neuronal entrenada."""
    if paquete is None:
        paquete = cargar_modelo()

    caracteristicas = paquete["caracteristicas"]
    modelo = paquete["modelo"]

    faltantes = [
        nombre for nombre in caracteristicas if nombre not in mediciones
    ]
    if faltantes:
        raise ValueError("Faltan mediciones: " + ", ".join(faltantes))

    try:
        valores = [float(mediciones[nombre]) for nombre in caracteristicas]
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Todas las mediciones deben ser numéricas."
        ) from error

    if not np.isfinite(valores).all():
        raise ValueError(
            "Las mediciones no pueden contener valores infinitos o NaN."
        )

    entrada = pd.DataFrame([valores], columns=caracteristicas)

    indice = float(modelo.predict(entrada)[0])
    indice = float(np.clip(indice, 0, 100))

    return {
        "indice_estimado": round(indice, 2),
        "categoria": clasificar_iirf(indice),
        "metodo": "red_neuronal",
        "origen_datos": paquete.get("origen_datos", "no especificado"),
        "interpretacion": (
            "Estimación de la red neuronal entrenada con datos sintéticos. "
            "No certifica la eficiencia real ni la potabilidad del agua."
        ),
    }


def evaluar_metodos(mediciones, paquete=None):
    """Devuelve los resultados de ambos métodos para una misma medición."""
    if paquete is None:
        paquete = cargar_modelo()

    return {
        "formula_directa": calcular_iirf_directo(mediciones),
        "red_neuronal": predecir_indice(mediciones, paquete),
    }


if __name__ == "__main__":
    mediciones_prueba = {
        "ph_entrada": 7.2,
        "ph_salida": 7.0,
        "tds_entrada_mg_L": 650,
        "tds_salida_mg_L": 420,
        "ce_entrada_uS_cm_sintetica": 1300,
        "ce_salida_uS_cm_sintetica": 850,
        "temperatura_entrada_C": 22.5,
        "temperatura_salida_C": 22.0,
    }

    try:
        resultados = evaluar_metodos(mediciones_prueba)

        print("\nCOMPARACIÓN DE MÉTODOS")
        print("=" * 45)

        for nombre, resultado in resultados.items():
            print(f"\nMétodo: {nombre}")
            print(f"IIRF: {resultado['indice_estimado']}/100")
            print(f"Categoría: {resultado['categoria']}")
            print(resultado["interpretacion"])

    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"\nERROR: {error}")
