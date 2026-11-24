
import numpy as np


def calcular_score_ph(ph):
    """Calcula la puntuación del pH en una escala de 0 a 100."""

    ph = float(ph)

    if not np.isfinite(ph) or not 0 <= ph <= 14:
        raise ValueError("El pH debe ser un número entre 0 y 14.")

    if 6.5 <= ph <= 8.5:
        return 100.0

    if ph < 6.5:
        return float(max(0, 100 - (6.5 - ph) * 50))

    return float(max(0, 100 - (ph - 8.5) * 50))


def calcular_score_tds(tds_salida):
    """Calcula la puntuación de TDS de salida."""

    tds_salida = float(tds_salida)

    if not np.isfinite(tds_salida) or tds_salida < 0:
        raise ValueError("Los TDS deben ser un número finito no negativo.")

    puntuacion = (2000 - tds_salida) / 1000 * 100

    return float(np.clip(puntuacion, 0, 100))


def calcular_reduccion_tds(tds_entrada, tds_salida):
    """Calcula el porcentaje de reducción de TDS."""

    tds_entrada = float(tds_entrada)
    tds_salida = float(tds_salida)

    if (
        not np.isfinite(tds_entrada)
        or not np.isfinite(tds_salida)
        or tds_entrada < 0
        or tds_salida < 0
    ):
        raise ValueError("Los TDS deben ser finitos y no negativos.")

    if tds_entrada == 0:
        return 0.0

    reduccion = (
        (tds_entrada - tds_salida)
        / max(tds_entrada, 1)
    ) * 100

    return float(np.clip(reduccion, 0, 100))


def clasificar_iirf(indice):
    """Clasifica el índice en verde, amarillo o rojo."""

    indice = float(indice)

    if not np.isfinite(indice) or not 0 <= indice <= 100:
        raise ValueError("El índice debe estar entre 0 y 100.")

    if indice >= 80:
        return "verde"

    if indice >= 40:
        return "amarillo"

    return "rojo"


def calcular_iirf(ph_salida, tds_entrada, tds_salida):
    """
    Calcula el índice de referencia del filtro.

    Fórmula:
        IIRF = 0.35 * score_pH
             + 0.35 * score_TDS
             + 0.30 * score_reduccion_TDS

    Este índice es una referencia experimental del proyecto,
    no una certificación de calidad ni de potabilidad.
    """

    score_ph = calcular_score_ph(ph_salida)
    score_tds = calcular_score_tds(tds_salida)
    reduccion = calcular_reduccion_tds(
        tds_entrada,
        tds_salida
    )

    indice = (
        0.35 * score_ph
        + 0.35 * score_tds
        + 0.30 * reduccion
    )

    indice = float(np.clip(indice, 0, 100))

    return {
        "score_ph": round(score_ph, 2),
        "score_tds": round(score_tds, 2),
        "reduccion_tds_pct": round(reduccion, 2),
        "iirf_referencia_pct": round(indice, 2),
        "color_referencia": clasificar_iirf(indice),
    }


if __name__ == "__main__":
    try:
        resultado = calcular_iirf(
            ph_salida=7.0,
            tds_entrada=650,
            tds_salida=420,
        )

        print("\nCÁLCULO DEL IIRF")
        print("-" * 40)

        for nombre, valor in resultado.items():
            print(f"{nombre}: {valor}")

    except ValueError as error:
        print(f"Error: {error}")
