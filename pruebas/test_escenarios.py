
import unittest
import pandas as pd

from src.indice_rendimiento import calcular_iirf
from src.prediccion import predecir_indice


ESCENARIOS = [
    {
        "nombre": "A",
        "ph_entrada": 8.0,
        "ph_salida": 7.2,
        "tds_entrada_mg_L": 1000,
        "tds_salida_mg_L": 400,
        "ce_entrada_uS_cm_sintetica": 1700,
        "ce_salida_uS_cm_sintetica": 700,
        "temperatura_entrada_C": 25,
        "temperatura_salida_C": 25,
    },
    {
        "nombre": "B",
        "ph_entrada": 7.5,
        "ph_salida": 7.3,
        "tds_entrada_mg_L": 800,
        "tds_salida_mg_L": 720,
        "ce_entrada_uS_cm_sintetica": 1400,
        "ce_salida_uS_cm_sintetica": 1250,
        "temperatura_entrada_C": 24,
        "temperatura_salida_C": 24.5,
    },
    {
        "nombre": "C",
        "ph_entrada": 7.0,
        "ph_salida": 6.0,
        "tds_entrada_mg_L": 500,
        "tds_salida_mg_L": 650,
        "ce_entrada_uS_cm_sintetica": 900,
        "ce_salida_uS_cm_sintetica": 1200,
        "temperatura_entrada_C": 25,
        "temperatura_salida_C": 26,
    },
]


class TestEscenarios(unittest.TestCase):

    def test_comparacion_con_referencia(self):
        resultados = []

        for escenario in ESCENARIOS:
            datos = {
                clave: valor
                for clave, valor in escenario.items()
                if clave != "nombre"
            }

            referencia = calcular_iirf(
                datos["ph_salida"],
                datos["tds_entrada_mg_L"],
                datos["tds_salida_mg_L"],
            )

            prediccion = predecir_indice(datos)

            estimado = float(prediccion["indice_estimado"])
            real = referencia["iirf_referencia_pct"]
            error = abs(estimado - real)

            resultados.append({
                "escenario": escenario["nombre"],
                "referencia": real,
                "prediccion": round(estimado, 2),
                "error_absoluto": round(error, 2),
            })

        tabla = pd.DataFrame(resultados)

        print("\nCOMPARACIÓN DE ESCENARIOS")
        print(tabla.to_string(index=False))

        print(
            "\nError absoluto promedio:",
            round(tabla["error_absoluto"].mean(), 2),
        )

        self.assertEqual(len(resultados), 3)
        self.assertTrue(
            tabla["prediccion"].between(0, 100).all()
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
