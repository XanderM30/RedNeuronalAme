
import unittest

from src.indice_rendimiento import (
    calcular_score_ph,
    calcular_score_tds,
    calcular_reduccion_tds,
    calcular_iirf,
    clasificar_iirf,
)


class TestIIRF(unittest.TestCase):

    def test_ph_neutro(self):
        self.assertEqual(calcular_score_ph(7.0), 100.0)

    def test_ph_fuera_del_rango(self):
        self.assertEqual(calcular_score_ph(6.0), 75.0)

    def test_tds_cero(self):
        self.assertEqual(calcular_score_tds(0), 100.0)

    def test_reduccion_tds(self):
        resultado = calcular_reduccion_tds(650, 420)
        self.assertAlmostEqual(resultado, 35.3846, places=3)

    def test_iirf_referencia(self):
        resultado = calcular_iirf(7.0, 650, 420)
        self.assertEqual(resultado["iirf_referencia_pct"], 80.62)
        self.assertEqual(resultado["color_referencia"], "verde")

    def test_clasificacion(self):
        self.assertEqual(clasificar_iirf(85), "verde")
        self.assertEqual(clasificar_iirf(60), "amarillo")
        self.assertEqual(clasificar_iirf(20), "rojo")

    def test_ph_invalido(self):
        with self.assertRaises(ValueError):
            calcular_score_ph(15)

    def test_tds_negativo(self):
        with self.assertRaises(ValueError):
            calcular_score_tds(-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
