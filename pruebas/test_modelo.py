
import unittest
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


class TestModelo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        raiz = Path(__file__).resolve().parent.parent
        ruta_modelo = raiz / "modelos" / "red_neuronal_filtro.joblib"

        cls.paquete = joblib.load(ruta_modelo)
        cls.modelo = cls.paquete["modelo"]
        cls.caracteristicas = cls.paquete["caracteristicas"]

        cls.muestra = pd.DataFrame(
            [[7.0, 7.2, 650, 420, 1000, 700, 25, 25.2]],
            columns=cls.caracteristicas,
        )

    def test_modelo_cargado(self):
        self.assertIsNotNone(self.modelo)

    def test_ocho_variables_de_entrada(self):
        self.assertEqual(len(self.caracteristicas), 8)

    def test_prediccion_numerica(self):
        prediccion = self.modelo.predict(self.muestra)

        self.assertEqual(len(prediccion), 1)
        self.assertTrue(np.isfinite(prediccion[0]))

    def test_prediccion_en_rango(self):
        prediccion = float(self.modelo.predict(self.muestra)[0])

        self.assertGreaterEqual(prediccion, 0)
        self.assertLessEqual(prediccion, 100)

    def test_metricas_guardadas(self):
        self.assertIn("metricas", self.paquete)
        metricas = self.paquete["metricas"]

        self.assertIn("MAE", metricas)
        self.assertIn("RMSE", metricas)
        self.assertIn("R2", metricas)


if __name__ == "__main__":
    unittest.main(verbosity=2)
