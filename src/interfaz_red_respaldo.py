
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from turtle import pd

import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection
import pandas as pd
import numpy as np

# ==========================================
# REDNEURONALAME - VISUALIZADOR MLP
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_MODELO = BASE_DIR / "modelos" / "red_neuronal_filtro.joblib"

PALETA = {
    "fondo": "#0B1220",
    "panel": "#111D30",
    "texto": "#E6F0FF",
    "azul": "#2794FF",
    "verde": "#19D3AE",
    "naranja": "#FFB547",
    "morado": "#A78BFA",
    "borde": "#28415F",
}


class VisualizadorRed:
    def __init__(self, root):
        self.root = root
        self.root.title("RedNeuronalAme | Visualizador neuronal")
        self.root.geometry("1450x850")
        self.root.minsize(1100, 700)
        self.root.configure(bg=PALETA["fondo"])

        if not RUTA_MODELO.exists():
            messagebox.showerror(
                "Modelo no encontrado",
                f"No existe el modelo:\n{RUTA_MODELO}\n\n"
                "Ejecuta primero: py main.py"
            )
            root.destroy()
            return

        paquete = joblib.load(RUTA_MODELO)
        self.modelo = paquete["modelo"]
        self.caracteristicas = paquete["caracteristicas"]
        self.red = self.modelo.named_steps["red"]

        self.capas = [8, 32, 16, 1]
        self.nombres = [
            "ENTRADAS",
            "CAPA OCULTA 1",
            "CAPA OCULTA 2",
            "SALIDA",
        ]

        self.valores = np.zeros(8)
        self._crear_interfaz()
        self._crear_grafica()
        self.actualizar_prediccion()

    def _crear_interfaz(self):
        encabezado = tk.Frame(
            self.root, bg=PALETA["fondo"], padx=18, pady=12
        )
        encabezado.pack(fill="x")

        tk.Label(
            encabezado,
            text="RedNeuronalAme",
            font=("Segoe UI", 22, "bold"),
            fg=PALETA["azul"],
            bg=PALETA["fondo"],
        ).pack(side="left")

        tk.Label(
            encabezado,
            text="Visualización de neuronas, conexiones y pesos",
            font=("Segoe UI", 12),
            fg=PALETA["texto"],
            bg=PALETA["fondo"],
        ).pack(side="left", padx=20)

        cuerpo = tk.Frame(self.root, bg=PALETA["fondo"])
        cuerpo.pack(fill="both", expand=True, padx=14, pady=8)

        # Panel de variables de entrada
        self.panel_entradas = tk.Frame(
            cuerpo,
            bg=PALETA["panel"],
            width=290,
            padx=14,
            pady=12,
            highlightbackground=PALETA["borde"],
            highlightthickness=1,
        )
        self.panel_entradas.pack(side="left", fill="y")
        self.panel_entradas.pack_propagate(False)

        tk.Label(
            self.panel_entradas,
            text="MEDICIONES DEL AGUA",
            font=("Segoe UI", 12, "bold"),
            fg=PALETA["texto"],
            bg=PALETA["panel"],
        ).pack(anchor="w", pady=(0, 10))

        self.campos = {}

        etiquetas = [
            ("pH de entrada", 7.2),
            ("pH de salida", 7.0),
            ("TDS entrada (mg/L)", 650),
            ("TDS salida (mg/L)", 420),
            ("CE entrada (uS/cm)", 1300),
            ("CE salida (uS/cm)", 850),
            ("Temperatura entrada (C)", 22.5),
            ("Temperatura salida (C)", 22.0),
        ]

        for i, (nombre, valor) in enumerate(etiquetas):
            tk.Label(
                self.panel_entradas,
                text=nombre,
                fg=PALETA["texto"],
                bg=PALETA["panel"],
                font=("Segoe UI", 9),
            ).pack(anchor="w", pady=(5, 1))

            variable = tk.StringVar(value=str(valor))
            entrada = ttk.Entry(
                self.panel_entradas,
                textvariable=variable,
                font=("Segoe UI", 10),
            )
            entrada.pack(fill="x")
            self.campos[self.caracteristicas[i]] = variable

        ttk.Button(
            self.panel_entradas,
            text="Calcular predicción",
            command=self.actualizar_prediccion,
        ).pack(fill="x", pady=(16, 8))

        self.lbl_prediccion = tk.Label(
            self.panel_entradas,
            text="Índice: --",
            font=("Segoe UI", 18, "bold"),
            fg=PALETA["verde"],
            bg=PALETA["panel"],
        )
        self.lbl_prediccion.pack(anchor="w", pady=(8, 2))

        self.lbl_color = tk.Label(
            self.panel_entradas,
            text="Categoría: --",
            font=("Segoe UI", 11),
            fg=PALETA["texto"],
            bg=PALETA["panel"],
        )
        self.lbl_color.pack(anchor="w")

        self.lbl_estado = tk.Label(
            self.panel_entradas,
            text="Modelo entrenado con datos sintéticos.",
            wraplength=245,
            justify="left",
            fg="#AFC5E5",
            bg=PALETA["panel"],
            font=("Segoe UI", 9),
        )
        self.lbl_estado.pack(anchor="w", pady=14)

        # Panel de red neuronal
        self.panel_red = tk.Frame(
            cuerpo,
            bg=PALETA["panel"],
            highlightbackground=PALETA["borde"],
            highlightthickness=1,
        )
        self.panel_red.pack(
            side="left", fill="both", expand=True, padx=(12, 0)
        )

        tk.Label(
            self.panel_red,
            text="ARQUITECTURA DE LA RED NEURONAL",
            font=("Segoe UI", 12, "bold"),
            fg=PALETA["texto"],
            bg=PALETA["panel"],
            padx=12,
            pady=10,
        ).pack(anchor="w")

        self.lbl_capas = tk.Label(
            self.panel_red,
            text="8 entradas  →  32 neuronas  →  16 neuronas  →  1 salida",
            font=("Segoe UI", 10),
            fg="#AFC5E5",
            bg=PALETA["panel"],
        )
        self.lbl_capas.pack(anchor="w", padx=12)

        self.area_grafica = tk.Frame(
            self.panel_red, bg=PALETA["panel"]
        )
        self.area_grafica.pack(fill="both", expand=True)

        pie = tk.Label(
            self.root,
            text=(
                "Los colores de las conexiones representan el signo y la "
                "magnitud relativa de los pesos aprendidos. "
                "La visualización no demuestra potabilidad."
            ),
            fg="#AFC5E5",
            bg=PALETA["fondo"],
            font=("Segoe UI", 9),
            padx=16,
            pady=8,
        )
        pie.pack(fill="x")

    def _crear_grafica(self):
        self.fig, self.ax = plt.subplots(figsize=(11, 7))
        self.fig.patch.set_facecolor(PALETA["panel"])
        self.ax.set_facecolor(PALETA["panel"])

        self.canvas = FigureCanvasTkAgg(
            self.fig, master=self.area_grafica
        )
        self.canvas.get_tk_widget().pack(
            fill="both", expand=True
        )

        self._dibujar_red()

    def _dibujar_red(self):
        self.ax.clear()
        self.ax.set_facecolor(PALETA["panel"])
        self.ax.axis("off")

        pesos = self.red.coefs_

        posiciones = []
        x_capas = np.linspace(0.08, 0.92, len(self.capas))

        for x, cantidad in zip(x_capas, self.capas):
            ys = np.linspace(0.06, 0.94, cantidad)
            posiciones.append([(x, y) for y in ys])

        # Dibujar conexiones reales usando los pesos entrenados.
        for indice_capa, matriz in enumerate(pesos):
            origenes = posiciones[indice_capa]
            destinos = posiciones[indice_capa + 1]

            segmentos = []
            colores = []
            anchos = []

            max_peso = max(float(np.max(np.abs(matriz))), 1e-9)

            for i, (x1, y1) in enumerate(origenes):
                for j, (x2, y2) in enumerate(destinos):
                    peso = float(matriz[i, j])
                    segmentos.append([(x1, y1), (x2, y2)])

                    if peso >= 0:
                        colores.append("#2794FF")
                    else:
                        colores.append("#FFB547")

                    anchos.append(
                        0.25 + 1.25 * abs(peso) / max_peso
                    )

            coleccion = LineCollection(
                segmentos,
                colors=colores,
                linewidths=anchos,
                alpha=0.45,
                zorder=1,
            )
            self.ax.add_collection(coleccion)

        colores_nodos = [
            PALETA["azul"],
            PALETA["verde"],
            PALETA["naranja"],
            PALETA["morado"],
        ]

        # Dibujar neuronas
        for k, capa in enumerate(posiciones):
            xs = [p[0] for p in capa]
            ys = [p[1] for p in capa]

            self.ax.scatter(
                xs,
                ys,
                s=85 if k != 3 else 170,
                c=colores_nodos[k],
                edgecolors="#FFFFFF",
                linewidths=0.6,
                zorder=3,
            )

            self.ax.text(
                x_capas[k],
                1.015,
                self.nombres[k],
                ha="center",
                va="bottom",
                color=PALETA["texto"],
                fontsize=9,
                fontweight="bold",
            )

        # Etiquetas abreviadas para las ocho entradas
        nombres_cortos = [
            "pH in", "pH out", "TDS in", "TDS out",
            "CE in", "CE out", "Temp in", "Temp out"
        ]

        for (x, y), nombre in zip(posiciones[0], nombres_cortos):
            self.ax.text(
                x - 0.025,
                y,
                nombre,
                ha="right",
                va="center",
                color=PALETA["texto"],
                fontsize=7,
            )

        self.ax.text(
            posiciones[-1][0][0] + 0.035,
            posiciones[-1][0][1],
            "IIRF",
            color=PALETA["texto"],
            fontsize=9,
            va="center",
        )

        self.ax.set_xlim(-0.18, 1.08)
        self.ax.set_ylim(0, 1.08)
        self.fig.tight_layout()
        self.canvas.draw_idle()


    def actualizar_prediccion(self):
        try:
            valores = [
                float(self.campos[columna].get())
                for columna in self.caracteristicas
            ]

            entrada = pd.DataFrame(
                [valores],
                columns=self.caracteristicas
            )

            prediccion = float(self.modelo.predict(entrada)[0])
            prediccion = float(np.clip(prediccion, 0, 100))

            if prediccion >= 80:
                color = "verde"
                color_texto = "#19D3AE"
            elif prediccion >= 40:
                color = "amarillo"
                color_texto = "#FFB547"
            else:
                color = "rojo"
                color_texto = "#FF647C"

            self.lbl_prediccion.config(
                text=f"Índice: {prediccion:.2f}/100",
                fg=color_texto
            )

            self.lbl_color.config(
                text=f"Categoría estimada: {color}",
                fg=color_texto
            )

            self.lbl_estado.config(
                text=(
                    "Predicción calculada por el modelo entrenado. "
                    "Verifica las unidades y los rangos de las mediciones."
                )
            )

        except ValueError:
            messagebox.showerror(
                "Dato inválido",
                "Introduce números válidos en todos los campos."
            )
        except Exception as error:
            messagebox.showerror(
                "Error de predicción",
                f"No se pudo calcular la predicción:\n{error}"
            )


if __name__ == "__main__":
    ventana = tk.Tk()
    app = VisualizadorRed(ventana)
    ventana.mainloop()
