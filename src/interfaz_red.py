
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection

from prediccion import calcular_iirf_directo, predecir_indice


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
                "Ejecuta primero: py main.py",
            )
            self.root.destroy()
            return

        try:
            paquete = joblib.load(RUTA_MODELO)

            self.modelo = paquete["modelo"]
            self.caracteristicas = paquete["caracteristicas"]
            self.red = self.modelo.named_steps["red"]

        except Exception as error:
            messagebox.showerror(
                "Error al cargar el modelo",
                f"No se pudo cargar el modelo:\n{error}",
            )
            self.root.destroy()
            return

        self.capas = [8, 32, 16, 1]

      	|  self.nombres = [
            "ENTRADAS",
            "CAPA OCULTA 1",
            "CAPA OCULTA 2",
            "SALIDA",
        ]

        self.valores = np.zeros(8)

        self.metodo_var = tk.StringVar(
            value="Comparar ambos"
        )

        self.lbl_comparacion = None

        self._crear_interfaz()
        self._crear_grafica()
        self.actualizar_prediccion()

    # ==========================================
    # INTERFAZ PRINCIPAL
    # ==========================================

    def _crear_interfaz(self):

        encabezado = tk.Frame(
            self.root,
            bg=PALETA["fondo"],
            padx=18,
            pady=12,
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

        cuerpo = tk.Frame(
            self.root,
            bg=PALETA["fondo"],
        )
        cuerpo.pack(
            fill="both",
            expand=True,
            padx=14,
            pady=8,
        )

        # --------------------------------------
        # PANEL DE MEDICIONES
        # --------------------------------------

        self.panel_entradas = tk.Frame(
            cuerpo,
            bg=PALETA["panel"],
            width=290,
            padx=14,
            pady=12,
            highlightbackground=PALETA["borde"],
            highlightthickness=1,
        )

        self.panel_entradas.pack(
            side="left",
            fill="y",
        )
        self.panel_entradas.pack_propagate(False)

        tk.Label(
            self.panel_entradas,
            text="MEDICIONES DEL AGUA",
            font=("Segoe UI", 12, "bold"),
            fg=PALETA["texto"],
            bg=PALETA["panel"],
        ).pack(
            anchor="w",
            pady=(0, 10),
        )

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

        if len(self.caracteristicas) != len(etiquetas):
            raise ValueError(
                "El modelo debe tener exactamente "
                "ocho características de entrada."
            )

        for i, (nombre, valor) in enumerate(etiquetas):

            tk.Label(
                self.panel_entradas,
                text=nombre,
                fg=PALETA["texto"],
                bg=PALETA["panel"],
                font=("Segoe UI", 9),
            ).pack(
                anchor="w",
                pady=(5, 1),
            )

            variable = tk.StringVar(
                value=str(valor)
            )

            entrada = ttk.Entry(
                self.panel_entradas,
                textvariable=variable,
                font=("Segoe UI", 10),
            )
            entrada.pack(fill="x")

            self.campos[self.caracteristicas[i]] = variable

        # --------------------------------------
        # SELECTOR DE MÉTODO
        # --------------------------------------

        tk.Label(
            self.panel_entradas,
            text="MÉTODO DE CÁLCULO",
            font=("Segoe UI", 9, "bold"),
            fg=PALETA["texto"],
            bg=PALETA["panel"],
        ).pack(
            anchor="w",
            pady=(12, 4),
        )

        self.selector_metodo = ttk.Combobox(
            self.panel_entradas,
            textvariable=self.metodo_var,
            values=(
                "Comparar ambos",
                "Fórmula directa",
                "Red neuronal",
            ),
            state="readonly",
        )

        self.selector_metodo.pack(fill="x")

        self.selector_metodo.bind(
            "<<ComboboxSelected>>",
            lambda evento: self.actualizar_prediccion(),
        )

        ttk.Button(
            self.panel_entradas,
            text="Calcular",
            command=self.actualizar_prediccion,
        ).pack(
            fill="x",
            pady=(10, 8),
        )

        # --------------------------------------
        # RESULTADO PRINCIPAL
        # --------------------------------------

        self.lbl_prediccion = tk.Label(
            self.panel_entradas,
            text="Índice: --",
            font=("Segoe UI", 18, "bold"),
            fg=PALETA["verde"],
            bg=PALETA["panel"],
        )

        self.lbl_prediccion.pack(
            anchor="w",
            pady=(8, 2),
        )

        self.lbl_color = tk.Label(
            self.panel_entradas,
            text="Categoría estimada: --",
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

        self.lbl_estado.pack(
            anchor="w",
            pady=10,
        )

        # --------------------------------------
        # COMPARACIÓN DE MÉTODOS
        # --------------------------------------

        self.lbl_comparacion = tk.Label(
            self.panel_entradas,
            text="",
            wraplength=245,
            justify="left",
            fg=PALETA["texto"],
            bg=PALETA["panel"],
            font=("Segoe UI", 9),
        )

        self.lbl_comparacion.pack(
            anchor="w",
            pady=(4, 8),
        )

        # --------------------------------------
        # PANEL DE VISUALIZACIÓN NEURONAL
        # --------------------------------------

        self.panel_red = tk.Frame(
            cuerpo,
            bg=PALETA["panel"],
            highlightbackground=PALETA["borde"],
            highlightthickness=1,
        )

        self.panel_red.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(12, 0),
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
            text=(
                "8 entradas  →  32 neuronas  →  "
                "16 neuronas  →  1 salida"
            ),
            font=("Segoe UI", 10),
            fg="#AFC5E5",
            bg=PALETA["panel"],
        )

        self.lbl_capas.pack(
            anchor="w",
            padx=12,
        )

        self.area_grafica = tk.Frame(
            self.panel_red,
            bg=PALETA["panel"],
        )

        self.area_grafica.pack(
            fill="both",
            expand=True,
        )

        # --------------------------------------
        # PIE DE VENTANA
        # --------------------------------------

        pie = tk.Label(
            self.root,
            text=(
                "Los colores de las conexiones representan el signo "
                "y la magnitud relativa de los pesos aprendidos. "
                "La visualización no demuestra potabilidad."
            ),
            fg="#AFC5E5",
            bg=PALETA["fondo"],
            font=("Segoe UI", 9),
            padx=16,
            pady=8,
        )

        pie.pack(fill="x")

    # ==========================================
    # CREAR GRÁFICA
    # ==========================================

    def _crear_grafica(self):

        self.fig, self.ax = plt.subplots(
            figsize=(11, 7)
        )

        self.fig.patch.set_facecolor(
            PALETA["panel"]
        )

        self.ax.set_facecolor(
            PALETA["panel"]
        )

        self.canvas = FigureCanvasTkAgg(
            self.fig,
            master=self.area_grafica,
        )

        self.canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
        )

        self._dibujar_red()

    # ==========================================
    # DIBUJAR RED NEURONAL
    # ==========================================

    def _dibujar_red(self):

        self.ax.clear()
        self.ax.set_facecolor(PALETA["panel"])
        self.ax.axis("off")

        pesos = self.red.coefs_

        posiciones = []

        x_capas = np.linspace(
            0.08,
            0.92,
            len(self.capas),
        )

        for x, cantidad in zip(x_capas, self.capas):

            ys = np.linspace(
                0.06,
                0.94,
                cantidad,
            )

            posiciones.append(
                [(x, y) for y in ys]
            )

        # Conexiones de acuerdo con los pesos del modelo.
        for indice_capa, matriz in enumerate(pesos):

            origenes = posiciones[indice_capa]
            destinos = posiciones[indice_capa + 1]

            segmentos = []
            colores = []
            anchos = []

            max_peso = max(
                float(np.max(np.abs(matriz))),
                1e-9,
            )

            for i, (x1, y1) in enumerate(origenes):

                for j, (x2, y2) in enumerate(destinos):

                    peso = float(matriz[i, j])

                    segmentos.append(
                        [(x1, y1), (x2, y2)]
                    )

                    if peso >= 0:
                        colores.append(PALETA["azul"])
                    else:
                        colores.append(PALETA["naranja"])

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

        # Neuronas de cada capa.
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

        # Etiquetas de las ocho entradas.
        nombres_cortos = [
            "pH in",
            "pH out",
            "TDS in",
            "TDS out",
            "CE in",
            "CE out",
            "Temp in",
            "Temp out",
        ]

        for (x, y), nombre in zip(
            posiciones[0],
            nombres_cortos,
        ):

            self.ax.text(
                x - 0.025,
                y,
                nombre,
                ha="right",
                va="center",
                color=PALETA["texto"],
                fontsize=7,
            )

        # Etiqueta de salida.
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

    # ==========================================
    # CALCULAR Y COMPARAR MÉTODOS
    # ==========================================

    def actualizar_prediccion(self):

        try:
            # Leer las mediciones.
            mediciones = {
                columna: float(variable.get())
                for columna, variable in self.campos.items()
            }

            valores = list(mediciones.values())

            if not np.isfinite(valores).all():
                raise ValueError(
                    "Todas las mediciones deben ser números finitos."
                )

            metodo = self.metodo_var.get()

            resultado_formula = None
            resultado_red = None

            # Calcular con la fórmula directa.
            if metodo in (
                "Comparar ambos",
                "Fórmula directa",
            ):

                resultado_formula = calcular_iirf_directo(
                    mediciones
                )

            # Calcular con la red neuronal.
            if metodo in (
                "Comparar ambos",
                "Red neuronal",
            ):

                resultado_red = predecir_indice(
                    mediciones,
                    paquete={
                        "modelo": self.modelo,
                        "caracteristicas": self.caracteristicas,
                    },
                )

            # Seleccionar el resultado principal.
            if resultado_red is not None:

                indice = resultado_red["indice_estimado"]
                categoria = resultado_red["categoria"]

            else:

                indice = resultado_formula["indice_estimado"]
                categoria = resultado_formula["categoria"]

            colores = {
                "verde": PALETA["verde"],
                "amarillo": PALETA["naranja"],
                "rojo": "#FF647C",
            }

            color_texto = colores.get(
                categoria,
                PALETA["texto"],
            )

            self.lbl_prediccion.config(
                text=f"Índice: {indice:.2f}/100",
                fg=color_texto,
            )

            self.lbl_color.config(
                text=f"Categoría estimada: {categoria}",
                fg=color_texto,
            )

            # Texto de comparación.
            if metodo == "Comparar ambos":

                indice_formula = resultado_formula[
                    "indice_estimado"
                ]

                indice_red = resultado_red[
                    "indice_estimado"
                ]

                diferencia = abs(
                    indice_formula - indice_red
                )

                texto_comparacion = (
                    "FÓRMULA DIRECTA\n"
                    f"IIRF: {indice_formula:.2f}/100\n"
                    f"Categoría: {resultado_formula['categoria']}\n\n"
                    "RED NEURONAL MLP\n"
                    f"IIRF: {indice_red:.2f}/100\n"
                    f"Categoría: {resultado_red['categoria']}\n\n"
                    f"Diferencia: {diferencia:.2f} puntos"
                )

            elif metodo == "Fórmula directa":

                texto_comparacion = (
                    "MÉTODO: FÓRMULA DIRECTA\n\n"
                    "El índice se calcula aplicando las reglas "
                    "definidas para el IIRF experimental."
                )

            else:

                texto_comparacion = (
                    "MÉTODO: RED NEURONAL MLP\n\n"
                    "El índice es una estimación del modelo "
                    "entrenado con datos sintéticos."
                )

            self.lbl_comparacion.config(
                text=texto_comparacion
            )

            self.lbl_estado.config(
                text=(
                    "Modelo experimental con datos sintéticos. "
                    "El resultado no demuestra la eficiencia real "
                    "del filtro ni certifica la potabilidad del agua."
                )
            )

        except (ValueError, TypeError) as error:

            messagebox.showerror(
                "Dato inválido",
                f"Revisa las mediciones introducidas:\n{error}",
            )

        except Exception as error:

            messagebox.showerror(
                "Error de predicción",
                f"No se pudo calcular el índice:\n{error}",
            )


# ==========================================
# INICIAR APLICACIÓN
# ==========================================

if __name__ == "__main__":

    ventana = tk.Tk()
    app = VisualizadorRed(ventana)
    ventana.mainloop()
