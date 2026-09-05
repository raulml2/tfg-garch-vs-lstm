# Predicción de la Volatilidad Bursátil: Modelo GARCH vs. Red Neuronal LSTM

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Framework-Keras](<https://img.shields.io/badge/Deep%20Learning-Keras%2FTensorFlow-orange.svg>)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Trabajo Fin de Grado (TFG)**
> **Autor:** Raúl Méndez López
> **Tutor:** Javier Oliver Muncharaz
> **Institución:** Universitat Politècnica de València (UPV) — Facultad de Administración y Dirección de Empresas (ADE)

---

## 📌 Resumen del Proyecto

La volatilidad financiera es uno de los parámetros clave en la gestión de riesgos y en la toma de decisiones de inversión. Este proyecto realiza un estudio comparativo y empírico entre la econometría clásica y el aprendizaje profundo (*Deep Learning*) para predecir la volatilidad del precio de cierre de las principales empresas tecnológicas del mercado **NASDAQ-100**:

* **Amazon (AMZN)**
* **Apple (AAPL)**
* **Alphabet / Google (GOOGL)**
* **Microsoft (MSFT)**
* **Meta (META)**
* **Tesla (TSLA)**

El periodo analizado abarca desde **mayo de 2012 hasta septiembre de 2023** (4.152 registros diarios), evaluando la precisión predictiva a **escala diaria, semanal y mensual**.

---

## 📊 Metodología e Implementación

### 1. Preprocesamiento e Inspección Estadística

* Obtención de series temporales financieras mediante la API de `yfinance`.
* Cálculo de rentabilidades logarítmicas ($R_t = \ln(P_t) - \ln(P_{t-1})$) y estimación de la volatilidad histórica mediante desviación típica móvil.
* Test de **Augmented Dickey-Fuller (ADF)** para verificar la estacionariedad de las series temporales de rentabilidad y volatilidad ($p \text{-valor} < 0.05$).

### 2. Modelos Econométricos Clásicos

* **GARCH(p, q)**: Selección de hiperparámetros óptimos $(p, q)$ mediante la minimización del Criterio de Información de Akaike (AIC). Estimación mediante Máxima Verosimilitud Condicional (MLE) usando la librería `arch`.
* **EGARCH**: Simulación complementaria para modelar la asimetría y el efecto apalancamiento (*leverage effect*) de los rendimientos negativos.

### 3. Deep Learning (Redes Neuronales Recurrentes)

* **LSTM (Long Short-Term Memory)**: Arquitectura secuencial diseñada para capturar dependencias temporales a largo plazo sin el problema del desvanecimiento del gradiente.
  * **Configuración:** 128 neuronas, función de activación `tanh`, optimizador `Adam`, función de pérdida `MSE`, 20 épocas y `batch_size = 10`.
  * **Estrategia de Ventana Deslizante (*Rolling Window*):** Optimización computacional y evaluación fuera de muestra (*out-of-sample*).

---

## 📈 Métricas de Evaluación y Resultados

Para evaluar la precisión fuera de muestra se emplearon las métricas **RMSE** (Root Mean Square Error) y **MAPE** (Mean Absolute Percentage Error).

### Conclusiones Principales:

1. **Dominio de LSTM:** La red LSTM superó consistentemente al modelo GARCH en todos los horizontes temporales (diario, semanal y mensual), logrando reducciones del RMSE de aproximadamente un **30% a 50%**, y del MAPE de hasta un **40%**.
2. **Volatilidad Semanal:** El modelo GARCH mostró un desempeño sólido y computacionalmente eficiente en la volatilidad semanal, siendo una alternativa económica frente a las redes profundas.
3. **Efecto de Asimetría (EGARCH):** Para activos con caídas abruptas de precio y alta volatilidad (ej. Meta o Tesla), el modelo EGARCH capturó de forma más precisa los picos de volatilidad al considerar la asimetría del mercado.

---

## 📁 Estructura del Repositorio

```text
.
├── notebooks/
│   ├── TFG_Raul_Mendez_EDA.ipynb               # Análisis Exploratorio de Datos (EDA) y tests estadísticos
│   ├── TFG_Raul_Mendez_GARCH_vs_EGARCH.ipynb   # Evaluación y comparación de modelos GARCH y EGARCH
│   ├── TFG_Raul_Mendez_LSTM.ipynb              # Desarrollo, entrenamiento y tuning de la red LSTM
│   └── TFG_Raul_Mendez_GARCH_vs_LSTM.ipynb     # Comparativa final de métricas (RMSE, MAPE) e inferencia
├── docs/
│   └── TFG_Raul_Mendez_ADE.pdf                 # Memoria completa del Trabajo Fin de Grado
├── requirements.txt                            # Librerías y dependencias necesarias
├── LICENSE                                     # Licencia de uso MIT
└── README.md                                   # Documentación principal
```

---



## 🛠️ Instalación y Requisitos

Para ejecutar los notebooks localmente o en entornos de nube como Google Colab:

1. **Clonar el repositorio:**

   ```bash
   git clone [https://github.com/TuUsuario/TFG-Volatilidad-GARCH-vs-LSTM.git](https://github.com/TuUsuario/TFG-Volatilidad-GARCH-vs-LSTM.git)
   cd tfg-garch-vs-lstm
   ```
2. **Instalar dependencias:**

   ```bash
   pip install -r requirements.txt
   ```

### Principales Librerías Utilizadas:

* `python` >= 3.8
* `yfinance` — Descarga de datos bursátiles
* `arch` — Modelización econométrica GARCH/EGARCH
* `tensorflow` / `keras` — Redes neuronales LSTM
* `scikit-learn` — Métricas de evaluación (RMSE, MAPE)
* `pandas`, `numpy`, `matplotlib` — Tratamiento de datos y visualización

## ✒️ Autor y Contacto

* **Raúl Méndez López**
* **Titulación:** Grado en Administración y Dirección de Empresas (ADE) — Universitat Politècnica de València (UPV)
* **LinkedIn:** [www.linkedin.com/in/raumenlo](https://www.linkedin.com/in/raumenlo/)
* **Email:** raumenlo26@gmail.com
