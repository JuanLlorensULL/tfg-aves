# TFG — Predicción de Desplazamientos de Aves

Sistema híbrido de predicción de la posición diaria de gaviotas *Larus fuscus*
a partir de datos GPS, combinando **cadenas de Markov**, **modelos ocultos de Markov (HMM)**
y **algoritmos de Machine Learning** (Random Forest, XGBoost, LightGBM), con un módulo
de visualización interactiva sobre mapas.

> Trabajo Fin de Grado — Facultad de Informática, UPM. Oferta 9519.
> Autor: Juan Llorens. Tutora: Adriana Toni Delgado.

## Aplicación interactiva

El fichero `reports/figures/o5_prediccion_app.html` es una aplicación de predicción interactiva que permite seleccionar una gaviota y una fecha y visualizar sobre el mapa la posición predicha, el intervalo de incertidumbre y el error cometido. Descárgalo y ábrelo en un navegador.

## Estructura del proyecto

```
.
├── data/
│   ├── raw/         # CSV original Movebank (NO en git, ver "Obtención de datos")
│   ├── interim/     # Datos intermedios (limpieza, secuencias diarias)
│   └── processed/   # Datasets finales listos para modelar
├── notebooks/       # Exploración (EDA) y prototipado, sincronizados con jupytext
├── src/tfg_aves/
│   ├── data/        # Limpieza y secuenciación
│   ├── markov/      # Cadenas de Markov visibles
│   ├── hmm/         # Detección de estados de comportamiento
│   ├── ml/          # Modelos supervisados (RF / XGBoost / LightGBM)
│   └── viz/         # Mapas interactivos y análisis de error
├── tests/           # pytest
├── reports/         # Figuras, tablas y resultados
└── pyproject.toml   # Definición del proyecto (uv)
```

## Requisitos

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) como gestor de dependencias

## Instalación

```bash
git clone https://github.com/JuanLlorensULL/tfg-aves.git tfg-aves && cd tfg-aves
uv sync
```

## Obtención de datos

El dataset original proviene de **Movebank**, estudio
*"Navigation experiments in lesser black-backed gulls (data from Wikelski et al. 2015)"*.
Descárgalo y colócalo en `data/raw/migration_original.csv`.

| | |
|---|---|
| Especie | *Larus fuscus* |
| Individuos | 126 |
| Registros | 89.867 GPS |
| Formato | CSV Movebank estándar |

## Uso

```bash
uv run pytest          # ejecutar tests
uv run jupyter lab     # abrir cuadernos
uv run ruff check src tests  # linter
```

## Objetivos

| | Objetivo | Horas |
|---|---|---|
| 1 | Preparación y limpieza de datos | 48 |
| 2 | Predicción estadística con Markov | 40 |
| 3 | Detección de comportamiento con HMM | 50 |
| 4 | Predicción con Machine Learning (RF, XGBoost, LightGBM) | 65 |
| 5 | Visualización en mapas y evaluación | 44 |
|   | Escritura de la memoria | 40 |
|   | Preparación de la defensa | 10 |
|   | **Total** | **297** |

## Licencia

MIT.
