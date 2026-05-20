# TFG — Predicción de Desplazamientos de Aves

Sistema híbrido de predicción de la posición diaria de gaviotas *Larus fuscus*
a partir de datos GPS, combinando **cadenas de Markov**, **modelos ocultos de Markov (HMM)**
y **algoritmos de Machine Learning** (Random Forest, XGBoost, LightGBM), con un módulo
de visualización interactiva sobre mapas.

> Trabajo Fin de Grado — Facultad de Informática, UPM. Oferta 9519.
> Tutora: Adriana Toni Delgado (atoni@fi.upm.es).

## Estructura del proyecto

```
.
├── data/
│   ├── raw/         # CSV original Movebank (NO en git, ver "Obtención de datos")
│   ├── interim/     # Datos intermedios (limpieza, secuencias diarias)
│   └── processed/   # Datasets finales listos para modelar
├── notebooks/       # Exploración (EDA) y prototipado, sincronizados con jupytext
├── src/tfg_aves/
│   ├── data/        # O1 — Limpieza y secuenciación
│   ├── markov/      # O2 — Cadenas de Markov visibles
│   ├── hmm/         # O3 — Detección de estados de comportamiento
│   ├── ml/          # O4 — Modelos supervisados (RF / XGBoost / LightGBM)
│   └── viz/         # O5 — Mapas interactivos y análisis de error
├── tests/           # pytest
├── reports/         # Memoria, figuras, resultados
└── pyproject.toml   # Definición del proyecto (uv)
```

## Requisitos

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) como gestor de dependencias

## Instalación

```bash
# Clonar el repositorio
git clone <url> tfg-aves && cd tfg-aves

# Crear entorno e instalar dependencias (uv lee pyproject.toml)
uv sync

# Activar el entorno (opcional — uv run lo hace automáticamente)
source .venv/bin/activate

# Registrar el kernel de Jupyter
python -m ipykernel install --user --name tfg-aves --display-name "TFG aves"
```

## Obtención de datos

El dataset original proviene de **Movebank**, estudio
*"Navigation experiments in lesser black-backed gulls (data from Wikelski et al. 2015)"*.
Descárgalo y colócalo en `data/raw/migration_original.csv`.

Resumen del dataset:

| | |
|---|---|
| Especie | *Larus fuscus* |
| Individuos | 126 |
| Registros | 89.867 GPS |
| Formato | CSV Movebank estándar |

## Convención de figuras, tablas y decisiones

Toda decisión no trivial de este TFG (umbrales, hiperparámetros, fronteras
estacionales, comparativas de modelos, etc.) **debe** ir respaldada por una
figura y/o una tabla, junto con un *caption* en castellano listo para pegar
en la memoria. Para garantizar nomenclatura uniforme y un índice centralizado,
usa siempre el helper `save_artifact`:

```python
from tfg_aves.reporting import save_artifact

save_artifact(
    "gap-distribution",
    objective="o1",
    num=3,
    decision="Umbral de gap temporal fijado en 2h.",
    caption_es=(
        "Distribución de los intervalos entre registros GPS consecutivos. "
        "El umbral de 2h se sitúa en el percentil 95, equilibrando "
        "continuidad de las secuencias diarias frente a pérdida de datos."
    ),
    fig=fig,                  # matplotlib Figure (opcional)
    table=df_summary,         # DataFrame (opcional)
    fig_format="png",         # "png" | "pdf" | "both"
)
```

El resultado se guarda en:
- `reports/figures/o1_fig03_gap-distribution.png`
- `reports/tables/o1_tab03_gap-distribution.csv`
- `reports/captions/o1_fig03_gap-distribution.md`

y añade automáticamente una fila al índice general `reports/INDEX.md`.

## Uso

```bash
# Ejecutar tests
uv run pytest

# Lanzar JupyterLab
uv run jupyter lab

# Linter
uv run ruff check src tests
```

## Objetivos del TFG

| | Objetivo | Horas |
|---|---|---|
| O1 | Preparación y limpieza de datos | 48 |
| O2 | Predicción estadística con Markov | 40 |
| O3 | Detección de comportamiento con HMM | 50 |
| O4 | Predicción con Machine Learning (RF, XGBoost, LightGBM) | 65 |
| O5 | Visualización en mapas y evaluación | 44 |
|   | Escritura de la memoria | 40 |
|   | Preparación de la defensa | 10 |
|   | **Total** | **297** |

## Licencia

MIT.
