"""Persistencia de figuras y tablas listas para la memoria.

Punto de entrada único: :func:`save_artifact`. Guarda una figura y/o una
tabla junto a un *caption* en castellano y registra la entrada en el índice
central, siguiendo una nomenclatura determinista que permite citar cualquier
decisión del TFG desde la memoria.

Esquema de ficheros:
    reports/figures/{objetivo}_fig{NN}_{slug}.{ext}
    reports/tables/{objetivo}_tab{NN}_{slug}.csv
    reports/captions/{objetivo}_fig{NN}_{slug}.md

Donde ``objetivo`` es uno de ``o1``..``o5`` y ``NN`` es un número de dos
dígitos.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import matplotlib.figure
    import pandas as pd

ProjectObjective = Literal["o1", "o2", "o3", "o4", "o5"]
FigureFormat = Literal["png", "pdf", "both"]

_VALID_OBJECTIVES: frozenset[str] = frozenset({"o1", "o2", "o3", "o4", "o5"})
_SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_INDEX_HEADER = (
    "# Índice de figuras y tablas — TFG\n\n"
    "Generado automáticamente por `tfg_aves.reporting.save_artifact`. "
    "Cada fila documenta una decisión justificada con datos.\n\n"
    "| Objetivo | Ref | Slug | Decisión | Figura | Tabla | Caption | Fecha |\n"
    "|---|---|---|---|---|---|---|---|\n"
)


def _project_root() -> Path:
    """Devuelve la raíz del proyecto buscando el ``pyproject.toml``."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError(
        "No se localizó la raíz del proyecto (no hay pyproject.toml en "
        f"ningún ancestro de {here})."
    )


@dataclass(frozen=True)
class ArtifactPaths:
    """Rutas generadas por :func:`save_artifact`. ``None`` si no se produce."""

    figure: Path | None
    figure_pdf: Path | None
    table: Path | None
    caption: Path | None
    ref: str  # p.ej. "o1_fig03_gap-distribution"


def _validate_slug(slug: str) -> None:
    if not _SLUG_RE.match(slug):
        raise ValueError(
            f"slug {slug!r} debe ser kebab-case en minúsculas "
            "(letras, dígitos y guiones; sin guion inicial ni final)."
        )


def _validate_objective(objective: str) -> None:
    if objective not in _VALID_OBJECTIVES:
        raise ValueError(
            f"objective {objective!r} debe ser uno de {sorted(_VALID_OBJECTIVES)}."
        )


def _ref(objective: str, num: int, slug: str, kind: Literal["fig", "tab"]) -> str:
    if not 1 <= num <= 99:
        raise ValueError(f"num debe estar en [1, 99]; recibido {num}.")
    return f"{objective}_{kind}{num:02d}_{slug}"


def _write_caption(
    path: Path,
    *,
    ref: str,
    objective: str,
    decision: str,
    caption_es: str,
    figure_rel: str | None,
    table_rel: str | None,
) -> None:
    body = [
        f"# {ref}",
        "",
        f"- **Objetivo:** {objective.upper()}",
        f"- **Decisión justificada:** {decision}",
    ]
    if figure_rel is not None:
        body.append(f"- **Figura:** `{figure_rel}`")
    if table_rel is not None:
        body.append(f"- **Tabla:** `{table_rel}`")
    body.extend(["", "## Caption (memoria)", "", caption_es.strip(), ""])
    path.write_text("\n".join(body), encoding="utf-8")


def _append_index_row(
    index_path: Path,
    *,
    objective: str,
    ref: str,
    slug: str,
    decision: str,
    figure_rel: str | None,
    table_rel: str | None,
    caption_rel: str,
) -> None:
    """Añade o actualiza una fila en ``INDEX.md`` (idempotente por ``ref``).

    Si ya existe una fila con el mismo ``(objetivo, ref)``, se sustituye in
    situ para reflejar la fecha más reciente y los captions actualizados.
    Si no existe, se appendea al final.
    """
    if not index_path.exists():
        index_path.write_text(_INDEX_HEADER, encoding="utf-8")
    today = _dt.date.today().isoformat()
    fig_cell = f"`{figure_rel}`" if figure_rel else "—"
    tab_cell = f"`{table_rel}`" if table_rel else "—"
    row = (
        f"| {objective.upper()} | {ref} | {slug} | {decision.replace('|', '\\|')} | "
        f"{fig_cell} | {tab_cell} | `{caption_rel}` | {today} |\n"
    )

    content = index_path.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)

    row_prefix = f"| {objective.upper()} | {ref} |"
    found_idx: int | None = None
    for i, line in enumerate(lines):
        if line.startswith(row_prefix):
            found_idx = i
            break

    if found_idx is not None:
        lines[found_idx] = row
    else:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] = lines[-1] + "\n"
        lines.append(row)

    index_path.write_text("".join(lines), encoding="utf-8")


def save_artifact(
    slug: str,
    *,
    objective: ProjectObjective,
    num: int,
    decision: str,
    caption_es: str,
    fig: matplotlib.figure.Figure | None = None,
    table: pd.DataFrame | None = None,
    fig_format: FigureFormat = "png",
    dpi: int = 200,
    overwrite: bool = False,
    project_root: Path | None = None,
) -> ArtifactPaths:
    """Persiste una figura y/o una tabla para la memoria.

    Debe proporcionarse al menos uno de ``fig`` o ``table``.

    Args:
        slug: Identificador kebab-case, p.ej. ``"gap-distribution"``.
        objective: Uno de ``"o1"``..``"o5"``.
        num: Número de la figura/tabla dentro del objetivo, ``1..99``.
        decision: Descripción breve en castellano de la decisión justificada.
        caption_es: Párrafo de pie de figura listo para la memoria (castellano).
        fig: Figura de Matplotlib. Se guarda en PNG (y/o PDF) a ``dpi``.
        table: DataFrame de pandas. Se guarda en CSV UTF-8 sin índice.
        fig_format: ``"png"``, ``"pdf"`` o ``"both"``.
        dpi: Resolución del PNG en puntos por pulgada.
        overwrite: Si es ``False`` (defecto), lanza ``FileExistsError`` ante
            colisión de nombre.
        project_root: Permite forzar la raíz del proyecto (principalmente
            para tests).

    Returns:
        ArtifactPaths con las rutas absolutas de los artefactos generados.
    """
    _validate_slug(slug)
    _validate_objective(objective)
    if not decision.strip():
        raise ValueError("decision debe ser una descripción no vacía.")
    if not caption_es.strip():
        raise ValueError("caption_es debe ser un párrafo no vacío.")
    if fig is None and table is None:
        raise ValueError("Hay que proporcionar al menos `fig` o `table`.")

    root = project_root if project_root is not None else _project_root()
    figures_dir = root / "reports" / "figures"
    tables_dir = root / "reports" / "tables"
    captions_dir = root / "reports" / "captions"
    for d in (figures_dir, tables_dir, captions_dir):
        d.mkdir(parents=True, exist_ok=True)

    ref = _ref(objective, num, slug, "fig" if fig is not None else "tab")
    tab_ref = _ref(objective, num, slug, "tab")

    figure_path: Path | None = None
    figure_pdf_path: Path | None = None
    table_path: Path | None = None

    if fig is not None:
        if fig_format in ("png", "both"):
            figure_path = figures_dir / f"{ref}.png"
            if figure_path.exists() and not overwrite:
                raise FileExistsError(
                    f"{figure_path} ya existe. Usa otro num/slug o "
                    "pasa overwrite=True."
                )
            fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
        if fig_format in ("pdf", "both"):
            figure_pdf_path = figures_dir / f"{ref}.pdf"
            if figure_pdf_path.exists() and not overwrite:
                raise FileExistsError(
                    f"{figure_pdf_path} ya existe. Usa otro num/slug o "
                    "pasa overwrite=True."
                )
            fig.savefig(figure_pdf_path, bbox_inches="tight")

    if table is not None:
        table_path = tables_dir / f"{tab_ref}.csv"
        if table_path.exists() and not overwrite:
            raise FileExistsError(
                f"{table_path} ya existe. Usa otro num/slug o "
                "pasa overwrite=True."
            )
        table.to_csv(table_path, index=False, encoding="utf-8")

    caption_ref = ref
    caption_path = captions_dir / f"{caption_ref}.md"
    if caption_path.exists() and not overwrite:
        raise FileExistsError(
            f"{caption_path} ya existe. Usa otro num/slug o "
            "pasa overwrite=True."
        )

    figure_rel = (
        str(figure_path.relative_to(root)) if figure_path is not None else None
    )
    table_rel = (
        str(table_path.relative_to(root)) if table_path is not None else None
    )
    _write_caption(
        caption_path,
        ref=caption_ref,
        objective=objective,
        decision=decision,
        caption_es=caption_es,
        figure_rel=figure_rel,
        table_rel=table_rel,
    )

    _append_index_row(
        root / "reports" / "INDEX.md",
        objective=objective,
        ref=caption_ref,
        slug=slug,
        decision=decision,
        figure_rel=figure_rel,
        table_rel=table_rel,
        caption_rel=str(caption_path.relative_to(root)),
    )

    return ArtifactPaths(
        figure=figure_path,
        figure_pdf=figure_pdf_path,
        table=table_path,
        caption=caption_path,
        ref=caption_ref,
    )
