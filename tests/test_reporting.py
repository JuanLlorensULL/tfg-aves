"""Tests de :mod:`tfg_aves.reporting`."""

from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tfg_aves.reporting import ArtifactPaths, save_artifact  # noqa: E402


@pytest.fixture
def project(tmp_path):
    """Crea una raíz de proyecto falsa con pyproject.toml y reports/."""
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (tmp_path / "reports" / "figures").mkdir(parents=True)
    (tmp_path / "reports" / "tables").mkdir(parents=True)
    (tmp_path / "reports" / "captions").mkdir(parents=True)
    return tmp_path


def _make_fig():
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    return fig


def test_saves_figure_table_caption_and_index(project):
    fig = _make_fig()
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    out = save_artifact(
        "gap-distribution",
        objective="o1",
        num=3,
        decision="Umbral de gap temporal fijado en 2h",
        caption_es=(
            "Distribución de los intervalos temporales entre registros GPS "
            "consecutivos. El umbral de 2h se sitúa en el percentil 95."
        ),
        fig=fig,
        table=df,
        project_root=project,
    )

    assert isinstance(out, ArtifactPaths)
    assert out.figure == project / "reports" / "figures" / "o1_fig03_gap-distribution.png"
    assert out.figure.is_file()
    assert out.figure_pdf is None
    assert out.table == project / "reports" / "tables" / "o1_tab03_gap-distribution.csv"
    assert out.table.is_file()
    assert out.caption == project / "reports" / "captions" / "o1_fig03_gap-distribution.md"
    caption_text = out.caption.read_text(encoding="utf-8")
    assert "O1" in caption_text
    assert "Umbral de gap temporal fijado en 2h" in caption_text
    assert "percentil 95" in caption_text

    index = (project / "reports" / "INDEX.md").read_text(encoding="utf-8")
    assert "gap-distribution" in index
    assert "Umbral de gap temporal fijado en 2h" in index
    assert "o1_fig03_gap-distribution" in index


def test_table_only_uses_tab_ref(project):
    df = pd.DataFrame({"x": [1]})
    out = save_artifact(
        "rmse-summary",
        objective="o4",
        num=1,
        decision="Comparativa RMSE entre modelos",
        caption_es="Tabla resumen del RMSE de los tres modelos ML evaluados.",
        table=df,
        project_root=project,
    )
    assert out.figure is None
    assert out.table.name == "o4_tab01_rmse-summary.csv"
    assert out.caption.name == "o4_tab01_rmse-summary.md"


def test_both_formats_produces_png_and_pdf(project):
    fig = _make_fig()
    out = save_artifact(
        "seasonal-pattern",
        objective="o2",
        num=2,
        decision="Definición de las cuatro estaciones meteorológicas",
        caption_es="Latitud media por mes mostrando los regímenes estacionales.",
        fig=fig,
        fig_format="both",
        project_root=project,
    )
    assert out.figure.suffix == ".png"
    assert out.figure_pdf.suffix == ".pdf"
    assert out.figure.is_file()
    assert out.figure_pdf.is_file()


def test_collision_raises_without_overwrite(project):
    fig = _make_fig()
    kwargs = dict(
        objective="o1",
        num=1,
        decision="x",
        caption_es="y",
        fig=fig,
        project_root=project,
    )
    save_artifact("dup", **kwargs)
    with pytest.raises(FileExistsError):
        save_artifact("dup", **kwargs)


def test_collision_overwrite_ok(project):
    fig = _make_fig()
    kwargs = dict(
        objective="o1",
        num=1,
        decision="x",
        caption_es="y",
        fig=fig,
        project_root=project,
    )
    save_artifact("dup", **kwargs)
    save_artifact("dup", overwrite=True, **kwargs)


@pytest.mark.parametrize(
    "bad_slug",
    ["Foo", "foo_bar", "-foo", "foo-", "foo--bar", "foo bar", ""],
)
def test_invalid_slug_rejected(project, bad_slug):
    with pytest.raises(ValueError, match="kebab-case"):
        save_artifact(
            bad_slug,
            objective="o1",
            num=1,
            decision="x",
            caption_es="y",
            table=pd.DataFrame({"a": [1]}),
            project_root=project,
        )


def test_invalid_objective_rejected(project):
    with pytest.raises(ValueError, match="objective"):
        save_artifact(
            "valid-slug",
            objective="o9",  # type: ignore[arg-type]
            num=1,
            decision="x",
            caption_es="y",
            table=pd.DataFrame({"a": [1]}),
            project_root=project,
        )


def test_empty_decision_rejected(project):
    with pytest.raises(ValueError, match="decision"):
        save_artifact(
            "valid",
            objective="o1",
            num=1,
            decision="   ",
            caption_es="y",
            table=pd.DataFrame({"a": [1]}),
            project_root=project,
        )


def test_missing_artefact_payload_rejected(project):
    with pytest.raises(ValueError, match="fig.*table"):
        save_artifact(
            "valid",
            objective="o1",
            num=1,
            decision="x",
            caption_es="y",
            project_root=project,
        )


def test_num_out_of_range_rejected(project):
    with pytest.raises(ValueError, match="num"):
        save_artifact(
            "valid",
            objective="o1",
            num=0,
            decision="x",
            caption_es="y",
            table=pd.DataFrame({"a": [1]}),
            project_root=project,
        )
