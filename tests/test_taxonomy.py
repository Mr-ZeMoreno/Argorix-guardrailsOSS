"""Resolución de etiquetas multi-clase y órdenes de prioridad."""

from __future__ import annotations

import pytest

from guardrails.data import build_dataset as bd
from guardrails.serving import backend

# Orden implícito en build_dataset.primary_label() (líneas 103-118).
ORDEN_ENTRENAMIENTO = [
    "PROMPT_INJECTION",
    "JAILBREAK",
    "SEXUAL",
    "VIOLENCE",
    "HATE",
    "POLITICS",
    "HARMFUL",
    "SAFE",
]


def _etiquetas(**activas: bool) -> dict[str, bool]:
    labels = dict.fromkeys(bd.LABEL_COLUMNS, False)
    labels.update({f"label_{k}": v for k, v in activas.items()})
    return labels


def test_prioridad_de_entrenamiento_es_la_documentada() -> None:
    for i, etiqueta in enumerate(ORDEN_ENTRENAMIENTO[:-1]):
        columna = f"label_{etiqueta.lower()}"
        for inferior in ORDEN_ENTRENAMIENTO[i + 1 : -1]:
            labels = _etiquetas()
            labels[columna] = True
            labels[f"label_{inferior.lower()}"] = True
            assert bd.primary_label(labels) == etiqueta


@pytest.mark.characterization
def test_pares_con_orden_distinto_entre_datos_y_runtime() -> None:
    """Comparación entre el orden de ``primary_label`` y el de ``LABEL_PRIORITY``.

    El primero resuelve la etiqueta de los datos; el segundo arbitra en runtime
    entre la variante original y la normalizada.
    """
    invertidos = [
        (a, b)
        for i, a in enumerate(ORDEN_ENTRENAMIENTO)
        for b in ORDEN_ENTRENAMIENTO[i + 1 :]
        if backend.LABEL_PRIORITY.get(a, 0) < backend.LABEL_PRIORITY.get(b, 0)
    ]
    assert len(invertidos) == 9
    assert ("PROMPT_INJECTION", "VIOLENCE") in invertidos
    assert ("JAILBREAK", "HARMFUL") in invertidos


@pytest.mark.characterization
def test_resolucion_de_harmful_con_jailbreak() -> None:
    """Un vector con harmful y jailbreak activos se resuelve como JAILBREAK."""
    assert bd.primary_label(_etiquetas(harmful=True, jailbreak=True)) == "JAILBREAK"
    assert bd.primary_label(_etiquetas(harmful=True)) == "HARMFUL"


@pytest.mark.characterization
def test_derivacion_de_label_harmful() -> None:
    """``label_harmful`` se deriva de sexual, violence, hate y politics.

    En consecuencia, una fila marcada POLITICS resuelve a BLOCK.
    """
    import pandas as pd

    fila = pd.Series({"POLITICS": 1, "HARMFULNESS": 0, "SEXUAL": 0, "VIOLENCE": 0, "HATE SPEECH": 0})
    etiquetas = bd.labels_guardrails(fila)
    assert etiquetas["label_politics"] is True
    assert etiquetas["label_harmful"] is True
    assert bd.decision_for(etiquetas) == "BLOCK"
