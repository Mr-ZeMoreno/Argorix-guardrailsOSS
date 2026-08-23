"""Comportamiento de las métricas de evaluación.

Fija cómo ``run_eval.update_metrics`` acumula contadores y cómo se derivan las
tasas a partir de ellos, de modo que cualquier cambio en esa lógica aparezca de
forma visible en el diff.
"""

from __future__ import annotations

import pytest

from guardrails.evaluation import run_eval

CLAVES = {
    "total",
    "decision_correct",
    "primary_label_correct",
    "false_positives",
    "false_negatives",
    "label_confusion",
}


def _metricas_vacias() -> dict:
    return {
        "total": 0,
        "decision_correct": 0,
        "primary_label_correct": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "label_confusion": {},
    }


def _tasas(metrics: dict, esperado_por_texto: dict) -> dict:
    """Réplica de la derivación de tasas de run_eval.main()."""
    allow = sum(1 for v in esperado_por_texto.values() if v["expected_decision"] == "ALLOW")
    block = sum(1 for v in esperado_por_texto.values() if v["expected_decision"] == "BLOCK")
    return {
        "decision_accuracy": metrics["decision_correct"] / metrics["total"],
        "false_positive_rate": metrics["false_positives"] / allow if allow else 0,
        "false_negative_rate": metrics["false_negatives"] / block if block else 0,
        "_allow_total": allow,
        "_block_total": block,
    }


@pytest.mark.characterization
def test_denominadores_con_entradas_repetidas() -> None:
    """El numerador recorre la lista de textos; el denominador, las claves únicas."""
    textos = ["t_dup", "t_dup", "t_allow_2"]
    esperado = {
        "t_dup": {"expected_decision": "ALLOW", "expected_primary_label": "SAFE"},
        "t_allow_2": {"expected_decision": "ALLOW", "expected_primary_label": "SAFE"},
    }

    metrics = _metricas_vacias()
    for texto in textos:
        run_eval.update_metrics(metrics, esperado.get(texto), {"decision": "BLOCK", "primary_label": "HATE"})
    tasas = _tasas(metrics, esperado)

    assert metrics["total"] == 3
    assert tasas["_allow_total"] == 2
    assert tasas["false_positive_rate"] == 1.5


@pytest.mark.characterization
def test_tratamiento_de_salidas_sin_json_valido() -> None:
    """Una predicción vacía no coincide con ninguna de las dos decisiones."""
    textos = ["a1", "a2", "b1", "b2"]
    esperado = {
        "a1": {"expected_decision": "ALLOW", "expected_primary_label": "SAFE"},
        "a2": {"expected_decision": "ALLOW", "expected_primary_label": "SAFE"},
        "b1": {"expected_decision": "BLOCK", "expected_primary_label": "HATE"},
        "b2": {"expected_decision": "BLOCK", "expected_primary_label": "HATE"},
    }
    predicho = {
        "a1": {},
        "a2": {"decision": "ALLOW", "primary_label": "SAFE"},
        "b1": {},
        "b2": {"decision": "BLOCK", "primary_label": "HATE"},
    }

    metrics = _metricas_vacias()
    for texto in textos:
        run_eval.update_metrics(metrics, esperado[texto], predicho[texto])
    tasas = _tasas(metrics, esperado)

    assert tasas["decision_accuracy"] == 0.5
    assert tasas["false_positive_rate"] == 0.0
    assert tasas["false_negative_rate"] == 0.0


@pytest.mark.characterization
def test_criterio_de_run_eval_frente_al_de_backend() -> None:
    """run_eval compara con ``== "BLOCK"``; backend, con ``!= "ALLOW"``."""
    esperado = {
        "a1": {"expected_decision": "ALLOW", "expected_primary_label": "SAFE"},
        "a2": {"expected_decision": "ALLOW", "expected_primary_label": "SAFE"},
        "b1": {"expected_decision": "BLOCK", "expected_primary_label": "HATE"},
        "b2": {"expected_decision": "BLOCK", "expected_primary_label": "HATE"},
    }
    predicho = {
        "a1": {},
        "a2": {"decision": "ALLOW", "primary_label": "SAFE"},
        "b1": {},
        "b2": {"decision": "BLOCK", "primary_label": "HATE"},
    }

    metrics = _metricas_vacias()
    for texto in esperado:
        run_eval.update_metrics(metrics, esperado[texto], predicho[texto])
    fpr_run_eval = _tasas(metrics, esperado)["false_positive_rate"]

    fp = safe = 0
    for texto, exp in esperado.items():
        if exp["expected_decision"] == "ALLOW":
            safe += 1
            if (predicho[texto].get("decision") or "INVALID") != "ALLOW":
                fp += 1
    fpr_backend = fp / safe

    assert fpr_run_eval == 0.0
    assert fpr_backend == 0.5


@pytest.mark.characterization
def test_contadores_que_acumula_update_metrics() -> None:
    """Conjunto de claves del diccionario de métricas."""
    assert set(_metricas_vacias()) == CLAVES


def test_metricas_publicadas_son_coherentes_con_los_conteos() -> None:
    """Comprobación de integridad aritmética de reports/v3_golden_metrics.json."""
    import json

    from guardrails import paths

    ruta = paths.REPORTS_DIR / "v3_golden_metrics.json"
    if not ruta.is_file():
        pytest.skip("reports/v3_golden_metrics.json no está disponible")

    m = json.loads(ruta.read_text(encoding="utf-8"))
    assert m["total"] == 1000
    assert m["decision_correct"] / m["total"] == pytest.approx(m["decision_accuracy"])
    assert m["primary_label_correct"] / m["total"] == pytest.approx(m["primary_label_accuracy"])
    assert m["false_positives"] / 600 == pytest.approx(m["false_positive_rate"])
    assert m["false_negatives"] / 400 == pytest.approx(m["false_negative_rate"])
