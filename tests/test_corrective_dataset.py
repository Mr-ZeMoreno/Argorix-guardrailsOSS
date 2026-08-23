"""Composición del dataset correctivo.

``guardrails.data.corrective`` construye su salida a partir de las predicciones
de una evaluación previa sobre el golden set. Estos tests reconstruyen esa
salida de forma determinista, sin necesidad del parquet original, y verifican
su composición.

La reconstrucción se contrasta primero contra los contadores publicados en
``guardrail_es_v3_corrective.summary.json``; el resto de tests sólo son válidos
si esa comprobación pasa.
"""

from __future__ import annotations

import collections
import csv

import pytest

from guardrails.data import corrective as corr
from guardrails.evaluation import golden_set

# Contadores publicados en
# hf_publish/dataset-governance-ai-guardrail-es/guardrail_es_v3_corrective.summary.json
PUBLICADO = {
    "corrective_rows": 18_104,
    "golden_v2_hard_negatives": 17_700,
    "golden_v2_contrastive_attacks": 404,
}


@pytest.fixture(scope="module")
def golden(tmp_path_factory: pytest.TempPathFactory) -> list[dict[str, str]]:
    """Regenera eval_golden_es.csv en un directorio temporal."""
    destino = tmp_path_factory.mktemp("golden")
    original = golden_set.OUTPUT
    golden_set.OUTPUT = destino / "eval_golden_es.csv"
    try:
        golden_set.main()
        with golden_set.OUTPUT.open(encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    finally:
        golden_set.OUTPUT = original


@pytest.fixture(scope="module")
def correctivo(golden: list[dict[str, str]]) -> list[dict]:
    """Reconstruye la porción correctiva del dataset."""
    safe_rows = [r for r in golden if r["expected_decision"] == "ALLOW"]
    block_rows = [r for r in golden if r["expected_decision"] == "BLOCK"]

    registros: list[dict] = []
    for index, item in enumerate(safe_rows):
        for vi, variante in enumerate(corr.variants_for_safe(item["text"])):
            registros.append(
                corr.make_record(variante, "SAFE", "golden_v2_hard_negatives", index, f"safe_variant_0_{vi}")
            )
    for index, item in enumerate(block_rows):
        registros.append(
            corr.make_record(
                item["text"],
                item["expected_primary_label"],
                "golden_v2_contrastive_attacks",
                index,
                "attack",
            )
        )
        for ei, (_, ataque) in enumerate(corr.CONTRAST_ATTACKS):
            etiqueta = (
                "PROMPT_INJECTION"
                if "prompt" in ataque.lower() or "sistema" in ataque.lower()
                else "JAILBREAK"
            )
            registros.append(
                corr.make_record(ataque, etiqueta, "golden_v2_contrastive_attacks", index, f"contrast_{ei}")
            )

    vistos: set[tuple[str, str]] = set()
    unicos: list[dict] = []
    for registro in registros:
        clave = (registro["text_es"], registro["target_json"])
        if clave in vistos:
            continue
        vistos.add(clave)
        unicos.append(registro)
    return unicos


def test_la_reconstruccion_reproduce_los_contadores_publicados(correctivo: list[dict]) -> None:
    """Valida el modelo del pipeline contra tres contadores independientes."""
    por_fuente = collections.Counter(r["source_dataset"] for r in correctivo)
    assert len(correctivo) == PUBLICADO["corrective_rows"]
    assert por_fuente["golden_v2_hard_negatives"] == PUBLICADO["golden_v2_hard_negatives"]
    assert por_fuente["golden_v2_contrastive_attacks"] == PUBLICADO["golden_v2_contrastive_attacks"]


@pytest.mark.characterization
def test_variants_for_safe_incluye_el_texto_de_entrada(correctivo: list[dict]) -> None:
    """La primera entrada del conjunto de variantes es el texto sin modificar."""
    texto = "Explica pan amasado chileno, sin instrucciones peligrosas."
    variantes = corr.variants_for_safe(texto)
    assert texto in variantes
    assert len(variantes) > 20


@pytest.mark.characterization
def test_composicion_respecto_del_conjunto_de_entrada(
    golden: list[dict[str, str]], correctivo: list[dict]
) -> None:
    """Relación entre las filas de entrada y las del dataset resultante."""
    indice = {r["text_es"]: r for r in correctivo}
    textos_entrada = {r["text"] for r in golden}

    presentes = [t for t in textos_entrada if t in indice]
    en_train = [t for t in presentes if indice[t]["split"] == "train"]

    assert len(textos_entrada) == 1000
    assert len(presentes) == 1000
    assert len(en_train) == 871

    etiqueta_entrada = {r["text"]: r["expected_primary_label"] for r in golden}
    distintas = [t for t in presentes if indice[t]["primary_label"] != etiqueta_entrada[t]]
    assert distintas == []


@pytest.mark.characterization
def test_variantes_generadas_por_cada_fila_de_entrada(
    golden: list[dict[str, str]], correctivo: list[dict]
) -> None:
    """Número de variantes que cada fila ALLOW aporta a la partición train."""
    por_fila = collections.Counter(
        r["source_row"]
        for r in correctivo
        if r["source_dataset"] == "golden_v2_hard_negatives" and r["split"] == "train"
    )
    allow = [r for r in golden if r["expected_decision"] == "ALLOW"]
    assert len(allow) == 600
    assert len(por_fila) == 600
    assert min(por_fila.values()) >= 10
