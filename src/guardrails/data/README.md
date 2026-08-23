# `guardrails.data` — ingesta, traducción y construcción de datasets

Produce el parquet de fine-tuning a partir de cuatro datasets de Hugging Face.

| Módulo | Qué hace |
|---|---|
| `ingest.py` | Descarga un dataset de Hugging Face a `data/<id>/<split>.parquet` |
| `translate.py` | Traduce al español con `facebook/nllb-200-distilled-600M` |
| `build_dataset.py` | Une los orígenes, etiqueta, muta, deduplica, particiona y balancea |
| `corrective.py` | Construye el dataset correctivo a partir de predicciones previas |

## Orígenes

| Dataset | Idioma | Columnas de texto |
|---|---|---|
| `huyhoangdinhcong/guardrails-dataset-full` | vi | `text` |
| `Necent/llm-jailbreak-prompt-injection-dataset` | mixto | `prompt`, `response` |
| `J1N2/mix-prompt-injection-dataset` | en | `prompt`, `base_prompt` |
| `bogdanminko/Catch_the_prompt_injection_or_jailbreak_or_benign` | en | `prompt` |

Cada origen tiene su propia función de etiquetado (`labels_*`), que traduce sus columnas al
esquema común de ocho etiquetas booleanas.

## Esquema de salida

23 columnas, definidas en `base_record()`. Las principales:

| Columna | Contenido |
|---|---|
| `id`, `parent_id` | Identificador SHA-256 truncado y, en mutaciones, el del registro origen |
| `source_dataset`, `source_split`, `source_row`, `text_role` | Procedencia |
| `original_text`, `text_es` | Texto de origen y su traducción |
| `mutation_type` | `translation` o `mutation_es_N` |
| `split` | `train` / `validation` / `test` |
| `primary_label`, `decision`, `target_json` | Etiqueta resuelta y objetivo de entrenamiento |
| `sft_text` | Secuencia completa que consume el entrenador |
| `label_*` (8) | Etiquetas booleanas |

## Uso

```bash
uv run guardrails data ingest --dataset-id J1N2/mix-prompt-injection-dataset
uv run guardrails data translate --profile finetune
uv run guardrails data build --output data_finetune/guardrail_es.parquet
```

Cada comando acepta `--help`. Los parámetros de las ejecuciones históricas están en
`configs/data/`.

## Tests

```bash
uv run pytest tests/test_splits.py tests/test_corrective_dataset.py tests/test_taxonomy.py
```
