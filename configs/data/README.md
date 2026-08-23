# `configs/data/` — parámetros de construcción de datasets

| Archivo | Dataset |
|---|---|
| `pipeline_v1.yaml` | `data_finetune/guardrail_es.parquet` |
| `pipeline_v2.yaml` | `data_finetune/guardrail_es_v2.parquet` |

## Estado: registro declarativo

Mismo criterio que `configs/train/`: registran los parámetros de las ejecuciones históricas y no
son leídos por el pipeline.

El bloque `splits:` es **descriptivo**. Las proporciones están codificadas en
`guardrails/data/build_dataset.py` y `guardrails/data/corrective.py`, y no aceptan semilla ni
parámetro: no se pueden cambiar desde el YAML.
