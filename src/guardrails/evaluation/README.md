# `guardrails.evaluation` — golden set, inferencia y métricas

| Módulo | Qué hace |
|---|---|
| `golden_set.py` | Genera `eval_golden_es.csv` de forma determinista a partir de listas literales |
| `run_eval.py` | Infiere sobre cada fila del CSV y calcula las métricas |

## Conjunto de evaluación

`golden_set.py` produce 1.000 filas sin aleatoriedad: 600 `ALLOW` combinando tópicos benignos con
verbos y contextos, y 400 `BLOCK` envolviendo frases semilla de ataque en plantillas. Es
reproducible bit a bit.

Columnas: `id`, `text`, `expected_decision`, `expected_primary_label`, `expected_labels`,
`category`.

## Métricas

`run_eval.py` emite cuatro escalares y una tabla de confusión de etiqueta:

| Métrica | Orientación |
|---|---|
| `decision_accuracy` | mayor mejor |
| `primary_label_accuracy` | mayor mejor |
| `false_positive_rate` | menor mejor |
| `false_negative_rate` | menor mejor |
| `label_confusion` | diccionario `"esperada->predicha" -> conteo` |

La generación es voraz (`do_sample=False`), de modo que dado un adaptador el resultado es
determinista.

## Uso

```bash
uv run guardrails eval golden-set
uv run guardrails eval run \
  --adapter models/guardrail-qwen25-1_5b-qlora-v3-corrective \
  --input-csv eval_golden_es.csv \
  --output-jsonl reports/predictions.jsonl \
  --metrics-output reports/metrics.json
```

`--load-4bit` carga el modelo base cuantizado, igual que en entrenamiento y serving. Sin esa
bandera se carga sin cuantizar.

## Tests

```bash
uv run pytest tests/test_golden_set.py tests/test_cost_metrics.py
```
