# `guardrails.training` — fine-tuning QLoRA

| Módulo | Qué hace |
|---|---|
| `qlora.py` | Ajusta un adaptador LoRA sobre el campo `sft_text` del parquet |

## Configuración

| Aspecto | Valor |
|---|---|
| Modelo base | `Qwen/Qwen2.5-1.5B-Instruct` |
| Cuantización | 4-bit NF4 con doble cuantización |
| LoRA | `r=16`, `alpha=32`, `dropout=0.05`, sobre las siete proyecciones |
| Optimizador | `paged_adamw_8bit`, planificador coseno, `warmup_ratio=0.03` |
| Longitud | 512 tokens |
| Semilla | 42 |

`SaveStateCallback` escribe `train_state.json` en el directorio de salida en cada log, para que la
consola de gobernanza pueda mostrar el progreso.

## Requisitos

GPU NVIDIA con CUDA, en **Linux o Windows**. El módulo aborta explícitamente si `torch.cuda` no
está disponible.

```bash
uv sync --extra train
uv run guardrails doctor          # confirma que torch ve la GPU
uv run guardrails train qlora --help
```

## Uso

```bash
uv run guardrails train qlora \
  --dataset data_finetune/guardrail_es.parquet \
  --output-dir models/guardrail-qwen25-1_5b-qlora-v1 \
  --max-steps 3000 --bf16
```

`--adapter-init` continúa el entrenamiento desde un adaptador existente en lugar de partir de
cero. Los parámetros de las ejecuciones históricas están en `configs/train/`, y los lanzadores
equivalentes en `scripts/linux/` y `scripts/windows/`.

## Nota sobre la plantilla de prompt

`sft_text` usa delimitadores `<start_of_turn>` / `<end_of_turn>`. El tokenizador de Qwen no los
trata como tokens especiales, así que se tokenizan como texto ordinario. La plantilla está
duplicada en cuatro módulos (`data/build_dataset.py`, `data/corrective.py`,
`evaluation/run_eval.py`, `serving/worker.py`) y debe mantenerse sincronizada a mano.
