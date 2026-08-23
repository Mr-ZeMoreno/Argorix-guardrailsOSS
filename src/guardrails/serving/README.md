# `guardrails.serving` — consola de gobernanza y worker de inferencia

| Módulo | Qué hace |
|---|---|
| `backend.py` | API FastAPI y paneles de entrenamiento y evaluación |
| `worker.py` | Proceso persistente que carga el adaptador y responde por stdin/stdout |
| `static/` | Frontend estático |

```bash
uv sync --extra serve
uv run guardrails serve            # http://127.0.0.1:8000
```

## Endpoints

| Método | Ruta | Qué hace |
|---|---|---|
| `POST` | `/api/analyze` | Clasifica un texto |
| `POST` | `/api/analyze-batch` | Clasifica hasta 200 textos |
| `POST` | `/api/upload-json` | Clasifica los elementos de un JSON subido |
| `GET` | `/api/train/status` | Progreso del entrenamiento |
| `GET` | `/api/eval/golden-status` | Progreso y resultados de la evaluación |
| `GET` | `/api/health` | Estado del worker |

## Cómo procesa una petición

`analyze_text()` normaliza el texto de entrada con `normalize_for_guardrail()`, que revierte
ofuscaciones comunes: sustitución *leet* (`0`→`o`, `3`→`e`…), separadores insertados entre letras
y letras espaciadas. Si la normalización cambia el texto, se invoca al modelo una segunda vez con
la versión normalizada y `merge_results()` se queda con el resultado de mayor rango según
`LABEL_PRIORITY`.

El worker corre en un proceso aparte, con el modelo cargado en 4 bits, y se comunica por
stdin/stdout con JSON por línea. En Windows puede ejecutarse dentro de WSL:
`GUARDRAIL_WSL_PROJECT` fuerza el punto de montaje y `guardrails.paths.to_wsl_path()` traduce
`D:\…` a `/mnt/d/…`.

## Rutas

Este módulo deriva la raíz del proyecto de `guardrails.paths`, de modo que funciona igual desde
cualquier ubicación y en ambos sistemas operativos. `GUARDRAIL_PROJECT_ROOT` permite forzarla.

## Deuda de lint declarada

En `pyproject.toml`, bajo `[tool.ruff.lint.per-file-ignores]`, este archivo tiene suprimidas
`I001`, `B905` y `SIM105`, y está excluido del formateador. El motivo es mantenerlo idéntico al
original mientras no se reescriba; ver el README raíz.
