# `tests/` — batería de pruebas

```bash
uv run pytest                       # 44 tests, < 3 s, sin GPU ni red
uv run pytest -m characterization   # sólo los de caracterización
```

`conftest.py` instala sustitutos mínimos de las dependencias pesadas (`torch`, `transformers`,
`peft`, `trl`, `datasets`, `tqdm`, `fastapi`, `pydantic`) **sólo si no están instaladas**, de modo
que la batería corre con `uv sync --group dev` y también en una máquina con la pila completa.

## Tests de caracterización

Los marcados con `@pytest.mark.characterization` **fijan el comportamiento actual** de funciones
puras del pipeline: determinismo de la partición, derivación de etiquetas, acumulación de
contadores. No comprueban que ese comportamiento sea el deseado, sino que cualquier cambio en él
aparezca como un fallo visible en el diff en lugar de pasar inadvertido.

## Contenido

| Archivo | Qué cubre |
|---|---|
| `test_splits.py` | Determinismo y proporciones de la partición; efecto de la aumentación |
| `test_corrective_dataset.py` | Reconstrucción y composición del dataset correctivo |
| `test_golden_set.py` | Composición y reproducibilidad del conjunto de evaluación |
| `test_cost_metrics.py` | Comportamiento de `update_metrics` y coherencia de los JSON publicados |
| `test_taxonomy.py` | Resolución de etiquetas multi-clase y órdenes de prioridad |
| `test_configs.py` | Que `configs/` coincida con los lanzadores y que Windows y Linux estén a la par |
| `test_packaging.py` | Que los módulos se importen, que las rutas sean portables y que el CLI resuelva |

No se cubren `translate.py` (requiere el modelo NLLB), `qlora.py` (requiere GPU) ni el frontend.
