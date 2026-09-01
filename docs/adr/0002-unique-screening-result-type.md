# ADR-0002: Unique Screening Result Type

## Status

✅ Accepted

## Context

Three distinct types shared the name `ModelResult`:

1. `modality_models/core.py:PredictionResult` — raw model output (predictions,
   probabilities, training/prediction time).  Used by modality model
   implementations.
2. `screening_engine/base.py:ScreeningResult` — evaluation summary with
   metrics, scores, and metadata.  Used by the screening engine.
3. `utils/results_db.py:ResultRecord` — persistence row with DB fields
   (session_id, created_at, etc.).  Used by the database layer.

Because they shared the same name, downstream code sometimes imported the wrong
one, and model implementations occasionally bypassed their local type to use the
screening result type directly, breaking the layer boundary.

## Decision

1. Rename each type to reflect its layer:
   - `modality_models/core.py` → `PredictionResult` (alias `ModelResult` for compat)
   - `screening_engine/base.py` → `ScreeningResult` (alias `ModelResult` for compat)
   - `utils/results_db.py` → `ResultRecord` (alias `ModelResult` for compat)
2. Modality model implementations must return `PredictionResult` and must not
   import from `screening_engine.base` or `utils.results_db`.
3. Conversions between the three types happen only at the database adapter
   boundary.
4. `docs/adr/0003-runtime-dependency-direction.md` codifies the import rule.

## Consequences

- Clear naming eliminates import confusion.
- Model implementations stay layer-correct.
- Backward-compat aliases (`ModelResult = ...`) allow gradual migration.
- The check script `tests/ci/check_layer_dependencies.py` enforces the rule.

## Current location

`ResultRecord` / `ScreeningSession` / `ScreeningResultsDB` now live in
`molblender/persistence/contracts.py`. The historic `utils/results_db.py` and
`utils/database.*` paths are compatibility adapters that re-export the same
objects (identity preserved, no behavior owned there). See
`persistence/__init__.py` and `models/api/utils/database/__init__.py`.
