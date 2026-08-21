# ADR-0004: Single Screening Config Source of Truth

## Status

✅ Accepted

## Context

Three config types coexist in the screening API:

- `molblender.models.api.screening_engine.base.ScreeningConfig` — the
  full configuration (~80 fields) used by `StandardScreener`,
  `ProfessionalDataHandler`, `StandardEvaluator`, `ResourcePolicy`, and
  every internal screening flow.
- `molblender.models.api.multimodal.config_contracts.CoreScreeningConfig`
  — the public `universal_screen()` grouped-config surface.  It carries a
  strict subset of fields (~10) plus child dataclasses `SplitConfig`,
  `ResourceConfig`, `HPOConfig`, `DatabaseConfig`, `WeightConfig`,
  `FusionConfig`.
- `molblender.models.api.multimodal.configuration` — the canonical
  compiler that builds a full `ScreeningConfig` from a
  `CoreScreeningConfig`.

Without an explicit decision, contributors may add new fields to one
without the other, leaving the grouped API and the resulting
`ScreeningConfig` silently out of sync.

## Decision

1. `ScreeningConfig` (in `screening_engine/base.py`) is the canonical
   single source of truth — every screening flow ultimately receives a
   `ScreeningConfig` instance.  All screening-internal modules import
   from this location only.
2. `CoreScreeningConfig` (in `multimodal/config_contracts.py`) is the
   public grouped API surface.  It is **not** a parallel definition;
   it is a deliberately smaller view grouped by `SplitConfig` /
   `ResourceConfig` / `HPOConfig` / `DatabaseConfig` / `WeightConfig` /
   `FusionConfig`.  `universal_screen` is the only function that
   accepts a `CoreScreeningConfig`; it builds a full `ScreeningConfig`
   from the grouped fields in a single, canonical location
   (`multimodal/configuration.py`).
3. Adding a new field to the screening config:
   - **Required for everyone**: add to `ScreeningConfig`.
   - **Required for universal_screen() callers**: also add to
     `CoreScreeningConfig` (or the appropriate sub-config) and route
     the field through the compiler in `multimodal/configuration.py`.
     Add a default value in both places.
4. No new top-level `ScreeningConfig` subclasses.  Variants are encoded
   via composition (sub-configs in the grouped API) or via `preset`
   string on `ScreeningConfig`.
5. The `tests/ci/check_layer_dependencies.py` script enforces:
   - `screening_engine.base.ScreeningConfig` is the only definition of
     `ScreeningConfig`.
   - `multimodal.config_contracts.CoreScreeningConfig` is the only
     definition of `CoreScreeningConfig`.

## Consequences

- Single canonical definition per config type.
- New fields need a one-line addition in `ScreeningConfig` and a
  one-line addition + one-line routing in `multimodal/configuration.py`.
- Grouped API stays small and friendly; internal API stays flat and
  complete.
- The check script prevents accidental duplicate definitions.