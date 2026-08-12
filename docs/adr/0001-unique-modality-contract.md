# ADR-0001: Unique Modality Contract

## Status

✅ Accepted

## Context

The codebase had two separate `InputModality` enumerations:

- `modality_models/modality_detector.py` — the canonical enum with values
  `VECTOR`, `STRING`, `LANGUAGE_MODEL`, `MATRIX`, `IMAGE`, `GRAPH`,
  `SPATIAL_3D`, `COMPLEX`, `UNKNOWN`.
- `api/multimodal/detectors.py` — a separate copy with different casing and
  members, causing silent mismatches in routing logic.

The `InputModality` type is used at every boundary: featurizer output type,
model-eligibility check, multimodal routing plan, and handler dispatch.  Two
definitions drifting independently meant that adding a new modality (e.g.
`SPATIAL_3D`) required updating both enums and hoping they stayed in sync.

## Decision

1. Delete the duplicate enum in `api/multimodal/detectors.py`.
2. All code imports `InputModality` from
   `molblender.models.modality_models.modality_detector`.
3. The `ModalityDetector` class in `api/multimodal/detectors.py` is an adapter
   that uses the canonical enum — it owns no enum definition of its own.
4. New modality members are added to the single enum only.

## Consequences

- Single enum eliminates silent routing mismatches.
- Adding a new modality requires one change instead of two.
- The `api/multimodal/detectors.py` adapter is clearly a thin detection layer,
   not a second source of truth.