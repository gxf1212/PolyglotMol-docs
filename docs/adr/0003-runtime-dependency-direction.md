# ADR-0003: Runtime Dependency Direction

## Status

✅ Accepted

## Context

The `models/` package had a layered dependency problem:

- `modality_models/` (model implementations) imported from `models.api.*`
  (screening layer), creating a reverse dependency.
- `api/utils/gpu_manager.py` and `api/utils/monitoring.py` contained
  hardware-detection and training-monitoring code that screening and modality
  models both needed, but the code lived in the `api` layer, forcing an
  upward dependency.
- `api/infrastructure/` imported from `screening_engine.base` (full config
  types), when it only needed a small set of runtime parameters.

The desired dependency direction is:

```
modality_models → runtime / training  →  config
                    ↓
api/ → screening_engine/ → infrastructure/
```

## Decision

1. Move pure hardware helpers (CPU count, CUDA error detection, device types)
   to `models/runtime/hardware.py`.
2. Move GPU management (GPUManager, device detection) to `models/runtime/gpu/`.
3. Move training utilities (TrainingMonitor, CheckpointManager) to
   `models/training/`.
4. Create deprecated-adapters at the old `api/utils/...` locations that
   re-export from the new `models/runtime.*` and `models.training.*`.
5. `api/infrastructure/` must not import `screening_engine.base.ScreeningConfig`
   or `screening_engine.base.ExecutionMode`.  Instead, infrastructure accepts
   only the small `ExecutionContext` contract.
6. `tests/ci/check_layer_dependencies.py` is the executable dependency-policy
   source and enforces:
   - `modality_models` and `corpus` must never import `api.*`
   - `api/infrastructure` must not import `screening_engine.base`,
     `screening_engine.result_processor`, or `screening_engine.evaluation`
     (except lazy `from_screening_config` adapters listed in
     `LAZY_EXEMPT_FORBIDDEN`)
   - `api/screening_engine/evaluation` must not import
     `api.infrastructure.resource_policy`,
     `api.infrastructure.representation_resource_policy`, or
     `api.infrastructure.telemetry`.  The `api.infrastructure` facade
     (`ExecutionContext`, `emit_event`, `classify_exception`) **is** a
     permitted dependency — evaluation depends on the small runtime
     contract, not on config-aware resource policies.

## Consequences

- `modality_models` are now independent of the screening layer.
- `api/utils/...` adapter files provide backward compat without breaking the
  dependency direction.
- Infrastructure layer accepts only `ExecutionContext` (a small, stable
  contract) instead of the full `ScreeningConfig`.
- The check script prevents regressions.