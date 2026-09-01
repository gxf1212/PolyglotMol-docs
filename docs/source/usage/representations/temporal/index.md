# Temporal Representations

The ``molblender.representations.temporal`` package is reserved for future
conformational-ensemble and molecular-dynamics trajectory featurizers. It
currently registers **no** temporal featurizer. In particular, there is no
``conformer_ensemble`` or ``md_trajectory_features`` name in the MolBlender
registry, and installing MDAnalysis does not add one.

The direct compatibility classes
``molblender.representations.temporal.ensemble.Ensemble`` and
``molblender.representations.temporal.trajectory.Trajectory`` intentionally
raise ``NotImplementedError`` at construction. They are not part of the
package public export and must not be used in a screening configuration.

## Available Alternative

MolBlender does provide the registered ``dynamics_trajectory`` featurizer in
the image/video representation family. It produces an image/video-shaped
representation and must be routed to a compatible image model:

```python
from molblender.representations import get_featurizer

featurizer = get_featurizer("dynamics_trajectory")
```

This is a generated molecular-motion representation; it is not an MDAnalysis
trajectory parser and does not accept arbitrary ``.dcd``/``.pdb`` files. For
scientific analysis of external MD trajectories, use MDAnalysis or another
trajectory-analysis package directly, then ingest any fixed-size numeric
features into MolBlender as precomputed features with their declared output
type.

## Status

Temporal MD/ensemble analysis is experimental and unavailable as a MolBlender
screening representation. A future implementation must define its input
artifact contract, optional dependency policy, cache identity, output shape,
and compatible routing before registration.
