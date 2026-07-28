Screening API
=============

Core screening functions for automated machine learning.

.. currentmodule:: molblender.models.api

Universal Screening
-------------------

.. autofunction:: universal_screen

.. autoclass:: UniversalScreener
   :members:
   :undoc-members:
   :show-inheritance:

Vector Representation Fusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Concatenate dense 2D vector representations (fingerprints, descriptors,
precomputed feature tables) into a synthetic representation and let the
rest of the screening pipeline treat it like any other row.

.. autoclass:: molblender.models.api.multimodal.api.FusionConfig
   :members:
   :undoc-members:
   :show-inheritance:

A typical call:

.. code-block:: python

   from molblender.models.api.multimodal.api import FusionConfig

   results = universal_screen(
       dataset=ds,
       target_column="y",
       fusion_config=FusionConfig(
           enabled=True,
           groups=[["morgan_fp", "maccs_keys"], ["morgan_fp", "rdkit_descriptors"]],
           name_prefix="fusion",
       ),
   )

Each fusion row is named ``fusion__A__B__h<8-hex-digest>`` (SHA-256 over
the ordered component names). The component composition is stored in
``representation_config`` (``type="fusion"``, ``schema_version=2``,
``components``, ``component_feature_counts``,
``component_feature_ranges``, ``total_features``,
``execution_modality="vector"``, ``display_modality="vector_fusion"``).
This metadata includes the ordered per-component quality operations and is
consumed by HPO reconstruction, Dashboard ``vector_fusion`` mapping, and
generated export scripts through the same core reconstruction helper.
Fusion is opt-in and currently limited to ordered concatenation of dense 2D
vector components; it is not general multimodal or learned fusion.

Standard Screening Functions
-----------------------------

.. autofunction:: screen_models

.. autofunction:: quick_screen

.. autofunction:: thorough_screen

.. autofunction:: interpretable_screen

Comparison Functions
--------------------

.. autofunction:: compare_models

.. autofunction:: compare_representations

Evaluation
----------

.. autofunction:: simple_evaluate

See Also
--------

- :doc:`../../usage/models/screening` - Usage guide
- :doc:`corpus` - Model definitions
- :doc:`utils` - Utility functions
