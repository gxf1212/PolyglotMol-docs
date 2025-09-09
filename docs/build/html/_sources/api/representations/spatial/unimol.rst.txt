.. default-role:: py:obj
.. automodule:: polyglotmol.representations.spatial.unimol
   :members: UniMolFeaturizer, UNIMOL_CONFIGS 
   :undoc-members:
   :show-inheritance:
   :inherited-members: BaseFeaturizer

Uni-Mol Representations
=======================

This module provides an interface to Uni-Mol models for generating molecule-level (CLS token) and atom-level representations. These models are designed to capture 3D spatial information.

This featurizer relies on the `unimol_tools` library. For installation and setup, refer to the :doc:`/usage/representations/spatial/unimol` guide.

Featurizer Class
----------------

.. py:class:: UniMolFeaturizer(model_config_name: str = "UniMol-Molecule-V2-84M", remove_hs: Optional[bool] = None, **kwargs)
   :noindex:

   Generates molecular and atomic representations using pre-trained Uni-Mol models.
   It wraps `unimol_tools.UniMolPredictor`. The output for each molecule is a dictionary
   containing ``'cls_repr'`` (molecule-level embedding) and ``'atomic_reprs'`` (atom-level embeddings).
   Inherits from :py:class:`~polyglotmol.representations.utils.base.BaseFeaturizer`.

   .. automethod:: polyglotmol.representations.spatial.unimol.UniMolFeaturizer._featurize

   .. automethod:: polyglotmol.representations.spatial.unimol.UniMolFeaturizer.get_output_info
   
   *Note: This method is overridden in ``UniMolFeaturizer`` to describe the dictionary output structure.*

Available Configurations
------------------------

.. py:data:: UNIMOL_CONFIGS
   :noindex:

   A dictionary defining pre-registered Uni-Mol model configurations. Each key is a registered name
   (e.g., ``"UniMol-Molecule-V2-84M"``) that can be used with 
   :py:func:`~polyglotmol.representations.get_featurizer`. The values are dictionaries specifying
   parameters for the `unimol_tools.UniMolPredictor` (like `data_type`, `model_name`, `version`, `model_size`, `remove_hs`, and an indicative `output_dim`).

   **Example structure:**

   .. code-block:: python

      UNIMOL_CONFIGS = {
          "UniMol-Molecule-V2-84M": {
              "data_type": "molecule", 
              "model_name": "unimol_plus", 
              "version": "v2", 
              "model_size": "84m", 
              "remove_hs": False, 
              "output_dim": 512 # Note: output_dim is an estimate
          },
          # ... other configurations ...
      }

Helper Functions
----------------

The module contains an internal helper function ``_ensure_unimol_loaded`` for dependency checking, which is not part of the public API for featurization.
