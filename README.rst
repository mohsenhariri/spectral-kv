==============
kvq
==============

Norm-Aware KV Cache Quantization

- `Quantize What Counts: Bit Allocation Insights Informed by Spectral Gaps in Keys and Values <https://arxiv.org/abs/2502.15075v2/>`_.

- Norm-Aware KVQuant: Precision Where It Counts 


Installation
------------

To install the package from PyPI, run the following command:

.. code-block:: bash

    pip install kvq


Usage
-----

1. Initialization

   1.1. Creating a KVQ object using a configuration object:

   .. code-block:: python

       import torch
       from kvq import KVQ, KVQCacheConfig

       config = KVQCacheConfig(
           nbits_k=4,
           nbits_v=2,
           axis_key=0,
           axis_value=0,
           q_group_size=64,
           residual_length=128,
           compute_dtype=torch.bfloat16,
           backend="quanto",
           device=model.device,
       )
       kvq = KVQ(config)

   1.2. Creating a KVQ object directly from a dictionary:

   .. code-block:: python

       kvq_dict = {
           "nbits_k": 4,
           "nbits_v": 2,
           "axis_key": 0,
           "axis_value": 0,
           "q_group_size": 64,
           "residual_length": 128,
           "compute_dtype": torch.bfloat16,
           "backend": "quanto",
           "device": model.device,
       }
       kvq = KVQ(kvq_dict)

2. Using KVQ during text generation with a transformer model

   .. code-block:: python

       # Assume 'model' is a transformer-like model (e.g. Llama, Mistral, ...)
       # that supports caching past key-value states.

       outputs = model.generate(
           **inputs,
           max_new_tokens=1024,
           use_cache=True,
           past_key_values=kvq,
       )
       print(outputs)

GitHub Repository
-----------------

The source code is hosted on GitHub:

`https://github.com/mohsenhariri/kvq <https://github.com/mohsenhariri/kvq>`_

Feel free to open issues, suggest improvements, or submit pull requests!


Citation
--------


If you find our work useful or interesting, please consider citing our paper:

.. code-block:: bibtex

    @article{hariri2025quantize,
    title     = {Quantize What Counts: Bit Allocation Insights Informed by Spectral Gaps in Keys and Values},
    author    = {Hariri, Mohsen and Luo, Alan and Nemati, Mohammadreza and Nguyen, Lam and Zhong, Shaochen and Wang, Qifan and Hu, Xia and Han, Xiaotian and Chaudhary, Vipin},
    journal   = {arXiv preprint arXiv:2502.15075},
    year      = {2025},
    url       = {https://arxiv.org/abs/2502.15075v2},
    }

