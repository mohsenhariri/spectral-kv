==============
kvq
==============

Norm-Aware KVQuant: Precision Where It Counts


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
           "compute_dtype": torch.float16,
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

If you find our method useful, please kindly cite our paper:

    Mohsen Hariri, Alan Luo, Mohammadreza Nemati, Lam Nguyen, Shaochen Zhong, Qifan Wang, Xia Hu, Xiaotian Han, Vipin Chaudhary,
    "More for Keys, Less for Values: Adaptive KV Cache Quantization",
    `https://arxiv.org/abs/2502.15075 <https://arxiv.org/abs/2502.15075>`_