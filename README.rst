==============
kvq
==============

Norm-Aware KV Cache Quantization

- `Quantize What Counts: More For Keys, Less For Values <https://arxiv.org/abs/2502.15075v3/>`_.

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

`https://github.com/mohsenhariri/spectral-kv <https://github.com/mohsenhariri/spectral-kv>`_

Feel free to open issues, suggest improvements, or submit pull requests!


Citation
--------


If you find our work useful or interesting, please consider citing our paper:

.. code-block:: bibtex

    @article{hariri2025quantize,
      title={Quantize What Counts: More for Keys, Less for Values}, 
      author={Mohsen Hariri and Alan Luo and Weicong Chen and Shaochen Zhong and Tianyi Zhang and Qifan Wang and Xia Hu and Xiaotian Han and Vipin Chaudhary},
      year={2025},
      journal={arXiv preprint arXiv:2502.15075},
      eprint={2502.15075},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.15075}, 
}

