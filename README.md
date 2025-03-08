# More for Keys, Less for Values: Adaptive KV Cache Quantization ☝️🔑👇🔢

[Source code](https://github.com/mohsenhariri/kvq) for [More for Keys, Less for Values: Adaptive KV Cache Quantization](https://arxiv.org/pdf/2502.15075v1).


## Supported backends

- [x] [Quanto](https://github.com/huggingface/optimum-quanto)
- [ ] [HQQ](https://mobiusml.github.io/hqq_blog/)

## Usage

### 1. Installation

### Installation

`KVQ` can be installed via pip:

```bash
pip install kvq
```

Please note that an NVIDIA `nvcc` compiler is required to build the package. Before installing, ensure that you have the following dependencies properly set up on your system:

- **GNU Binutils (e.g., GNU assembler 2.42)**
- **C/C++ compiler** (e.g., GCC via `build-essential` or `cmake`)

### 2. Initialization

#### 2.1. Creating a KVQ object using a configuration object:

```python
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
```

#### 2.2. Alternatively, you can create a KVQ object using a dictionary:

```python
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
```

### 3. Using KVQ during text generation

```python
# Assume 'model' is a transformer-like model (e.g. Llama, Mistral, ...)
# that supports caching past key-value states.

outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    use_cache=True,
    past_key_values=kvq,
)
print(outputs)
```



## Citation

If you find our method useful, please kindly cite our paper.

```bibtex
@article{hariri2025kvq,
  title={More for Keys, Less for Values: Adaptive KV Cache Quantization},
  author={Hariri, Mohsen and Nguyen, Lam and Chen, Sixu and Zhong, Shaochen and Wang, Qifan and Hu, Xia and Han, Xiaotian and Chaudhary, Vipin},
  journal={arXiv preprint arXiv:2502.15075},
  year={2025}
}

```

## Contributing
We welcome contributions from the research community to improve this work. If you have any idea or would like to report a bug, please open an issue or submit a pull request.

## License
The code is released under the MIT License.


