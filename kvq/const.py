from hqq.core.quantize import Quantizer

# 3.5 is not supported yet
_SUPPORTED_BITS = (*Quantizer.SUPPORTED_BITS, 3.5)


llama3_3 = [
    "meta-llama/Llama-3.3-70B-Instruct",
]

llama3_2 = [
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
]

llama3_1 = [
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
]

llama3 = [
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
]


qwen3 = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
]

supported_models = llama3 + llama3_1 + llama3_2 + llama3_3 + qwen3

model_dict = {
    "deepseek-ai/deepseek-math-7b-base": "DeepSeekMath-7B",
    "deepseek-ai/deepseek-math-7b-instruct": "DeepSeekMath-7B-it",
    "deepseek-ai/deepseek-math-7b-rl": "DeepSeekMath-7B-rl",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "DeepSeekR1Q-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "DeepSeekR1Q-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "DeepSeekR1Q-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "DeepSeekR1Q-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "DeepSeekR1L-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "DeepSeekR1L-70B",
    "mistralai/Mistral-7B-v0.1": "Mistral0.1-7B",
    "mistralai/Mathstral-7B-v0.1": "Mathstral0.1-7B",
    "mistralai/Mistral-7B-v0.3": "Mistral0.3-7B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral0.3-7B-it",
    "mistralai/Ministral-8B-Instruct-2410": "Ministral2410-8B-it",
    "mistralai/Mistral-Large-Instruct-2411": "Mistral2411-Large-it",
    "meta-llama/Llama-3.3-70B-Instruct": "Llama3.3-70B-it",
    "meta-llama/Llama-3.2-1B": "Llama3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct": "Llama3.2-1B-it",
    "meta-llama/Llama-3.2-3B": "Llama3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama3.2-3B-it",
    "meta-llama/Llama-3.1-8B": "Llama3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama3.1-8B-it",
    "meta-llama/Llama-3.1-70B": "Llama3.1-70B",
    "meta-llama/Llama-3.1-70B-Instruct": "Llama3.1-70B-it",
    "meta-llama/Llama-3.1-405B-Instruct": "Llama3.1-405B-it",
    "meta-llama/Meta-Llama-3-8B": "Llama3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct": "Llama3-8B-it",
    "meta-llama/Meta-Llama-3-70B": "Llama3-70B",
    "meta-llama/Meta-Llama-3-70B-Instruct": "Llama3-70B-it",
    "meta-llama/Llama-2-7b-hf": "Llama2-7B",
    "meta-llama/Llama-2-7b-chat-hf": "Llama2-Chat-7B",
    "meta-llama/Llama-2-13b-hf": "Llama2-13B",
    "meta-llama/Llama-2-13b-chat-hf": "Llama2-Chat-13B",
    "meta-llama/Llama-2-70b-hf": "Llama2-70B",
    "meta-llama/Llama-2-70b-chat-hf": "Llama2-Chat-70B",
    "openai-community/gpt2": "Gpt2",
    "openai-community/gpt2-medium": "Gpt2-Medium",
    "openai-community/gpt2-large": "Gpt2-Large",
    "openai-community/gpt2-xl": "Gpt2-Xl",
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF": "Memotron3.1-it",
    "google/gemma-2-2b": "Gemma2-2B",
    "google/gemma-2-2b-it": "Gemma2-2B-it",
    "google/gemma-2-9b": "Gemma2-9B",
    "google/gemma-2-9b-it": "Gemma2-9B-it",
    "google/gemma-2-27b": "Gemma2-27B",
    "google/gemma-2-27b-it": "Gemma2-27B-it",
    "google/recurrentgemma-2b": "Recurrentgemma-2B",
    "google/recurrentgemma-2b-it": "Recurrentgemma-2B-it",
    "google/recurrentgemma-9b": "Recurrentgemma-9B",
    "google/recurrentgemma-9b-it": "Recurrentgemma-9B-it",
    "microsoft/phi-4": "Phi4",
    "microsoft/Phi-3.5-mini-instruct": "Phi3.5-Mini-it",
    "microsoft/Phi-3-medium-128k-instruct": "Phi3-Medium-128K-it",
    "microsoft/Phi-3-mini-128k-instruct": "Phi3-Mini-128K-it",
    "Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B-it",
    "Qwen/Qwen2.5-7B": "Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B-it",
    "Qwen/Qwen2.5-14B": "Qwen2.5-14B",
    "Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B-it",
    "Qwen/Qwen2.5-72B": "Qwen2.5-72B",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5-72B-it",
    "Qwen/Qwen2.5-Math-1.5B": "Qwen2.5-Math-1.5B",
    "Qwen/Qwen2.5-Math-1.5B-Instruct": "Qwen2.5-Math-1.5B-it",
    "Qwen/Qwen2.5-Math-7B": "Qwen2.5-Math-7B",
    "Qwen/Qwen2.5-Math-7B-Instruct": "Qwen2.5-Math-7B-it",
    "Qwen/Qwen2.5-Math-72B": "Qwen2.5-Math-72B",
    "Qwen/Qwen2.5-Math-72B-Instruct": "Qwen2.5-Math-72B-it",
    "Qwen/Qwen2.5-Math-PRM-7B": "Qwen2.5-Math-7B-PRM",
    "Qwen/Qwen2.5-Math-PRM-72B": "Qwen2.5-Math-72B-PRM",
    "Qwen/Qwen2.5-Math-RM-72B": "Qwen2.5-Math-72B-RM",
    "Qwen/QwQ-32B-Preview": "QwenQwQ-32B-pr",

    "Qwen/Qwen3-0.6B": "Qwen3-0.6B",
    "Qwen/Qwen3-4B": "Qwen3-4B",
    "Qwen/Qwen3-8B": "Qwen3-8B",
    "Qwen/Qwen3-14B": "Qwen3-14B",
    "Qwen/Qwen3-32B": "Qwen3-32B",
    
}
