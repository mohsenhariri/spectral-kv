#!/usr/bin/env python3
"""
KVQ Test & Profiling Suite

Comprehensive testing and profiling for all KVQ cache implementations:
- KVQ (original)
- KVQHuggingFace

Run with: python test_and_profile.py
"""

import gc
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

from kvq.KVQ import KVQ
from kvq.KVQ_huggingface import KVQHuggingFace
from kvq.KVQConfig import KVQConfig


# ============================================================================
# Configuration
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"


# ============================================================================
# Test Results Tracking
# ============================================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    error: Optional[str] = None
    time: float = 0.0

@dataclass
class ProfileResult:
    impl_name: str
    tokens_per_sec: float
    cache_size_mb: float
    time: float
    success: bool
    error: Optional[str] = None


# ============================================================================
# Utility Functions
# ============================================================================

def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    width = 80
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}\n")

def print_section(text: str):
    """Print a section header."""
    print(f"\n{'─' * 80}")
    print(f"  {text}")
    print(f"{'─' * 80}")

def get_cache_size(cache) -> float:
    """Calculate cache memory size in MB."""
    total_bytes = 0

    if isinstance(cache, DynamicCache):
        for layer in cache.key_cache:
            if isinstance(layer, torch.Tensor):
                total_bytes += layer.element_size() * layer.nelement()
        for layer in cache.value_cache:
            if isinstance(layer, torch.Tensor):
                total_bytes += layer.element_size() * layer.nelement()

    elif isinstance(cache, KVQHuggingFace):
        # Quantized data
        for quant_data in cache._quantized_key_cache:
            if quant_data is not None:
                qtensor, meta = quant_data
                if isinstance(qtensor, torch.Tensor):
                    total_bytes += qtensor.element_size() * qtensor.nelement()
                if "scale" in meta and isinstance(meta["scale"], torch.Tensor):
                    total_bytes += meta["scale"].element_size() * meta["scale"].nelement()
                if "zero" in meta and isinstance(meta["zero"], torch.Tensor):
                    total_bytes += meta["zero"].element_size() * meta["zero"].nelement()

        for quant_data in cache._quantized_value_cache:
            if quant_data is not None:
                qtensor, meta = quant_data
                if isinstance(qtensor, torch.Tensor):
                    total_bytes += qtensor.element_size() * qtensor.nelement()
                if "scale" in meta and isinstance(meta["scale"], torch.Tensor):
                    total_bytes += meta["scale"].element_size() * meta["scale"].nelement()
                if "zero" in meta and isinstance(meta["zero"], torch.Tensor):
                    total_bytes += meta["zero"].element_size() * meta["zero"].nelement()

        # Residual cache
        for residual in cache._residual_key_cache:
            if isinstance(residual, torch.Tensor) and residual.numel() > 0:
                total_bytes += residual.element_size() * residual.nelement()
        for residual in cache._residual_value_cache:
            if isinstance(residual, torch.Tensor) and residual.numel() > 0:
                total_bytes += residual.element_size() * residual.nelement()

    elif isinstance(cache, KVQ):
        # Original KVQ - similar to HuggingFace version
        for quant_data in cache._quantized_key_cache:
            if quant_data is not None:
                qtensor, meta = quant_data
                if isinstance(qtensor, torch.Tensor):
                    total_bytes += qtensor.element_size() * qtensor.nelement()
        for quant_data in cache._quantized_value_cache:
            if quant_data is not None:
                qtensor, meta = quant_data
                if isinstance(qtensor, torch.Tensor):
                    total_bytes += qtensor.element_size() * qtensor.nelement()

        for residual in cache.key_cache:
            if isinstance(residual, torch.Tensor) and residual.numel() > 0:
                total_bytes += residual.element_size() * residual.nelement()
        for residual in cache.value_cache:
            if isinstance(residual, torch.Tensor) and residual.numel() > 0:
                total_bytes += residual.element_size() * residual.nelement()

    return total_bytes / 1024 / 1024


# ============================================================================
# Unit Tests
# ============================================================================

def test_cache_initialization(cache_class, config) -> TestResult:
    """Test that cache initializes without errors."""
    try:
        start = time.time()
        cache = cache_class(config)
        elapsed = time.time() - start
        return TestResult(
            name=f"{cache_class.__name__}: Initialization",
            passed=cache is not None,
            time=elapsed
        )
    except Exception as e:
        return TestResult(
            name=f"{cache_class.__name__}: Initialization",
            passed=False,
            error=str(e)
        )

def test_single_update(cache_class, config) -> TestResult:
    """Test single layer update."""
    try:
        start = time.time()
        cache = cache_class(config)

        # Generate random states
        key_states = torch.randn(1, 8, 10, 64, dtype=DTYPE, device=DEVICE)
        value_states = torch.randn(1, 8, 10, 64, dtype=DTYPE, device=DEVICE)

        # Update cache
        keys_out, values_out = cache.update(key_states, value_states, layer_idx=0)

        # Check outputs
        assert keys_out.shape == key_states.shape, f"Key shape mismatch: {keys_out.shape} vs {key_states.shape}"
        assert values_out.shape == value_states.shape, f"Value shape mismatch"

        elapsed = time.time() - start
        return TestResult(
            name=f"{cache_class.__name__}: Single update",
            passed=True,
            time=elapsed
        )
    except Exception as e:
        return TestResult(
            name=f"{cache_class.__name__}: Single update",
            passed=False,
            error=str(e)
        )

def test_multi_layer(cache_class, config) -> TestResult:
    """Test multi-layer updates."""
    try:
        start = time.time()
        cache = cache_class(config)
        num_layers = 4

        # Update multiple layers
        for layer_idx in range(num_layers):
            key_states = torch.randn(1, 8, 5, 64, dtype=DTYPE, device=DEVICE)
            value_states = torch.randn(1, 8, 5, 64, dtype=DTYPE, device=DEVICE)
            cache.update(key_states, value_states, layer_idx=layer_idx)

        elapsed = time.time() - start
        return TestResult(
            name=f"{cache_class.__name__}: Multi-layer ({num_layers} layers)",
            passed=True,
            time=elapsed
        )
    except Exception as e:
        return TestResult(
            name=f"{cache_class.__name__}: Multi-layer",
            passed=False,
            error=str(e)
        )

def test_incremental_updates(cache_class, config) -> TestResult:
    """Test incremental updates (like autoregressive generation)."""
    try:
        start = time.time()
        cache = cache_class(config)

        # Initial context
        key_states = torch.randn(1, 8, 10, 64, dtype=DTYPE, device=DEVICE)
        value_states = torch.randn(1, 8, 10, 64, dtype=DTYPE, device=DEVICE)
        keys_out, values_out = cache.update(key_states, value_states, layer_idx=0)
        assert keys_out.shape[-2] == 10

        # Incremental single-token updates
        for i in range(5):
            new_key = torch.randn(1, 8, 1, 64, dtype=DTYPE, device=DEVICE)
            new_value = torch.randn(1, 8, 1, 64, dtype=DTYPE, device=DEVICE)
            keys_out, _ = cache.update(new_key, new_value, layer_idx=0)
            assert keys_out.shape[-2] == 10 + i + 1, f"Expected {10 + i + 1}, got {keys_out.shape[-2]}"

        elapsed = time.time() - start
        return TestResult(
            name=f"{cache_class.__name__}: Incremental updates",
            passed=True,
            time=elapsed
        )
    except Exception as e:
        return TestResult(
            name=f"{cache_class.__name__}: Incremental updates",
            passed=False,
            error=str(e)
        )


# ============================================================================
# Integration Tests (with Real Model)
# ============================================================================

def test_model_generation(cache_class, config, model, tokenizer) -> TestResult:
    """Test generation with real model."""
    try:
        start = time.time()
        prompt = "The capital of France is"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        cache = cache_class(config)
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            past_key_values=cache,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        elapsed = time.time() - start

        return TestResult(
            name=f"{cache_class.__name__}: Model generation",
            passed=len(generated_text) > len(prompt),
            time=elapsed
        )
    except Exception as e:
        return TestResult(
            name=f"{cache_class.__name__}: Model generation",
            passed=False,
            error=str(e)
        )


# ============================================================================
# Performance Profiling
# ============================================================================

def profile_implementation(cache_class, config, model, tokenizer, prompt: str, max_tokens: int) -> ProfileResult:
    """Profile a single cache implementation."""
    try:
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # DynamicCache doesn't take a config argument
        if cache_class == DynamicCache:
            cache = cache_class()
        else:
            cache = cache_class(config)

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                past_key_values=cache,
                use_cache=True,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.time() - start

        new_tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]
        tokens_per_sec = new_tokens / elapsed
        cache_size = get_cache_size(cache)

        return ProfileResult(
            impl_name=cache_class.__name__,
            tokens_per_sec=tokens_per_sec,
            cache_size_mb=cache_size,
            time=elapsed,
            success=True
        )
    except Exception as e:
        return ProfileResult(
            impl_name=cache_class.__name__,
            tokens_per_sec=0,
            cache_size_mb=0,
            time=0,
            success=False,
            error=str(e)
        )


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    print_header("KVQ Test & Profiling Suite")

    print(f"Device: {DEVICE}")
    print(f"Dtype: {DTYPE}")
    print(f"Model: {MODEL_NAME}")

    # Configuration
    config = KVQConfig(
        nbits={"k": 4, "v": 2},
        group_size={"k": 64, "v": 64},
        axis={"k": 1, "v": 1},
        residual_length={"k": 128, "v": 128},
        compute_dtype=DTYPE,
        device=DEVICE,
    )

    print(f"\nKVQ Config:")
    print(f"  Keys:   {config.nbits_k} bits, group size {config.group_size_k}")
    print(f"  Values: {config.nbits_v} bits, group size {config.group_size_v}")

    # Cache implementations to test
    implementations = [
        ("KVQ (Original)", KVQ),
        ("KVQHuggingFace", KVQHuggingFace),
    ]

    # ========================================================================
    # UNIT TESTS
    # ========================================================================

    print_header("UNIT TESTS", "=")

    all_results: List[TestResult] = []

    for impl_name, impl_class in implementations:
        print_section(f"Testing: {impl_name}")

        tests = [
            test_cache_initialization,
            test_single_update,
            test_multi_layer,
            test_incremental_updates,
        ]

        for test_func in tests:
            result = test_func(impl_class, config)
            all_results.append(result)

            status = " PASS" if result.passed else " FAIL"
            print(f"{status} {result.name} ({result.time:.3f}s)")
            if result.error:
                print(f"      Error: {result.error}")

    # ========================================================================
    # INTEGRATION TESTS
    # ========================================================================

    print_header("INTEGRATION TESTS (Real Model)", "=")
    print("Loading model... (this may take a moment)")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=DTYPE,
            device_map=DEVICE,
        )
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"Model loaded: {model.config.num_hidden_layers} layers\n")

        for impl_name, impl_class in implementations:
            result = test_model_generation(impl_class, config, model, tokenizer)
            all_results.append(result)

            status = " PASS" if result.passed else " FAIL"
            print(f"{status} {result.name} ({result.time:.3f}s)")
            if result.error:
                print(f"      Error: {result.error}")

        # ====================================================================
        # PROFILING
        # ====================================================================

        print_header("PERFORMANCE PROFILING", "=")

        test_cases = [
            ("Short generation (30 tokens)", "The capital of France is", 30),
            ("Medium generation (50 tokens)", "In a world where artificial intelligence", 50),
        ]

        # Add baseline
        baseline_name = "DynamicCache (Baseline)"
        all_implementations = [(baseline_name, DynamicCache)] + implementations

        profile_results: Dict[str, List[ProfileResult]] = {
            impl_name: [] for impl_name, _ in all_implementations
        }

        for test_name, prompt, max_tokens in test_cases:
            print_section(test_name)
            print(f"Prompt: '{prompt[:50]}...'\n")

            for impl_name, impl_class in all_implementations:
                result = profile_implementation(impl_class, config, model, tokenizer, prompt, max_tokens)
                profile_results[impl_name].append(result)

                if result.success:
                    print(f"  {impl_name:25} {result.tokens_per_sec:6.2f} tok/s  {result.cache_size_mb:6.2f} MB  {result.time:6.3f}s")
                else:
                    print(f"  {impl_name:25}: FAILED: {result.error[:40]}")

        # ====================================================================
        # PROFILING SUMMARY
        # ====================================================================

        print_header("PROFILING SUMMARY", "=")

        # Calculate averages
        for impl_name, results in profile_results.items():
            successful = [r for r in results if r.success]
            if successful:
                avg_tps = sum(r.tokens_per_sec for r in successful) / len(successful)
                avg_cache = sum(r.cache_size_mb for r in successful) / len(successful)
                avg_time = sum(r.time for r in successful) / len(successful)

                print(f"\n{impl_name}:")
                print(f"  Average tokens/sec: {avg_tps:.2f}")
                print(f"  Average cache size: {avg_cache:.2f} MB")
                print(f"  Average time:       {avg_time:.3f}s")
            else:
                print(f"\n{impl_name}: No successful runs")

        # Compression comparison
        baseline_results = [r for r in profile_results[baseline_name] if r.success]
        if baseline_results:
            print(f"\n{'─' * 80}")
            print("Compression vs Baseline:")
            print(f"{'─' * 80}")

            baseline_avg_cache = sum(r.cache_size_mb for r in baseline_results) / len(baseline_results)

            for impl_name, results in profile_results.items():
                if impl_name == baseline_name:
                    continue

                successful = [r for r in results if r.success]
                if successful:
                    avg_cache = sum(r.cache_size_mb for r in successful) / len(successful)
                    compression = baseline_avg_cache / avg_cache if avg_cache > 0 else 1.0
                    print(f"  {impl_name:25} {compression:.2f}x compression")

    except Exception as e:
        print(f" Failed to load model: {e}")
        traceback.print_exc()

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print_header("TEST SUMMARY", "=")

    passed = sum(1 for r in all_results if r.passed)
    failed = sum(1 for r in all_results if not r.passed)
    total = len(all_results)

    print(f"Total tests: {total}")
    print(f" Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"\nSuccess rate: {(passed/total)*100:.1f}%")

    if failed > 0:
        print(f"\n{'─' * 80}")
        print("Failed tests:")
        print(f"{'─' * 80}")
        for result in all_results:
            if not result.passed:
                print(f" {result.name}")
                if result.error:
                    print(f"     {result.error[:70]}")

    print()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
