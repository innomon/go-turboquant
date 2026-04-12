# Specification: Production-Ready KV Cache Refinement

## Objective
The goal of this track is to fix the critical architectural bugs and performance bottlenecks identified in the current Gemma/TurboQuant integration. We will transition the KV cache from a stateless "sketch" to a stateful, memory-efficient production component that properly supports Gemma's features (RoPE, SWA, Dual-mode generation).

## Architectural Requirements

### 1. State Persistence via GoMLX Variables
The current KV cache is a Go struct with ephemeral `*Node` fields. To survive multiple executions of the graph (e.g., during auto-regressive generation), the KV cache must be stored as **GoMLX Variables** within the `ml/context.Context`.

- **Persistent Variables:** Each layer (or shared group) will have two variables: `kv_k_cache` and `kv_v_cache`.
- **Initialization:** These variables must be pre-allocated with a fixed maximum sequence length (or a dynamically growable, but padded, size) to avoid re-compilation.

### 2. Efficient Memory Allocation (Padded Buffers)
The current `Concatenate`-based update is $O(N^2)$ in terms of memory allocation. We must switch to a **Padded Buffer** strategy, similar to the `gomlx/examples/gemma3.go` implementation.

- **Pre-allocation:** Pre-allocate buffers for `MaxSeqLen`.
- **Atomic Updates:** Use `DynamicUpdateSlice` to insert new quantized tokens into the pre-allocated buffers.
- **Pointer Management:** Maintain a `current_length` context variable (as a persistent `Scalar` variable) to track the current sequence length.

### 3. Dynamic Axis and Rank Handling
Instead of hardcoded axis indices (e.g., `AxisRange(), AxisRange(), AxisRange(0, mid)`), the implementation must be **Rank-Agnostic**.

- **Coordinate Slicing:** Dynamically determine the hidden dimension axis (usually the last rank) and split it based on `hiddenDim / 2`.
- **Batch/Seq Axis Selection:** Correctly identify the sequence axis for concatenation/update based on the input tensor's rank.

### 4. RoPE Integration
Gemma requires Rotary Positional Embeddings (RoPE).

- **RoPE Application:** RoPE must be applied to `Q` and `K` *before* the dot-product attention.
- **Dual RoPE:** Support both standard RoPE (for SWA) and Proportional RoPE (for Global layers).
- **Correct Placement:** Ensure RoPE is applied to the *dequantized* `K` value or applied to `K` before quantization if feasible (standard practice is to apply RoPE to raw `K` then store in cache).

### 5. Specialized "Turbo Mode" Efficiency
Optimize the "Quantize New / Dequantize All" workflow.

- **Quantize New:** Only the incoming token(s) are quantized and appended to the cache.
- **Dequantize All:** For self-attention, the relevant portion (or all) of the cache is dequantized.
- **Fused Kernels (Optional/Future):** Eventually explore fused quantized-attention kernels that operate directly on the packed bits.

## Data Flow (Updated)
1. **Input:** New token(s) `k, v` (FP16/BF16).
2. **RoPE:** Apply RoPE to `k`.
3. **Quantization:** Quantize `k, v` $\to$ `k_packed, v_packed`.
4. **Persistent Update:** `DynamicUpdateSlice(kv_cache_var, k_packed, [batch, current_length, ...])`.
5. **Dequantization:** `TurboDequantize(kv_cache_var[:current_length+new_len])` $\to$ `k_prime, v_prime`.
6. **Attention:** Compute dot-product using dequantized values.
7. **Increment Pointer:** `current_length += new_len`.
