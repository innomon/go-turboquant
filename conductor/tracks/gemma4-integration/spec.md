# Gemma 4 Incorporation Spec: TurboQuant-GoMLX

This specification outlines the architectural changes required to support **Gemma 4 (E2B/E4B)** within the TurboQuant framework, specifically optimized for edge devices.

## 1. Per-Layer Embeddings (PLE) Integration
Gemma 4 "Effective Parameters" (E-series) leverage PLE to re-condition layers.

*   **Mechanism:** Each layer $l$ receives a secondary embedding $E_l \in \mathbb{R}^{d_{ple}}$.
*   **Implementation:** `TurboGemma4Block` will accept a `ple_input`.
*   **Integration Point:**
    ```go
    func TurboGemma4Block(ctx *context.Context, x, ple *Node, ...) *Node {
        // PLE Injection (Rescaling + Addition)
        x = Add(x, layers.Dense(ctx.In("ple_proj"), ple, false, hiddenDim))
        // ... Standard Attention + MLP ...
    }
    ```

## 2. Shared KV Cache Optimization
The most critical memory-saving feature of Gemma 4.

*   **Logic:** Multiple layers (e.g., the last 8) share the same $K$ and $V$ projections.
*   **TurboQuant Refactor:** 
    *   Introduce `SharedKVCache` that stores `KPacked` and `VPacked`.
    *   Quantization only happens once at the "Entry Layer" of the shared group.
    *   Subsequent layers only perform `TurboDequantize`.
*   **Impact:** Reduces KV cache quantization compute overhead by $N \times$, where $N$ is the shared group size.

## 3. Hybrid Attention & Dual RoPE
Supporting 128K-256K context windows on edge devices.

*   **Sliding Window Attention (SWA):** 
    *   Implement a 1D convolution or circular buffer slice for the KV cache.
    *   `KVCache.GetWindow(windowSize int)` method.
*   **Dual RoPE:**
    *   **Standard RoPE:** For SWA layers.
    *   **Proportional RoPE:** For Global layers, scaling the base frequency $10000$ by a factor proportional to the sequence length.

## 4. Multi-Modality: Audio-Aware PolarQuant
Gemma 4's USM-style audio encoder output enters the unified embedding space.

*   **Challenge:** Audio features may have different radius distributions (higher variance/frequency).
*   **Solution:** 
    *   Analyze `r` distributions from USM-conformer outputs.
    *   Optionally provide a specialized `AudioRadiusCodebook` (4-bit) for early layers that process primarily audio tokens.

## 5. Agentic "Thinking" Mode Support
Native `<|think|>` tokens require reliable long-range reasoning.

*   **TurboQuant Adaptive Bit-depth:**
    *   If `<|think|>` token is detected in the sequence, switch the **residual QJL correction** from 1-bit to 2-bit for the current sequence.
    *   This preserves the "Thinking" state's precision without bloating the entire cache.

---

## Proposed Implementation Track (Conductor)

1.  **Phase 1 (Core):** Implement `SharedKVCache` and `DualRoPE`.
2.  **Phase 2 (Structure):** `TurboGemma4Block` with PLE support.
3.  **Phase 3 (Attention):** `SlidingWindowAttention` logic in GoMLX.
4.  **Phase 4 (Refinement):** Audio-specific radius distributions and Adaptive QJL.
