# GEMINI.md

> **Status:** 🚧 Work in Progress (WIP)

## 🚀 Project Overview: TurboQuant-GoMLX
This project implements **TurboQuant**, a state-of-the-art data-oblivious quantization framework, natively within **GoMLX**. It is specifically optimized for **Multimodal Small Language Models (MSLMs)**, such as the **Gemma 3** family, to enable ultra-efficient KV cache compression during long-context inference.

### Key Objectives
* **Memory Efficiency:** Achieve up to $6\times$ reduction in KV cache footprint (e.g., compressing 16-bit floats to 3-bit Polar + 1-bit QJL).
* **Performance:** Leverage **XLA (Accelerated Linear Algebra)** via GoMLX to fuse quantization kernels, minimizing memory bandwidth bottlenecks.
* **Modality Agnostic:** Unified compression for text, image, and audio tokens within the Gemma 3 unified embedding space.

---

## 🛠 Technical Architecture

### 1. PolarQuant Engine
Instead of standard linear quantization, we transform Cartesian $(x, y)$ latent pairs into Polar coordinates. 
* **Radius ($r$):** Quantized to 4-bits using a Lloyd-Max distribution.
* **Angle ($\theta$):** Quantized to 3-bits on a circular grid.
* **GoMLX Ops:** `Atan2Node` (polynomial approx), `Sqrt`, and `Gather` (for codebook mapping).

### 2. QJL (Quantized Johnson-Lindenstrauss)
To recover the precision lost in the 3-bit polar stage, we apply a 1-bit residual correction:
$$\text{Error} = x_{original} - \text{Dequant}(x_{polar})$$
$$QJL_{sign} = \text{Sign}(\text{Error} \cdot \Omega)$$
where $\Omega$ is a fixed random orthogonal rotation matrix.

### 3. Target Model: Gemma 3 (4B/12B)
While compatible with any Transformer, this implementation is tuned for **Gemma 3**:
* **Unified KV Cache:** Compresses visual tokens from the SigLIP encoder and text tokens identically.
* **Layer-Adaptive:** Support for "Heavy-Hitter" (H2O) awareness, keeping critical attention heads at higher precision.

---

## 🏗 Implementation Progress

- [x] **Core Math:** GoMLX graph implementation of Polar transforms and high-precision `Atan2` approximation.
- [x] **Bit-Packing:** Custom Go utility to pack 3-bit/4-bit indices into `uint8` buffers.
- [x] **XLA Fusion:** Optimization of the `TurboDequantize` kernel to run entirely in GPU SRAM.
- [x] **OpenAI API:** Fully functional OpenAI-compatible server for chat completions.
- [x] **Gemma 3 Integration:** `TurboGemmaAttention` wrapper for seamless integration into transformer blocks.

---

## 📊 Performance Benchmarks (Projected)

| Metric | Baseline (FP16) | TurboQuant (4-bit) | Improvement |
| :--- | :--- | :--- | :--- |
| **Max Context (Gemma 3 4B)** | 16K Tokens | 96K Tokens | **6.0x** |
| **Memory per Token** | 2.0 MB | 0.35 MB | **82% Reduction** |
| **Inference Latency** | 1.0x | 1.15x* | *Slight compute overhead* |

---

## 📖 Usage in GoMLX

### Engine Integration
```go
// Wrapping a Gemma 3 Attention Layer
func TurboAttention(ctx *context.Context, q, k, v *Node, numHeads, headDim int) *Node {
    return turboquant.TurboGemmaAttention(ctx, q, k, v, numHeads, headDim)
}
```
**IMPORTANT**: remember to use CGO_ENABLED=1 while building GoMLX 

### API Access
```bash
# Chat Completion via OpenAI-compatible API
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-3-4b-turboquant", "messages": [{"role": "user", "content": "How does KV cache compression work?"}]}'
```

---

## 📜 References
* *Google Research (2025/26):* "TurboQuant: Data-Oblivious Compression for MSLMs." [TurboQuant blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
* *GoMLX Framework:* [github.com/gomlx/gomlx](https://github.com/gomlx/gomlx)
* *Gemma 3 Technical Report:* [ai.google.dev/gemma](https://ai.google.dev/gemma)

---
*Created for the GoMLX Open Source Community.*
