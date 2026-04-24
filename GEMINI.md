# GEMINI.md

> **Status:** 🚧 Work in Progress (WIP)

## 🚀 Project Overview: TurboQuant-GoMLX
This project implements **TurboQuant**, a state-of-the-art data-oblivious quantization framework, natively within **GoMLX**. It is specifically optimized for **Multimodal Small Language Models (MSLMs)**, such as the **Gemma 4** family, to enable ultra-efficient KV cache compression and **Multi-Token Prediction (MTP)** for accelerated inference.

### Key Objectives
* **Memory Efficiency:** Achieve up to $6\times$ reduction in KV cache footprint (e.g., compressing 16-bit floats to 3-bit Polar + 1-bit QJL).
* **Performance:** Leverage **XLA (Accelerated Linear Algebra)** via GoMLX to fuse quantization kernels and MTP heads, minimizing memory bandwidth bottlenecks.
* **Speculative Decoding:** Integrate 3-head MTP to predict tokens $t+1$ to $t+4$ in a single pass, achieving $>80$ tokens/sec on Apple M4.

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

### 3. Multi-Token Prediction (MTP)
Gemma 4 integration includes a "Medusa-style" MTP architecture:
### Speculative Decoding
* **Heads:** 3 independent projection branches sharing the base model's unembedding matrix.
* **Verification:** Tree-based speculative decoding logic to validate draft tokens against the base model.
* **XLA Fusion:** Base LM head and MTP heads are fused into a single `gomlx.Tuple` for atomic GPU dispatch.

### 4. Runtime Configurability
The engine supports dynamic toggling of advanced features via `Gemma4Config`:
* **`IncludeMTP`**: Toggles the generation of speculative draft tokens.
* **`IncludeTurbo`**: Toggles the PolarQuant KV cache compression. When disabled, the model uses raw FP32 for comparative precision analysis.
* **`UseSWA`**: Toggles Sliding Window Attention for local vs. global context.

### 5. Weight Management & Checkpointing
* **Base Weights**: Loaded from standard `.safetensors` files.
* **MTP Checkpoints**: MTP head weights are stored as GoMLX checkpoints in `./checkpoints/mtp/`.
* **Incremental Training**: The distillation loop automatically detects and restores existing MTP checkpoints to support additive training sessions.
* **Inference Priority**: Trained MTP checkpoints override the base model's default MTP parameters during runtime.

---

## 🏗 Implementation Progress

- [x] **Core Math:** GoMLX graph implementation of Polar transforms and high-precision `Atan2` approximation.
- [x] **Bit-Packing:** Custom Go utility to pack 3-bit/4-bit indices into `uint8` buffers.
- [x] **XLA Fusion:** Optimization of the `TurboDequantize` kernel and MTP head bundling.
- [x] **OpenAI API:** Fully functional OpenAI-compatible server for chat completions.
- [x] **Gemma 4 Integration:** `BuildGemma4Model` with MTP, Shared KV Cache, and Dual RoPE support.
- [x] **MTP Training:** Self-distillation loop using Kiwix ZIM archives (Medicine subject).

---

## 📊 Performance Benchmarks (Projected)

| Metric | Baseline (FP16) | TurboQuant + MTP | Improvement |
| :--- | :--- | :--- | :--- |
| **Max Context (Gemma 4 4B)** | 16K Tokens | 128K Tokens | **8.0x** |
| **Memory per Token** | 2.0 MB | 0.35 MB | **82% Reduction** |
| **Throughput (M4)** | ~25 tok/sec | >80 tok/sec | **3.2x Speedup** |

---

## 📖 Usage in GoMLX

**Remember** for GoMLX builds use CGO_ENABLED=1.

### Select your Backend
GoMLX supports multiple backends (XLA, Apple Silicon Native, Pure Go). For Apple Silicon M4 users, **`go-darwinml`** is the recommended choice for maximum performance. See **[conductor/tech-stack.md](./conductor/tech-stack.md#gomlx-backends-2026)** for a full list of available backends and setup tips.

### MTP Training
```go
// Training MTP heads on Medicine data
turboquant.TrainMTP(ctx, turboquant.DefaultGemma4E4BConfig())
```

### Inference with Speculative Decoding
```go
// Verify and accept draft tokens
result := turboquant.VerifyMTP(ctx, baseLogits, mtpLogits)
fmt.Printf("Accepted %d tokens\n", result.AcceptCount)
```

---

## 📜 References
* *Google Research (2025/26):* "TurboQuant: Data-Oblivious Compression for MSLMs." [TurboQuant blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
* *GoMLX Framework:* [github.com/gomlx/gomlx](https://github.com/gomlx/gomlx)
* *Gemma 4 Technical Report:* [ai.google.dev/gemma](https://ai.google.dev/gemma)

---
*Created for the GoMLX Open Source Community.*
