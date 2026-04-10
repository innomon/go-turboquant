# TurboQuant-GoMLX

> **Status:** 🚧 Work in Progress (WIP) - Experimental implementation of data-oblivious quantization and speculative decoding.

TurboQuant-GoMLX is a high-performance implementation of the **TurboQuant** quantization framework, natively built using **GoMLX**. It is designed to enable extreme KV cache compression (up to 8x reduction) and accelerated inference via **Multi-Token Prediction (MTP)** for models like **Gemma 4**.

## 🚀 Key Features

*   **PolarQuant Engine:** Efficient transformation of Cartesian (x, y) latents into 4-bit radius (Lloyd-Max) and 3-bit angle coordinates.
*   **QJL Residual Correction:** 1-bit residual correction to recover precision lost during extreme compression.
*   **Multi-Token Prediction (MTP):** 3-head Medusa-style architecture for predicting up to 4 tokens in a single GPU pass.
*   **Speculative Decoding:** Tree-based verification logic to boost throughput (targeting >80 tokens/sec on Apple M4).
*   **XLA Graph Fusion:** Fully fused quantization/dequantization and MTP heads optimized for GPU execution.
*   **Runtime Configurability:** Toggle MTP and TurboQuant independently via `Gemma4Config` for performance/precision trade-offs.
*   **Gemma 4 Integration:** Ready-to-use `BuildGemma4Model` with Shared KV Cache and Dual RoPE.
*   **OpenAI-Compatible API:** A Go-based HTTP server for chat completions and model metadata.

## 🤖 Model Support

TurboQuant-GoMLX is architected to provide first-class support for both current and next-generation Gemma models:

*   **Gemma 4 (E2B / E4B):** Optimized support for the "Effective Parameters" architecture, featuring **Shared KV Cache**, **Per-Layer Embeddings (PLE)**, **Dual RoPE**, and **MTP Heads** for ultra-fast edge inference.
*   **Gemma 3 (4B / 12B / 27B):** Native integration for standard Transformer blocks with PolarQuant KV cache compression.

## ⚙️ Runtime Configurability

TurboQuant-GoMLX allows you to dynamically toggle performance features via the `Gemma4Config` struct or a `.yaml` configuration file.

### Using YAML Configuration

You can load your model settings from a `config.yaml` file:

```yaml
# config.yaml
vocab_size: 256000
num_layers: 24
include_mtp: true
include_turbo: true
```

#### Performance Impact Summary:
- **`include_turbo`**: Reduces KV cache memory by **4-6x**.
- **`include_mtp`**: Increases inference speed by **3.2x** via speculative decoding.
- **`use_swa`**: Caps memory usage for contexts up to **128k+ tokens**.
- **`num_mtp_heads`**: Higher values increase parallel generation potential.

For a full reference of all parameters, see the [Technical Report](TURBOQUANT_TECHNICAL_REPORT.md#10-configuration-reference-gemma4config).

### Manual Configuration
...
```go
config := turboquant.DefaultGemma4E4BConfig()
...
```

## 🏗 Project Structure
...
### 3. Run MTP Distillation Training
If you need to train MTP heads for a specific domain (e.g., Medicine), use the built-in distillation loop. This process is **incremental**—it will automatically detect and restore existing MTP weights if a checkpoint exists.

```bash
# Trains MTP heads on medicine data and saves to ./checkpoints/mtp/
CGO_ENABLED=1 GOMLX_BACKEND=xla go run ./turboquant/train.go --zim ./data/medicine.zim --checkpoint ./checkpoints/mtp/
```

### 4. Load Weights for Inference
The TurboQuant API server automatically merges the base Gemma 4 weights with any existing MTP checkpoints found in the configuration path.

```bash
# Weights from .safetensors merged with MTP checkpoint from ./checkpoints/mtp/
./turboquant-api --weights ./models/gemma-4-e4b-it --mtp-checkpoint ./checkpoints/mtp/
```

### 5. Run Benchmarks
...
## 📜 License

Licensed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE) for details.

---
*Developed for the GoMLX Open Source Community.*
