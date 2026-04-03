# TurboQuant-GoMLX

> **Status:** 🚧 Work in Progress (WIP) - Experimental implementation of data-oblivious quantization.

TurboQuant-GoMLX is a high-performance implementation of the **TurboQuant** quantization framework, natively built using **GoMLX**. It is designed to enable extreme KV cache compression (up to 6x reduction) for Multimodal Small Language Models (MSLMs) like **Gemma 3**.

## 🚀 Key Features

*   **PolarQuant Engine:** Efficient transformation of Cartesian (x, y) latents into 4-bit radius (Lloyd-Max) and 3-bit angle coordinates.
*   **QJL Residual Correction:** 1-bit residual correction to recover precision lost during extreme compression.
*   **XLA Graph Fusion:** Fully fused quantization/dequantization kernels optimized for GPU execution.
*   **Gemma 3 Integration:** Ready-to-use `TurboGemmaBlock` and `TurboGemmaAttention` layers.
*   **OpenAI-Compatible API:** A Go-based HTTP server for chat completions and model metadata.

## 🤖 Model Support

TurboQuant-GoMLX is architected to provide first-class support for both current and next-generation Gemma models:

*   **Gemma 3 (4B / 12B / 27B):** Native integration for standard Transformer blocks with PolarQuant KV cache compression.
*   **Gemma 4 (E2B / E4B):** Optimized support for the "Effective Parameters" architecture, featuring **Shared KV Cache**, **Per-Layer Embeddings (PLE)**, and **Dual RoPE** for long-context (128K+) edge inference.

Both model families are supported via a unified, modality-agnostic PolarQuant backbone, ensuring high-performance execution on mobile and edge hardware.

## 🏗 Project Structure

*   `turboquant/`: Core math engine, PolarQuant, QJL, and bit-packing logic.
*   `cmd/api/`: Entry point for the OpenAI-compatible server.
*   `internal/api/`: HTTP handlers and server orchestration.
*   `conductor/`: Implementation plans, specifications, and project tracking.

## 🛠 Prerequisites

*   Go 1.25+
*   CGO Enabled (for XLA/PJRT support)
*   GoMLX installed with XLA backend support.

## 📖 Quick Start

### 1. Download Model Weights (Optional)
Gemma 3 weights are required for real inference. Ensure you have accepted the license on Hugging Face.
```bash
pip install huggingface_hub
huggingface-cli login
python download_weights.py --model google/gemma-3-4b-it --dir models/gemma-3-4b-it
```

### 2. Build and Run the API Server
```bash
CGO_ENABLED=1 go build -o turboquant-api ./cmd/api
./turboquant-api --port 8080 --weights ./models/gemma-3-4b-it
```

### 3. Run Benchmarks
```bash
CGO_ENABLED=1 GOMLX_BACKEND=xla go test -bench . ./turboquant/...
```

## 📜 License

Licensed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE) for details.

---
*Developed for the GoMLX Open Source Community.*
