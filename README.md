# TurboQuant-GoMLX

> **Status:** 🚧 Work in Progress (WIP) - Experimental implementation of data-oblivious quantization.

TurboQuant-GoMLX is a high-performance implementation of the **TurboQuant** quantization framework, natively built using **GoMLX**. It is designed to enable extreme KV cache compression (up to 6x reduction) for Multimodal Small Language Models (MSLMs) like **Gemma 3**.

## 🚀 Key Features

*   **PolarQuant Engine:** Efficient transformation of Cartesian (x, y) latents into 4-bit radius (Lloyd-Max) and 3-bit angle coordinates.
*   **QJL Residual Correction:** 1-bit residual correction to recover precision lost during extreme compression.
*   **XLA Graph Fusion:** Fully fused quantization/dequantization kernels optimized for GPU execution.
*   **Gemma 3 Integration:** Ready-to-use `TurboGemmaBlock` and `TurboGemmaAttention` layers.
*   **OpenAI-Compatible API:** A Go-based HTTP server for chat completions and model metadata.

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

### Build the API Server
```bash
CGO_ENABLED=1 go build -o turboquant-api ./cmd/api
```

### Run Benchmarks
```bash
CGO_ENABLED=1 GOMLX_BACKEND=xla go test -bench . ./turboquant/...
```

## 📜 License

Licensed under the **Apache License, Version 2.0**. See [LICENSE](LICENSE) for details.

---
*Developed for the GoMLX Open Source Community.*
