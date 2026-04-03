# TurboQuant Technical Report: Extreme KV Cache Compression for Gemma 3/4

This document provides a comprehensive technical overview of the **TurboQuant** project, detailing its mathematical foundations, integration with Google's Gemma 3 and 4 model families, and the implementation architecture using the **GoMLX** framework.

---

## 1. TurboQuant Concept, Components, and Mathematics

### 1.1 The Concept
TurboQuant is a **data-oblivious quantization** framework designed specifically for compressing the Key-Value (KV) cache in Transformer-based models. Unlike standard linear quantization (e.g., INT8), which quantizes values along a linear scale, TurboQuant treats the latent vectors as pairs of coordinates in a high-dimensional space and transforms them into a representation that better preserves the geometric properties of the attention mechanism.

The primary goal is to achieve **6× to 8× compression** (down to 3-4 bits per element) while maintaining high retrieval precision for long-context windows.

### 1.2 Core Components and Mathematics
TurboQuant operates on pairs of Cartesian coordinates $(x, y)$ from the KV cache:

1.  **PolarQuant Transformation:**
    The transformation from Cartesian to Polar coordinates preserves the "energy" (radius) and "direction" (angle) of the latent vector, which are critical for the dot-product attention mechanism.
    *   **Radius ($r$):** $r = \sqrt{x^2 + y^2}$
    *   **Angle ($\theta$):** $\theta = \text{atan2}(y, x)$

2.  **Lloyd-Max Radius Quantization (4-bit):**
    The radius distribution in modern LLMs is typically non-uniform. TurboQuant uses a **Lloyd-Max distribution** codebook to quantize the radius into 16 levels ($2^4$), minimizing the mean-squared error (MSE) of the reconstruction.

3.  **Circular Grid Angle Quantization (3-bit):**
    The angle $\theta \in [-\pi, \pi]$ is mapped to a circular grid with 8 sectors ($2^3$). This ensures that the directional information of the attention key is preserved.

4.  **QJL (Quantized Johnson-Lindenstrauss) Residual Correction:**
    To recover precision lost during the 3-bit polar stage, a 1-bit residual correction is applied:
    *   $\text{Error} = x_{original} - \text{Dequant}(x_{polar})$
    *   $QJL_{sign} = \text{Sign}(\text{Error} \cdot \Omega)$
    where $\Omega$ is a fixed random orthogonal rotation matrix. This 1-bit "bonus" significantly improves the signal-to-noise ratio of the reconstructed cache.

### 1.3 Why it is Useful
*   **Memory Efficiency:** Reduces KV cache footprint from FP16 (16 bits) to 4 bits (3-bit Polar + 1-bit QJL), enabling 4× longer context windows on the same hardware.
*   **Throughput:** By reducing memory bandwidth requirements, it can increase inference speed on memory-bound edge devices.
*   **Accuracy:** Superior to 4-bit linear quantization because it prioritizes the "angle" between vectors, which is the primary driver of attention scores.

---

## 2. Gemma 3 and Gemma 4 Internals and Architecture

### 2.1 Gemma 3 (Unified Multi-Modal)
Gemma 3 is a unified model family capable of processing text and vision (SigLIP-encoded images) in the same embedding space.
*   **Architecture:** Standard Transformer with RoPE (Rotary Positional Embeddings).
*   **KV Cache:** Uses a single, high-dimensional KV cache for all modalities. TurboQuant treats visual tokens and text tokens identically, as they share the same underlying latent manifold.

### 2.2 Gemma 4 (Adaptive Thinking & Edge-Optimized)
Gemma 4 introduces several architectural innovations that TurboQuant leverages:

1.  **Shared KV Cache:** Multiple adjacent layers share the same $K$ and $V$ projections. TurboQuant implements an "Entry Layer" quantization strategy where the compression happens once, and subsequent layers dequantize the same shared buffer.
2.  **PLE (Per-Layer Embeddings):** Each layer receives unique conditioning signals. The project implements `TurboGemma4Block` to handle PLE projection and injection.
3.  **SWA (Sliding Window Attention):** To handle massive contexts (128K+), Gemma 4 uses local sliding windows. TurboQuant manages this via circular buffer slicing of the compressed cache.
4.  **Dual RoPE:** Uses standard RoPE for local (SWA) layers and Proportional RoPE (scaling frequencies by context length) for global layers.
5.  **Agentic "Thinking" Mode:** When the model enters a `<|think|>` state, TurboQuant adaptively increases the QJL bit-depth (from 1-bit to 2-bit) for the current sequence to ensure no loss of reasoning stability.

---

## 3. Project Implementation using GoMLX

The project is implemented natively in **GoMLX**, a Go framework that leverages **XLA (Accelerated Linear Algebra)** via C++ bindings for GPU/TPU acceleration.

### 3.1 Architecture Overview
*   **Backend Layer:** Interfaces with XLA for hardware acceleration.
*   **Graph Layer:** Defines the TurboQuant operations as an execution graph.
*   **Context Management:** Stores model weights (loaded from `.safetensors`) and persistent KV cache buffers.
*   **Kernel Fusion:** TurboQuant's dequantization is fused into a single XLA operation to prevent memory round-trips to the GPU VRAM.

### 3.2 Key Implementation Files
*   `turboquant/polar.go`: Math for Cartesian/Polar transforms and `atan2` approximation.
*   `turboquant/qjl.go`: Residual sign correction logic.
*   `turboquant/packing.go`: Bit-level manipulation to pack indices into `uint8` buffers.
*   `turboquant/attention4.go`: The heavy-duty wrapper for Gemma 4 attention mechanism.
*   `internal/api/server.go`: OpenAI-compatible inference server.

---

## 4. Architecture of GoMLX and Components Used

GoMLX provides a graph-based approach to deep learning, similar to TensorFlow or JAX.

### 4.1 Core Components
*   **Graph:** The main container for computations. TurboQuant builds separate graphs for `Quantize` and `Dequantize`.
*   **Node:** Represents a tensor operation.
*   **Context:** Manages hyperparameters and learned variables (weights).
*   **Layers:** High-level abstractions (like `MultiHeadAttention`) that the project wraps with `TurboQuant` logic.
*   **Backends (XLA):** Compiles the Go-defined graph into highly optimized machine code for CUDA or CPU.

---

## 5. Mapping of TurboQuant Concepts to GoMLX Functions

| TurboQuant Concept | GoMLX Implementation / Function |
| :--- | :--- |
| **Radius ($r$)** | `Sqrt(Add(Square(x), Square(y)))` |
| **Angle ($\theta$)** | `Atan2Node(y, x)` (Custom rational approximation) |
| **Quantization** | `Floor`, `Mod`, and `Gather` (for codebook lookup) |
| **Bit-Packing** | `BitwiseOr`, `BitwiseShiftLeftScalar`, `BitwiseAnd` |
| **QJL Correction** | `Sign(Sub(orig, recon))` |
| **XLA Fusion** | Implemented by combining all nodes into a single `g.Run()` or `layers` call. |

---

## 6. Mapping Gemma Features to GoMLX Functions

| Gemma Feature | GoMLX Implementation |
| :--- | :--- |
| **Attention** | `layers.MultiHeadAttention` |
| **RoPE** | Custom `ApplyRoPE` and `ApplyProportionalRoPE` using `Sin`, `Cos`, and `Concat`. |
| **PLE Injection** | `layers.Dense` + `Add` |
| **SWA Slicing** | `Slice(cache, AxisRange(start, end))` |
| **Shared KV** | Persistent `*KVCache` struct pointers passed between `TurboGemma4Attention` calls. |

---

## 7. Unit Tests and Testing Coverage

The project uses Go's native `testing` package to verify both mathematical correctness and system integrity.

### 7.1 Key Tests
1.  **`TestTurboQuantRoundTrip`**: 
    *   **Goal:** Verifies that a tensor quantized to 4-bits and then dequantized recovers values within an acceptable MSE margin.
    *   **Scope:** Cartesian $\to$ Polar $\to$ Packed $\to$ Unpacked $\to$ Cartesian.
2.  **`TestPacking`**:
    *   **Goal:** Ensures bit-level integrity.
    *   **Scope:** Confirms that shifting and masking indices (4-bit radius, 3-bit angle, 1-bit sign) results in the exact same integers after unpacking.
3.  **`TestPrecision`**: 
    *   **Goal:** Statistical validation of the QJL correction.
    *   **Scope:** Compares 3-bit PolarQuant with and without the 1-bit QJL residual to prove accuracy gain.
4.  **`TestGemma4SharedCache`**:
    *   **Goal:** Validates that sharing the KV cache across layers doesn't lead to memory leaks or dimension mismatches.

---

## 8. Usage of the Project (OpenAI API)

The project includes a production-ready API server that mimics the OpenAI Chat Completions API, allowing users to drop TurboQuant-accelerated models into existing LLM workflows.

### 8.1 Starting the Server
```bash
# Set backend and start the API
export GOMLX_BACKEND="xla"
go run cmd/api/main.go --port 8080 --weights ./weights/gemma-3-4b/
```

### 8.2 API Consumption
The server exposes the standard `/v1/chat/completions` endpoint.

**Request:**
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-3-4b-turboquant",
    "messages": [{"role": "user", "content": "Explain KV cache compression."}]
  }'
```

**Response Schema:**
The server returns a standard JSON response containing the generated text, model ID, and usage statistics, fully compatible with tools like `openai-python` or `langchain`.

---
*Document Version: 1.0.0 | Date: April 2026*
