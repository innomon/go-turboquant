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

## 3. Multi-Token Prediction (MTP) and Speculative Decoding

### 3.1 MTP Architecture
To further increase inference throughput beyond KV cache compression, TurboQuant-GoMLX implements **Multi-Token Prediction (MTP)** for Gemma 4. This follows a "Medusa-style" architecture where $k$ independent heads are attached to the final hidden state of the backbone.
*   **Structure:** Each head consists of a ResNet-style skip connection followed by a dense layer and SiLU activation.
*   **Weight Sharing:** To fit within the 16GB RAM limit of an Apple M4, all MTP heads share the same unembedding matrix as the base Gemma 4 model.
*   **Prediction:** In a single forward pass, the model generates $t+1$ (base) and $t+2, t+3, t+4$ (MTP heads).

### 3.2 Tree-Based Speculative Verification
TurboQuant-GoMLX uses a verification sequence to ensure accuracy:
1.  **Drafting:** Parallel generation of 4 tokens.
2.  **Verification:** The draft tokens are fed back into the base model (leveraging the TurboQuant-compressed KV cache) to confirm if the base model's greedy choices match the MTP predictions.
3.  **Acceptance:** The system commits $N$ accepted tokens to the KV cache using a single "jump-ahead" update operation.

### 3.3 MTP Distillation (ZIM Archive Training)
If pre-trained MTP weights are unavailable, the project includes a self-distillation loop:
*   **Loss:** Cross-entropy between MTP head predictions and the frozen backbone's target distribution.
*   **Data Pipeline:** Streams medical text from Kiwix ZIM archives using a sliding window batcher.
*   **Optimizer:** Adagrad (bfloat16) to conserve memory during the distillation pass on Apple M4.

### 3.4 Runtime Configurability
The architecture is designed for dynamic experimentation and evaluation via the `Gemma4Config` structure:
*   **Ablation Studies:** Researchers can toggle `IncludeTurbo` to compare the accuracy of TurboQuant against a floating-point baseline without changing the model architecture.
*   **Adaptive Performance:** `IncludeMTP` can be disabled for applications requiring maximum single-token consistency or to save memory during training.

### 3.5 Weight Management & Checkpointing

TurboQuant-GoMLX employs a dual-source weight loading strategy:

1.  **Backbone Weights:** Loaded from standard HuggingFace `.safetensors` files into the `gemma4` scope. These weights are set to `trainable: false` during MTP distillation.
2.  **MTP Checkpoints:** Specialized MTP head weights are saved using GoMLX's native `checkpoints` package. These are typically stored in a separate `./checkpoints/mtp/` directory to allow for domain-specific fine-tuning (e.g., medicine, code, legal).
3.  **Incremental Restoration:** The distillation training loop calls `cp.Restore()` at the start of every session. If a checkpoint exists, the training resumes with the previously learned MTP weights, enabling additive learning over multiple datasets.
4.  **Runtime Merging:** During inference, the API server initializes the model with base weights and then overlays any available MTP checkpoints, ensuring that the most recent or relevant heads are prioritized.

---

## 4. Project Implementation using GoMLX

The project is implemented natively in **GoMLX**, a Go framework that leverages **XLA (Accelerated Linear Algebra)** via C++ bindings for GPU/TPU acceleration.

### 4.1 Architecture Overview
*   **Backend Layer:** Interfaces with XLA for hardware acceleration.
*   **Graph Layer:** Defines the TurboQuant operations as an execution graph.
*   **Context Management:** Stores model weights (loaded from `.safetensors`) and persistent KV cache buffers.
*   **Kernel Fusion:** TurboQuant's dequantization is fused into a single XLA operation to prevent memory round-trips to the GPU VRAM.

### 4.2 Key Implementation Files
*   `turboquant/polar.go`: Math for Cartesian/Polar transforms and `atan2` approximation.
*   `turboquant/qjl.go`: Residual sign correction logic.
*   `turboquant/packing.go`: Bit-level manipulation to pack indices into `uint8` buffers.
*   `turboquant/attention4.go`: The heavy-duty wrapper for Gemma 4 attention mechanism.
*   `turboquant/mtp.go`: MTP head definitions and speculative decoding graph.
*   `internal/api/server.go`: OpenAI-compatible inference server.

---

## 5. Architecture of GoMLX and Components Used

GoMLX provides a graph-based approach to deep learning, similar to TensorFlow or JAX.

### 5.1 Core Components
*   **Graph:** The main container for computations. TurboQuant builds separate graphs for `Quantize` and `Dequantize`.
*   **Node:** Represents a tensor operation.
*   **Context:** Manages hyperparameters and learned variables (weights).
*   **Layers:** High-level abstractions (like `MultiHeadAttention`) that the project wraps with `TurboQuant` logic.
*   **Backends (XLA):** Compiles the Go-defined graph into highly optimized machine code for CUDA or CPU.

---

## 6. Mapping of TurboQuant Concepts to GoMLX Functions

| TurboQuant Concept | GoMLX Implementation / Function |
| :--- | :--- |
| **Radius ($r$)** | `Sqrt(Add(Square(x), Square(y)))` |
| **Angle ($\theta$)** | `Atan2Node(y, x)` (Custom rational approximation) |
| **Quantization** | `Floor`, `Mod`, and `Gather` (for codebook lookup) |
| **Bit-Packing** | `BitwiseOr`, `BitwiseShiftLeftScalar`, `BitwiseAnd` |
| **QJL Correction** | `Sign(Sub(orig, recon))` |
| **XLA Fusion** | Implemented by combining all nodes into a single `g.Run()` or `layers` call. |

---

## 7. Mapping Gemma Features to GoMLX Functions

| Gemma Feature | GoMLX Implementation |
| :--- | :--- |
| **Attention** | `layers.MultiHeadAttention` |
| **RoPE** | Custom `ApplyRoPE` and `ApplyProportionalRoPE` using `Sin`, `Cos`, and `Concat`. |
| **PLE Injection** | `layers.Dense` + `Add` |
| **SWA Slicing** | `Slice(cache, AxisRange(start, end))` |
| **Shared KV** | Persistent `*KVCache` struct pointers passed between `TurboGemma4Attention` calls. |

---

## 8. Unit Tests and Testing Coverage

The project uses Go's native `testing` package to verify both mathematical correctness and system integrity.

### 8.1 Key Tests
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

## 9. Usage of the Project (OpenAI API)

The project includes a production-ready API server that mimics the OpenAI Chat Completions API, allowing users to drop TurboQuant-accelerated models into existing LLM workflows.

### 9.1 Starting the Server
```bash
# Set backend and start the API
export GOMLX_BACKEND="xla"
go run cmd/api/main.go --port 8080 --weights ./weights/gemma-3-4b/
```

### 9.2 API Consumption
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

## 10. Configuration Reference (Gemma4Config)

The behavior and performance of TurboQuant-GoMLX are controlled via the `Gemma4Config` structure. Below is a detailed breakdown of each parameter.

### 10.1 Model Hyperparameters

| Parameter | Default | Min / Max | Description | Performance Impact |
| :--- | :--- | :--- | :--- | :--- |
| `vocab_size` | 256,000 | 1 / 512k+ | The number of unique tokens in the embedding space. | **Memory:** High. Impacts embedding and unembedding matrices. |
| `num_layers` | 24 (E4B) | 1 / 128 | Total transformer blocks in the model. | **Latency/Memory:** Linear. Directly scales total KV cache and forward pass time. |
| `num_heads` | 16 | 1 / 128 | Number of attention heads for Query, Key, and Value. | **Memory:** High. Directly affects the size of each layer's KV cache. |
| `head_dim` |  head_dim | 32 / 256 | Internal dimension per attention head. | **Compute:** Affects GEMM kernel efficiency. |
| `hidden_dim` | 2048 | 512 / 8k | The width of the residual stream ($d_{model}$). | **Memory:** Very High. Primary driver of model weight memory footprint. |
| `ple_dim` | 512 | 0 / `hidden_dim` | Dimension of Per-Layer Embeddings (PLE). | **Compute:** Minor. Adds a small dense projection per layer. |

### 10.2 Feature & Optimization Toggles

| Parameter | Default | Type | Description | Performance Impact |
| :--- | :--- | :--- | :--- | :--- |
| `include_turbo` | `true` | `bool` | Toggles TurboQuant KV Cache compression. | **Memory:** Reduces KV cache footprint by **4-6x**. **Latency:** Adds slight overhead for dequantization fusion. |
| `include_mtp` | `true` | `bool` | Toggles Multi-Token Prediction heads. | **Throughput:** Enables **3x+** speedup via speculative decoding. **Memory:** Adds weights for $k$ heads. |
| `num_mtp_heads` | 3 | `int` | Number of speculative tokens to predict ($k$). | **Throughput:** Higher $k$ increases potential speedup but may lower acceptance rate. |
| `use_swa` | `true` | `bool` | Toggles Sliding Window Attention (SWA). | **Memory:** Caps KV cache growth for ultra-long context (128k+). |
| `max_window` | 8192.0 | `float` | The window size for SWA and RoPE scaling. | **Memory:** Defines the maximum buffer size per layer when SWA is active. |

---

## 11. Performance Tuning Guide for Apple M4 (16GB)

To achieve optimal performance on a 16GB M4, we recommend the following configurations:

1.  **For 128K Context:** Set `include_turbo: true` and `use_swa: true`. This keeps the total VRAM usage under 14GB by compressing the KV cache of the Gemma 4 E4B backbone.
2.  **For Maximum Chat Speed:** Set `include_mtp: true` with `num_mtp_heads: 3`. This utilizes the Apple M4's Neural Engine and GPU parallelism to bundle 4 tokens into a single dispatch.
3.  **For High Precision Reasoning:** Toggle `include_turbo: false` for short-context tasks where the 16GB memory limit is not a bottleneck, providing raw FP32 attention precision.

---
*Document Version: 1.2.0 | Date: April 2026*
