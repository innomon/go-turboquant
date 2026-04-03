# TurboQuant Specification: Engine Implementation

## Objective
Implement the core mathematical components of TurboQuant in GoMLX, including PolarQuant transformation and QJL residual correction, with efficient bit-packing and XLA fusion for high performance. Provide an OpenAI-compatible API server to expose the functionality.

## Architectural Components

### 1. PolarQuant Engine
Transform Cartesian $(x, y)$ latent pairs into Polar coordinates $(r, \theta)$.

*   **Radius ($r$):** Quantized to 4-bits using a Lloyd-Max distribution.
    *   *Input:* `gomlx.Node` representing Cartesian $x, y$.
    *   *Calculation:* $r = \sqrt{x^2 + y^2}$.
    *   *Quantization:* Map $r$ to one of 16 levels defined by a pre-computed Lloyd-Max codebook.
*   **Angle ($\theta$):** Quantized to 3-bits on a circular grid.
    *   *Calculation:* $\theta = \text{atan2}(y, x)$.
    *   *Quantization:* Divide the circle into 8 sectors ($2\pi / 8$ each).

### 2. QJL (Quantized Johnson-Lindenstrauss)
A 1-bit residual correction to recover precision.

*   **Residual Calculation:** $\text{Error} = x_{original} - \text{Dequant}(x_{polar})$.
*   **Projection:** $QJL_{sign} = \text{Sign}(\text{Error} \cdot \Omega)$, where $\Omega$ is a fixed random orthogonal rotation matrix.
*   **Rotation Matrix ($\Omega$):** A stable, deterministic random orthogonal matrix generated based on a seed.

### 3. Bit-Packing Utilities
To achieve the 6x memory reduction, we need to pack the quantized indices.

*   **Format:**
    *   Radius: 4 bits
    *   Angle: 3 bits
    *   QJL Sign: 1 bit
    *   *Total:* 8 bits per Cartesian pair.
*   **Packing Strategy:** Use `uint8` or pack multiple pairs into `uint32`/`uint64` for XLA-friendly throughput.

### 4. XLA Kernel Fusion & Backend Pattern
Ensure that operations run optimally on the GPU.

*   **Backend Initialization:** Following the pattern from `dyna-slm`:
    *   Import `_ "github.com/gomlx/gomlx/backends/xla"` to register the backend.
    *   Set `GOMLX_BACKEND="xla"` in the environment.
    *   Initialize using `backends.New()`.
*   **Fusion Strategy:** Combine `Gather` (for codebook lookup) with the `Atan2` and `Cos/Sin` reconstruction in a single XLA computation block via GoMLX nodes to prevent intermediate memory allocation.
*   **Interface:** `TurboQuantize(x, y)` and `TurboDequantize(indices, codebook)`.

### 5. OpenAPI Server (OpenAI-Compatible)
A standard HTTP server exposing the GoMLX-accelerated models.

*   **Architecture:** Use `net/http` to serve API endpoints similar to `dyna-slm`'s `cmd/api/main.go`.
*   **Authentication:** Implement JWT-based auth using Ed25519 keys derived from a seeded secret.
*   **Model Registry:** Maintain an in-memory configuration for loaded models and endpoints, wrapping the GoMLX backend operations (PolarQuant/QJL functions).

## Data Flows
1.  **API Request:** User sends OpenAI-compatible JSON payload to the API server.
2.  **Quantization (Offline/Inference):** FP16/BF16 Latents $\to$ Polar Transformation $\to$ Lloyd-Max Quantization $\to$ QJL Projection $\to$ Bit-Packing $\to$ Compressed KV Cache.
3.  **Dequantization (Inference):** Compressed KV Cache $\to$ Unpacking $\to$ Codebook Lookup $\to$ Polar Reconstruction $\to$ FP16/BF16 Latents $\to$ API Response.
