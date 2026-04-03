# Track Spec: TurboQuant Engine Implementation

## Objective
Implement the core mathematical components of TurboQuant in GoMLX, including PolarQuant transformation and QJL residual correction, with efficient bit-packing and XLA fusion for high performance.

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

### 4. XLA Kernel Fusion
Ensure that `Dequantize` runs entirely in GPU SRAM.

*   **Fusion Strategy:** Combine `Gather` (for codebook lookup) with the `Atan2` and `Cos/Sin` reconstruction in a single XLA computation block.
*   **Interface:** `TurboQuantize(x, y)` and `TurboDequantize(indices, codebook)`.

## Data Flows
1.  **Quantization (Offline/Inference):**
    -   FP16/BF16 Latents $\to$ Polar Transformation $\to$ Lloyd-Max Quantization $\to$ QJL Projection $\to$ Bit-Packing $\to$ Compressed KV Cache.
2.  **Dequantization (Inference):**
    -   Compressed KV Cache $\to$ Unpacking $\to$ Codebook Lookup $\to$ Polar Reconstruction $\to$ FP16/BF16 Latents.
