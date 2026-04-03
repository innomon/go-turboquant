# TurboQuant-GoMLX Tech Stack

## Core Technologies
* **GoMLX**: Go-based deep learning framework with XLA backend.
* **XLA (Accelerated Linear Algebra)**: Backend for hardware acceleration (GPU/TPU) and kernel fusion.
* **Go (1.20+)**: Main programming language for the GoMLX implementation and bit-packing utilities.
* **Lloyd-Max Distribution**: Used for 4-bit quantization of the radius in PolarQuant.
* **Sign Function**: Residual correction in QJL.

## Libraries and Components
* **github.com/gomlx/gomlx**: Core library for tensor operations and computational graphs.
* **github.com/gomlx/gemma**: Go implementation of the Gemma family of models (target for integration).
* **math/bits**: Go standard library for efficient bit manipulation.

## Deployment Target
* **GPU (CUDA)**: Optimization focus for inference speed and memory footprint.
* **TPU**: Compatibility through GoMLX's XLA backend.

## Design Patterns
* **Tensor Fusion**: Combining quantization/dequantization operations into a single XLA kernel to minimize memory access.
* **Data-Oblivious Design**: Quantization parameters are independent of the data distribution, ensuring consistency across modalities.
* **Modularity**: Separation of the core math, bit-packing, and model integration layers.
