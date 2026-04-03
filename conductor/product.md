# TurboQuant-GoMLX Product Definition

## Vision
To provide a state-of-the-art, data-oblivious quantization framework for GoMLX, specifically optimized for Multimodal Small Language Models (MSLMs) like Gemma 3. TurboQuant aims to significantly reduce the KV cache footprint during long-context inference while maintaining high precision and performance.

## Core Features
* **PolarQuant Engine**: Transformation and quantization of Cartesian pairs to Polar coordinates (4-bit radius, 3-bit angle).
* **QJL (Quantized Johnson-Lindenstrauss)**: 1-bit residual correction to recover precision using random orthogonal rotation.
* **Efficient Bit-Packing**: Custom Go utilities to pack 3-bit/4-bit indices into `uint32` buffers for memory efficiency.
* **XLA-Fused Dequantization**: GPU-optimized kernels to minimize memory bandwidth bottlenecks.
* **Modality Agnostic**: Support for unified compression of text, image, and audio tokens.
* **Layer-Adaptive Awareness**: Support for "Heavy-Hitter" (H2O) head selection for higher precision where needed.

## Target Audience
* Researchers and engineers using GoMLX for large-scale model inference.
* Developers building long-context MSLM applications (e.g., Gemma 3).
* GoMLX community members interested in state-of-the-art quantization techniques.

## Success Metrics
* **Memory Efficiency**: Up to 6x reduction in KV cache footprint.
* **Precision**: Minimal impact on model performance compared to FP16 baseline.
* **Performance**: Maintain high inference speed through XLA fusion, with minimal compute overhead (projected < 15%).
