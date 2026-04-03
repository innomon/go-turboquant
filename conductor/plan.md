# Track Plan: TurboQuant Engine Implementation

## Objective
Implement and verify the TurboQuant engine (PolarQuant + QJL) with efficient bit-packing and XLA fusion in GoMLX.

## Key Files & Context
* `turboquant/`: Main implementation directory.
* `turboquant/polar.go`: Polar transformation and quantization logic.
* `turboquant/qjl.go`: Residual correction and Johnson-Lindenstrauss projection.
* `turboquant/packing.go`: Bit-packing and unpacking utilities.
* `turboquant/fusion.go`: XLA fusion and kernel optimization.

## Implementation Steps

### Phase 1: Core Math (PolarQuant & QJL)
1.  **[ ]** Implement `CartesianToPolar` using `gomlx.Node` operations (`Atan2`, `Sqrt`).
2.  **[ ]** Implement `PolarToCartesian` using `gomlx.Node` operations (`Cos`, `Sin`).
3.  **[ ]** Define the 4-bit Lloyd-Max codebook for radius $r$.
4.  **[ ]** Implement `QuantizeRadius` using `Gather` and quantization thresholds.
5.  **[ ]** Implement `QuantizeAngle` (3-bit circular grid).
6.  **[ ]** Implement `QJLProjection`: Calculate $\text{Error}$ and project using a fixed rotation matrix $\Omega$.

### Phase 2: Bit-Packing & Unpacking
1.  **[ ]** Implement `PackIndices(r_idx, theta_idx, qjl_sign)` to a single `uint8` or `uint32` buffer.
2.  **[ ]** Implement `UnpackIndices` for use during dequantization.
3.  **[ ]** Ensure GoMLX compatibility for passing packed buffers to XLA graphs.

### Phase 3: XLA Fusion & Optimization
1.  **[ ]** Implement `TurboDequantize` as a single GoMLX computational graph.
2.  **[ ]** Optimize the `Gather` operation for the radius codebook for GPU memory access patterns.
3.  **[ ]** Verify XLA fusion using HLO output to ensure no intermediate large tensors are created.

### Phase 4: Verification & Testing
1.  **[ ]** **Unit Tests:**
    -   Verify `Polar <-> Cartesian` round-trip accuracy (pre-quantization).
    -   Verify bit-packing/unpacking logic for all edge cases.
    -   Verify the accuracy of the QJL residual correction.
2.  **[ ]** **Performance Benchmarks:**
    -   Compare inference latency of FP16 vs. TurboQuant (4-bit).
    -   Measure memory footprint reduction.
3.  **[ ]** **Quantization Accuracy:**
    -   Measure RMSE between FP16 latents and TurboQuant-dequantized latents across different modalities.

## Alternatives Considered
* **Standard Linear Quantization (e.g., bitsandbytes):** While easier to implement, it lacks the precision and data-oblivious properties of TurboQuant, which is crucial for long-context stability in MSLMs.
* **8-bit Quantization:** Offers better precision but doesn't meet the 6x memory reduction goal required for massive context lengths.

## Migration & Rollback
* **Rollback:** Revert to the standard GoMLX FP16 attention layer if TurboQuant introduces unacceptable precision loss or performance overhead.
* **Compatibility:** Maintain an FP16 baseline path for comparison and fallback.
