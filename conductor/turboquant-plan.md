# TurboQuant Implementation Plan: Engine Implementation

## Objective
Implement and verify the TurboQuant engine (PolarQuant + QJL) with efficient bit-packing, XLA fusion in GoMLX, and an OpenAI-compatible API server.

## Key Files & Context
* `turboquant/`: Main math implementation directory.
    * `polar.go`: Polar transformation and quantization logic.
    * `qjl.go`: Residual correction and Johnson-Lindenstrauss projection.
    * `packing.go`: Bit-packing and unpacking utilities.
    * `fusion.go`: XLA graph definitions.
* `cmd/api/main.go`: The main entry point for the OpenAPI server.

## Implementation Steps

### Phase 1: XLA Backend Initialization
1.  **[x]** Implement the standard `backends.New()` pattern.
2.  **[x]** Import the `_ "github.com/gomlx/gomlx/backends/xla"` package to register XLA natively.
3.  **[x]** Add programmatic environment variable configuration (`os.Setenv("GOMLX_BACKEND", "xla")`).

### Phase 2: Core Math (PolarQuant & QJL)
1.  **[x]** Implement `CartesianToPolar` using `gomlx.Node` operations (`Atan2`, `Sqrt`).
2.  **[x]** Implement `PolarToCartesian` using `gomlx.Node` operations (`Cos`, `Sin`).
3.  **[x]** Define the 4-bit Lloyd-Max codebook for radius $r$.
4.  **[x]** Implement `QuantizeRadius` using `Gather` and quantization thresholds.
5.  **[x]** Implement `QuantizeAngle` (3-bit circular grid).
6.  **[x]** Implement `QJLProjection`: Calculate $\text{Error}$ and project using a fixed rotation matrix $\Omega$.

### Phase 3: Bit-Packing & Unpacking
1.  **[x]** Implement `PackIndices(r_idx, theta_idx, qjl_sign)` to a single `uint8` or `uint32` buffer.
2.  **[x]** Implement `UnpackIndices` for use during dequantization.
3.  **[x]** Ensure GoMLX compatibility for passing packed buffers to XLA graphs.

### Phase 4: XLA Fusion & Optimization
1.  **[x]** Implement `TurboDequantize` as a single GoMLX computational graph.
2.  **[x]** Optimize the `Gather` operation for the radius codebook for GPU memory access patterns.
3.  **[x]** Verify XLA fusion using HLO output to ensure no intermediate large tensors are created.

### Phase 5: OpenAPI Server Implementation
1.  **[x]** Scaffold `cmd/api/main.go` using `net/http`.
2.  **[x]** Implement OpenAI-compatible HTTP handlers (e.g., `/v1/chat/completions`) that interface with the TurboQuant engine.
3.  **[x]** Verify OpenAI-compatible endpoints using `curl`.

### Phase 6: Verification & Testing
1.  **[x]** **Unit Tests:** Verify `Polar <-> Cartesian` round-trip accuracy, bit-packing logic, and QJL correction.
2.  **[x]** **API Tests:** Ensure the server boots correctly and returns valid responses for OpenAI-compatible schema requests.
3.  **[x]** **Performance Benchmarks:** Compare inference latency of FP16 vs. TurboQuant (4-bit).

### Phase 7: Gemma 3 Integration
1.  **[x]** Implement `TurboGemmaAttention` wrapper in `turboquant/gemma.go`.
2.  **[x]** Create a simulation of a compressed KV cache storage.
3.  **[x]** Verify the precision loss of `TurboGemmaAttention` compared to standard `MultiHeadAttention`.
4.  **[x]** Integrate `TurboGemmaAttention` into a full Gemma 3 block (similar to `dyna-slm` pattern).

## Alternatives Considered
* **Standard Linear Quantization (e.g., bitsandbytes):** While easier to implement, it lacks the precision and data-oblivious properties of TurboQuant.
* **RPC vs. REST:** Opted for a standard HTTP REST OpenAPI to maintain maximal compatibility with existing AI tooling ecosystems.

## Migration & Rollback
* **Rollback:** Revert to the standard GoMLX FP16 attention layer if TurboQuant introduces unacceptable precision loss or performance overhead.
* **Compatibility:** Maintain an FP16 baseline path for comparison and fallback.
