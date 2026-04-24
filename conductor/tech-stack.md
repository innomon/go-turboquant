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
* **Apple Silicon (M4)**: Leveraging `go-darwinml` for native acceleration on M4 Mac Mini/Pro.
* **Raspberry Pi 5 (ARM64)**: Support via `xla:cpu` with SIMD optimizations or the pure `go` backend for extreme portability.

## GoMLX Backends (2026)
In 2026, **GoMLX** has expanded its backend support to cover everything from ultra-portable pure Go environments to high-performance Apple Silicon and TPU clusters.

To use these backends, you typically set the `GOMLX_BACKEND` environment variable or specify it when calling `backends.New()`.

### 1. `xla` (OpenXLA Backend)
This is the **primary performance backend** and the one you should use for your Gemma 4 training. It interfaces with Google’s XLA (Accelerated Linear Algebra) via the `gopjrt` wrapper.
* **`xla:cpu`**: Runs on standard CPU (supports Linux/AMD64, Linux/ARM64, and Darwin/ARM64).
* **`xla:cuda`**: The standard for NVIDIA GPUs.
* **`xla:rocm`**: Support for AMD GPUs.
* **`xla:tpu`**: Specifically for Google Cloud TPU nodes.
* **`xla:metal` (New in 2025/2026)**: Leverages OpenXLA’s Metal plugin for native GPU acceleration on Apple Silicon (M1/M2/M3/M4).

### 2. `go-darwinml` (Apple Silicon Native)
As of early 2026, this is the **recommended backend for Mac Mini M4 users**.
* **Description:** A specialized bridge that connects GoMLX directly to Apple’s **MLX** and **CoreML** frameworks.
* **Benefits:** It provides the most efficient access to the M4’s Unified Memory and Neural Engine. It is often faster than the generic XLA Metal plugin for transformer-based models like Gemma.
* **Usage:** Set `GOMLX_BACKEND="go-darwinml"`.

### 3. `go` (SimpleGo / Pure Go)
The "fallback" backend.
* **Description:** A 100% pure Go implementation with zero C++ dependencies.
* **Benefits:** Highly portable; can compile to **WASM**, Windows, or tiny Linux distros.
* **2026 Update:** Now includes support for **SIMD (Go-Highway)** and basic quantization, though it is still significantly slower than XLA for large-scale LLM training.

### 4. `stablehlo` (Low-Level)
Technically an alias for the XLA backend but focused on the **StableHLO** (Stable High-Level Operations) set.
* **Use Case:** Best for developers building their own compilers or specialized hardware integrations. It allows for cross-platform portability of compiled "computation graphs."

---

### Summary Table for your M4 Mac

| Backend Name | Device | Best For | Recommendation |
| :--- | :--- | :--- | :--- |
| **`go-darwinml`** | **M4 GPU / ANE** | **Gemma 4 Training** | **Top Choice** |
| `xla:metal` | M4 GPU | General XLA Research | Second Choice |
| `xla:cpu` | M4 CPU | Debugging Logic | Fallback |
| `go` | M4 CPU | Portability / WASM | Avoid for LLMs |

> **Setup Tip:** To verify which backends are currently compiled into your Go environment, you can run this small Go snippet:
> ```go
> fmt.Printf("Available GoMLX Backends: %v\n", backends.List())
> ```

## Environment Configuration

### PJRT vs Native Backends
A common question for M4 users is whether they need the `PJRT_NAMES_AND_LIBRARY_PATHS` environment variable.

The short answer is **no**. If you use the `go-darwinml` backend, you generally do **not** need the `PJRT_NAMES_AND_LIBRARY_PATHS` environment variable.

#### 1. Why PJRT is not required for `go-darwinml`
The `PJRT` variables (like `PJRT_NAMES_AND_LIBRARY_PATHS` or `PJRT_PLUGIN_LIBRARY_PATH`) are specifically for the **XLA backend**. XLA uses a plugin architecture where it dynamically loads a "PJRT plugin" (a `.so` or `.dylib` file) to talk to specific hardware like NVIDIA GPUs or TPUs.

In contrast, **`go-darwinml`** is a native Darwin (macOS) bridge. It bypasses the XLA/PJRT stack entirely to communicate directly with:
* **Metal:** Apple's GPU programming framework.
* **MLX:** Apple's array framework specifically for Apple Silicon.
* **CoreML:** For specialized Neural Engine (ANE) acceleration.

#### 2. What you actually need for `go-darwinml`
Since this backend uses native macOS libraries, you don't need "plugins," but you do need the standard Apple development environment:

* **Xcode Command Line Tools:** Ensure you have them installed (`xcode-select --install`).
* **CGO Enabled:** Since `go-darwinml` wraps Apple's C/Swift-based frameworks, you must have `CGO_ENABLED=1` when building your Go app.
* **No extra environment variables:** Usually, `GOMLX_BACKEND="go-darwinml"` is enough. The system will find the required frameworks (Metal, Accelerate) in the standard macOS system paths.

#### 3. Comparison of Backends for your M4
| Backend | Hardware Path | Requires PJRT? | Env Var Needed |
| :--- | :--- | :--- | :--- |
| **`xla:metal`** | XLA → PJRT Plugin → Metal | **Yes** | `PJRT_PLUGIN_LIBRARY_PATH` |
| **`go-darwinml`**| GoMLX → Native Bridge → Metal/MLX | **No** | `GOMLX_BACKEND="go-darwinml"` |

**When *would* you need PJRT?**
You only need to worry about those paths if you decide to switch back to the **XLA backend** (e.g., `GOMLX_BACKEND="xla:metal"`). In that case, GoMLX looks for a file like `libpjrt_metal.dylib`. If that file isn't in a standard location like `/usr/local/lib`, you would use the environment variable to tell GoMLX where you've hidden it.

**Pro-Tip for M4:** Stick with `go-darwinml`. It is specifically tuned for the Unified Memory architecture of your M4 chip and usually offers a much smoother "out-of-the-box" experience without the headache of managing external C++ plugin paths.
