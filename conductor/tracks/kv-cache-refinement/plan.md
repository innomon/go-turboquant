# Plan: KV Cache Refinement (Production-Ready State Management)

This plan outlines the steps to refactor the TurboQuant KV cache to be fully stateful, memory-efficient, and rank-agnostic.

## 🎯 Implementation Objectives
- [ ] **Statefulness:** Transition `KVCache` to use GoMLX `Variable` nodes for persistence.
- [ ] **Performance:** Implement Padded Buffers and `DynamicUpdateSlice` to eliminate $O(N^2)$ allocations.
- [ ] **Robustness:** Fix Rank-Agnostic Slicing/Concatenation in `gemma.go` and `attention4.go`.
- [ ] **Completeness:** Integrate RoPE/Dual-RoPE into the attention mechanism.

---

## 📅 Roadmap

### 🧱 Phase 1: Stateful KVCache Core (1-2 Turns)
- [ ] **Refactor `KVCache` struct:** Update the `KVCache` struct to store references to `Variable` names or indices instead of raw `*Node`.
- [ ] **Persistent Variable Management:** Implement `InitializePersistentCache` function to pre-allocate buffers in the context.
- [ ] **Update Logic:** Rewrite `Update()` to use `DynamicUpdateSlice` and increment the `current_length` scalar variable.

### 🏗️ Phase 2: Dynamic Slicing & Rank Abstraction (1 Turn)
- [ ] **Dynamic Axes:** Use `tensor.Rank()` and `-1` indices to perform Polar splitting.
- [ ] **Reshape Safety:** Ensure `ApplyRoPE` and `TurboGemmaAttention` handle variable batch sizes and sequence lengths without hardcoded axis ranges.

### 🌀 Phase 3: Integration & RoPE Fix (2 Turns)
- [ ] **Gemma 3 Fix:** Update `TurboGemmaAttention` in `turboquant/gemma.go` with state persistence and RoPE.
- [ ] **Gemma 4 Fix:** Update `TurboGemma4Attention` in `turboquant/attention4.go` to use the new `KVCache` and persistent state.
- [ ] **MTP Head Compatibility:** Ensure that the stateful cache works correctly when MTP heads predict multiple future tokens (i.e., `Update` with `n > 1`).

### 🧪 Phase 4: Validation & Benchmarking (1-2 Turns)
- [ ] **Unit Test (State):** Create a test in `turboquant/gemma_test.go` that runs multiple forward passes and verifies that the KV cache grows correctly.
- [ ] **Precision Test:** Run `turboquant/precision_test.go` to ensure RoPE and quantization haven't introduced regressions.
- [ ] **Memory Benchmarking:** Compare memory usage over 1000 tokens using the old vs. new implementation.

---

## 🛠 Verification Strategy
- **Auto-Regressive Verification:** Run a multi-step generation loop in a unit test to confirm that previous token context is preserved.
- **RoPE Position Verification:** Ensure that the `cos/sin` phases are correctly computed based on the `current_length` pointer.
- **Graph Compilation:** Confirm that the graph compiles and executes on the XLA backend without error.
