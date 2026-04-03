# Gemma 4 Integration Plan: TurboQuant-GoMLX

This plan outlines the implementation steps to incorporate Gemma 4 (E2B/E4B) support into the TurboQuant framework.

## 🎯 Implementation Objectives
- [ ] **Core Refactor:** Support Shared KV Cache across multiple layers.
- [ ] **Architecture:** Implement `TurboGemma4Block` with PLE (Per-Layer Embeddings).
- [ ] **Attention:** Support Sliding Window Attention (SWA) and Proportional RoPE.
- [ ] **Multimodality:** Optimize PolarQuant radius distributions for USM-audio tokens.
- [ ] **Optimization:** Adaptive QJL bit-depth for `<|think|>` tokens.

---

## 📅 Roadmap

### 🧱 Phase 1: Shared KV Cache & RoPE (1-2 Days)
- [ ] Modify `KVCache` to support a `Shared` mode.
- [ ] Implement `ProportionalRoPE` for Global Attention layers.
- [ ] **Validation:** Test `KVCache` update/read logic with multiple simulated layers.

### 🏗️ Phase 2: Gemma 4 Block & PLE (2-3 Days)
- [ ] Create `TurboGemma4Block` in `turboquant/block4.go`.
- [ ] Add `PLE` projection and residual addition.
- [ ] **Validation:** Compare `TurboGemma4Block` output against a reference Gemma 4 block (if available, or simulate expected behavior).

### 🌀 Phase 3: Hybrid Attention & SWA (3-4 Days)
- [ ] Implement `SlidingWindowAttention` logic in `turboquant/attention.go`.
- [ ] Add logic to switch between SWA and Global attention based on layer index.
- [ ] **Validation:** Verify context retrieval accuracy for both SWA and Global layers.

### 🔊 Phase 4: Audio & Reasoning Optimization (4-5 Days)
- [ ] Extract and analyze radius distributions for audio embeddings.
- [ ] Implement `AdaptiveQJL` bit-depth based on token presence.
- [ ] **Validation:** End-to-end benchmark for audio-heavy and reasoning-heavy sequences.

---

## 🛠 Verification Strategy
- **Unit Tests:** `turboquant/gemma4_test.go` for PLE and Shared KV cache.
- **Precision:** `turboquant/precision_test.go` expanded to cover audio-heavy sequences.
- **Throughput:** Benchmark memory usage on simulated edge devices (via GoMLX memory profiling).
