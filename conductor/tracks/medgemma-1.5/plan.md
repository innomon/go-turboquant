# MedGemma 1.5 Support Plan

This plan outlines the implementation steps to add MedGemma 1.5 multimodal support to the TurboQuant framework.

## 🎯 Implementation Objectives
- [ ] **Vision Encoder:** Implement a GoMLX-native SigLIP architecture for 896x896 medical images.
- [ ] **Weight Loading:** Add support for loading MedGemma 1.5 specialized vision and backbone weights.
- [ ] **Multimodal Pipeline:** Implement token interleaving logic for text and vision.
- [ ] **API Extension:** Update the API server to handle multimodal (image + text) prompts.
- [ ] **Optimization:** Refine PolarQuant radius distributions for medical visual tokens.

---

## 📅 Roadmap

### 🧱 Phase 1: SigLIP Vision Encoder (3-4 Days)
- [ ] Implement `Patchify` operation in GoMLX.
- [ ] Build the SigLIP transformer block (Attention + MLP).
- [ ] Add `LoadSigLIPWeights` to `internal/api/server.go`.
- [ ] **Validation:** Verify SigLIP output tokens against a reference implementation for a sample medical image.

### 🏗️ Phase 2: Multimodal Integration (2-3 Days)
- [ ] Modify `BuildGemma3Model` in `turboquant/model.go` to accept `visual_tokens`.
- [ ] Implement `InterleaveTokens` logic to merge text embeddings and visual tokens.
- [ ] **Validation:** Ensure the Transformer backbone correctly processes a mixed sequence of text and image tokens.

### 🌀 Phase 3: API & Server Updates (2 Days)
- [ ] Update `ChatCompletionRequest` to support the multimodal message format (e.g., `image_url` with base64).
- [ ] Implement image decoding and normalization (896x896) in the API layer.
- [ ] **Validation:** Successfully send a multimodal prompt to the `/v1/chat/completions` endpoint and receive a text response.

### 🩺 Phase 4: Medical Optimization & Long Context (3 Days)
- [ ] Analyze radius distributions for medical tokens and update `PolarQuant` codebooks if necessary.
- [ ] Stress-test the KV cache with 128K token clinical histories + images.
- [ ] **Validation:** End-to-end benchmark on medical VQA (Visual Question Answering) tasks.

---

## 🛠 Verification Strategy
- **Unit Tests:** `turboquant/medgemma_test.go` for SigLIP and interleaving logic.
- **Integration Tests:** Use real medical safetensors to verify weight loading and inference parity.
- **Performance:** Measure throughput (tokens/sec) and memory footprint for multimodal inputs.
