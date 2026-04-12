# MedGemma 1.5 Multimodal Support Spec

This specification outlines the architectural changes required to support **MedGemma 1.5** within the TurboQuant framework. MedGemma 1.5 is built on the Gemma 3 backbone and introduces specialized multimodal capabilities for medical imaging and long-context clinical data.

## 1. Medical SigLIP Vision Encoder
MedGemma 1.5 uses a specialized SigLIP encoder trained on medical data (X-rays, CT, Pathology).

*   **Input Resolution:** 896 x 896 pixels (normalized).
*   **Output Tokens:** 256 visual tokens per image.
*   **GoMLX Implementation:** A native SigLIP implementation is required to convert raw pixel data into the Gemma 3 embedding space.
*   **Integration Point:**
    ```go
    func BuildMedGemmaVisionEncoder(ctx *context.Context, images *Node) *Node {
        // 1. Patchify (896x896 -> 256 patches)
        // 2. Linear Projection + Positional Embeddings
        // 3. Multi-head Self-Attention layers (SigLIP-style)
        // 4. Output: [batch, 256, hidden_dim]
    }
    ```

## 2. Multimodal Token Interleaving
MedGemma 1.5 processes text and vision in a unified embedding space.

*   **Mechanism:** Image tokens (256) are interleaved with text tokens based on the prompt structure (e.g., `<image>` placeholders).
*   **Unified Cache:** TurboQuant treats visual tokens identically to text tokens once they enter the KV cache.
*   **Implementation:**
    - Update `internal/api/server.go` to handle image uploads and interleaving logic.
    - Modify `BuildGemma3Model` to accept an optional `visual_tokens` tensor.

## 3. High-Dimensional Medical Data Support
Supporting 3D volumetric imaging (CT/MRI) and whole-slide pathology.

*   **3D Volumetric:** 
    - Strategy: Process CT/MRI slices as sequential image blocks or use a 3D patch projection.
    - TurboQuant Impact: 3D data significantly increases KV cache pressure; PolarQuant's $6\times$ compression is critical here.
*   **Longitudinal Imaging:**
    - Comparing current vs. historical scans requires very long context (128K+ tokens).
    - TurboQuant must ensure stability across these massive windows.

## 4. Specialized Radius Distributions for Medical Latents
Medical image tokens may exhibit unique manifold properties compared to general-purpose SigLIP tokens.

*   **Optimization:**
    - Analyze `r` distributions from medical SigLIP outputs.
    - If necessary, provide a specialized `MedicalRadiusCodebook` (4-bit) for MedGemma models.

---

## Proposed Implementation Track (Conductor)

1.  **Phase 1 (Encoder):** Implement GoMLX-native SigLIP architecture and load MedGemma vision weights.
2.  **Phase 2 (Integration):** Update `BuildGemma3Model` to support interleaved visual tokens.
3.  **Phase 3 (API):** Extend the OpenAI-compatible server to handle multimodal requests (base64 images).
4.  **Phase 4 (Validation):** End-to-end verification using medical imaging benchmarks (e.g., Chest X-ray QA).
