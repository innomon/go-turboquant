package turboquant

import (
	"fmt"
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"os"
	"testing"
)

func TestMedGemmaVisionEncoder(t *testing.T) {
	os.Setenv("GOMLX_BACKEND", "go")
	backend, err := backends.New()
	if err != nil {
		t.Fatalf("Failed to create backend: %v", err)
	}
	ctx := context.New()

	// Mock 896x896 image
	imgShape := []int{1, 896, 896, 3}
	imgTensor := tensors.FromScalarAndDimensions(float32(0.5), imgShape...)

	exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, images *Node) *Node {
		config := DefaultMedGemmaSigLIPConfig()
		return BuildMedGemmaVisionEncoder(ctx, images, config)
	})
	if err != nil {
		t.Fatalf("Failed to build execution graph: %v", err)
	}

	results, err := exec.Exec(imgTensor)
	if err != nil {
		t.Fatalf("Execution failed: %v", err)
	}

	output := results[0]
	fmt.Printf("Vision Encoder output shape: %v\n", output.Shape())

	expectedShape := []int{1, 256, 1152}
	for i, dim := range output.Shape().Dimensions {
		if dim != expectedShape[i] {
			t.Errorf("Dimension %d mismatch: got %d, want %d", i, dim, expectedShape[i])
		}
	}
}

func TestMedGemmaInterleaving(t *testing.T) {
	os.Setenv("GOMLX_BACKEND", "go")
	backend, err := backends.New()
	if err != nil {
		t.Fatalf("Failed to create backend: %v", err)
	}
	ctx := context.New()

	// Mock embeddings and visual tokens
	hiddenDim := 128
	textSeqLen := 10
	visualSeqLen := 256
	
	textEmbedsShape := []int{1, textSeqLen, hiddenDim}
	textEmbedsTensor := tensors.FromScalarAndDimensions(float32(0.5), textEmbedsShape...)
	
	visualShape := []int{1, visualSeqLen, hiddenDim}
	visualTensor := tensors.FromScalarAndDimensions(float32(1.0), visualShape...)

	exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, x, visual *Node) *Node {
		// Test the InterleaveTokens function directly to avoid KV cache issues in go backend
		return InterleaveTokens(ctx, x, visual, nil, 5005)
	})
	if err != nil {
		t.Fatalf("Failed to build execution graph: %v", err)
	}

	results, err := exec.Exec(textEmbedsTensor, visualTensor)
	if err != nil {
		t.Fatalf("Execution failed: %v", err)
	}

	output := results[0]
	fmt.Printf("MedGemma Interleaved embeddings shape: %v\n", output.Shape())
	
	expectedSeqLen := textSeqLen + visualSeqLen
	if output.Shape().Dimensions[1] != expectedSeqLen {
		t.Errorf("Sequence length mismatch: got %d, want %d", output.Shape().Dimensions[1], expectedSeqLen)
	}
}

func TestMedGemmaQuantization(t *testing.T) {
	os.Setenv("GOMLX_BACKEND", "go")
	backend, err := backends.New()
	if err != nil {
		t.Fatalf("Failed to create backend: %v", err)
	}
	ctx := context.New()

	exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, r *Node) *Node {
		isMedical := Const(r.Graph(), true)
		indices := QuantizeRadiusAdaptive(r, nil, isMedical)
		return DequantizeRadiusAdaptive(indices, nil, isMedical)
	})
	if err != nil {
		t.Fatalf("Failed to build execution graph: %v", err)
	}

	// Test with a few medical-range radii
	input := tensors.FromFlatDataAndDimensions([]float32{0.05, 1.0, 10.0}, 3)
	results, err := exec.Exec(input)
	if err != nil {
		t.Fatalf("Execution failed: %v", err)
	}

	recon := results[0].Value().([]float32)
	fmt.Printf("Medical Radius Reconstruction: %v -> %v\n", []float32{0.05, 1.0, 10.0}, recon)
	
	if len(recon) != 3 {
		t.Errorf("Expected 3 results, got %d", len(recon))
	}
}
