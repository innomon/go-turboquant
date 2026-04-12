package turboquant

import (
	"fmt"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"
)

func TestTurboGemmaAttention(t *testing.T) {
	backend, err := InitializeBackend()
	if err != nil {
		t.Fatalf("Failed to initialize backend: %v", err)
	}

	ctx := context.New()
	
	// Simulation parameters
	batchSize := 1
	seqLen := 8
	numHeads := 4
	headDim := 32
	hiddenDim := numHeads * headDim
	
	// Use context.Exec to handle parameters automatically
	cache := NewKVCache("test_layer_0")
	exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, q, k, v *graph.Node) (mha_out, turbo_out *graph.Node) {
		// Initialize variables if they don't exist
		if ctx.GetVariableByScopeAndName("/"+cache.Name, "k_cache") == nil {
			cache.InitializeVariables(ctx, batchSize, seqLen, headDim*numHeads/2)
		}
		// 1. Standard Attention (Baseline)
		mha := attention.MultiHeadAttention(ctx.In("standard"), q, k, v, numHeads, hiddenDim)
		mha_out = mha.Done()
		
		// 2. TurboQuant Attention
		turbo_out = TurboGemmaAttention(ctx.In("turbo"), q, k, v, cache, numHeads, headDim)
		return
	})
	if err != nil {
		t.Fatalf("Failed to create execution: %v", err)
	}

	err = ctx.InitializeVariables(backend, nil)
	if err != nil {
		t.Fatalf("Failed to initialize variables: %v", err)
	}

	// Create dummy inputs as tensors
	q_val := tensors.FromScalarAndDimensions(float32(0.5), batchSize, seqLen, hiddenDim)
	k_val := tensors.FromScalarAndDimensions(float32(0.1), batchSize, seqLen, hiddenDim)
	v_val := tensors.FromScalarAndDimensions(float32(0.2), batchSize, seqLen, hiddenDim)
	
	resBaseline, resTurbo, err := exec.Exec2(q_val, k_val, v_val)
	if err != nil {
		t.Fatalf("Failed to run execution: %v", err)
	}
	
	baseline_val := resBaseline.Value().([][][]float32)
	turbo_val := resTurbo.Value().([][][]float32)
	
	// Compare first few values
	fmt.Printf("Baseline[0][0][:4]: %v\n", baseline_val[0][0][:4])
	fmt.Printf("Turbo   [0][0][:4]: %v\n", turbo_val[0][0][:4])
}
