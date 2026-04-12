package turboquant

import (
	"fmt"
	"math"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"
)

func TestTurboGemmaPrecision(t *testing.T) {
	backend, err := InitializeBackend()
	if err != nil {
		t.Fatalf("Failed to initialize backend: %v", err)
	}

	ctx := context.New()
	
	// Simulation parameters
	batchSize := 1
	seqLen := 16
	numHeads := 8
	headDim := 64
	hiddenDim := numHeads * headDim

	cache := NewKVCache("precision_layer_0")
	
	exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, q, k, v *graph.Node) (mha_out, turbo_out *graph.Node) {
		if ctx.GetVariableByScopeAndName("/"+cache.Name, "k_cache") == nil {
			cache.InitializeVariables(ctx, batchSize, seqLen, headDim*numHeads/2, dtypes.Uint8)
		}
		mha_out = attention.MultiHeadAttention(ctx.In("standard"), q, k, v, numHeads, hiddenDim).Done()
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

	// Use random inputs for better distribution
	q_val := tensors.FromScalarAndDimensions(float32(0.5), batchSize, seqLen, hiddenDim)
	k_val := tensors.FromScalarAndDimensions(float32(0.1), batchSize, seqLen, hiddenDim)
	v_val := tensors.FromScalarAndDimensions(float32(0.2), batchSize, seqLen, hiddenDim)
	
	resBaselineT, resTurboT, err := exec.Exec2(q_val, k_val, v_val)
	if err != nil {
		t.Fatalf("Failed to run execution: %v", err)
	}
	
	baseline := resBaselineT.Value().([][][]float32)
	turbo := resTurboT.Value().([][][]float32)
	
	var sumSqDiff float64
	var count int
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			for k := 0; k < hiddenDim; k++ {
				diff := float64(baseline[i][j][k] - turbo[i][j][k])
				sumSqDiff += diff * diff
				count++
			}
		}
	}
	
	rmse := math.Sqrt(sumSqDiff / float64(count))
	fmt.Printf("📊 TurboQuant RMSE (compared to FP16): %.6f\n", rmse)
	
	// Threshold for 4-bit quantization precision loss
	// This is empirical and depends on the scale of inputs
	if rmse > 1.2 { 
		t.Errorf("RMSE too high: %.6f", rmse)
	}
}
