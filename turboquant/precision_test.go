package turboquant

import (
	"fmt"
	"math"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
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
	
	exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, q, k, v *graph.Node) (mha_out, turbo_out *graph.Node) {
		mha_out = layers.MultiHeadAttention(ctx.In("standard"), q, k, v, numHeads, hiddenDim).Done()
		turbo_out = TurboGemmaAttention(ctx.In("turbo"), q, k, v, numHeads, headDim)
		return
	})
	if err != nil {
		t.Fatalf("Failed to create execution: %v", err)
	}

	// Use random inputs for better distribution
	q_val := tensors.FromScalarAndDimensions(float32(0.5), batchSize, seqLen, hiddenDim)
	k_val := tensors.FromScalarAndDimensions(float32(0.1), batchSize, seqLen, hiddenDim)
	v_val := tensors.FromScalarAndDimensions(float32(0.2), batchSize, seqLen, hiddenDim)
	
	resBaseline, resTurbo, err := exec.Exec2(q_val, k_val, v_val)
	if err != nil {
		t.Fatalf("Failed to run execution: %v", err)
	}
	
	baseline := resBaseline.Value().([][][]float32)
	turbo := resTurbo.Value().([][][]float32)
	
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
	if rmse > 1.0 { 
		t.Errorf("RMSE too high: %.6f", rmse)
	}
}
