package turboquant

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
)

func BenchmarkAttention(b *testing.B) {
	backend, err := InitializeBackend()
	if err != nil {
		b.Fatalf("Failed to initialize backend: %v", err)
	}

	ctx := context.New()

	batchSize := 1
	seqLen := 128 // slightly larger sequence for benchmark
	numHeads := 4
	headDim := 32
	hiddenDim := numHeads * headDim

	// Exec for Standard Attention
	execStandard, err := context.NewExec(backend, ctx, func(ctx *context.Context, q, k, v *graph.Node) *graph.Node {
		mha := layers.MultiHeadAttention(ctx.In("standard"), q, k, v, numHeads, hiddenDim)
		return mha.Done()
	})
	if err != nil {
		b.Fatalf("Failed to create standard execution: %v", err)
	}

	// Exec for TurboQuant Attention
	execTurbo, err := context.NewExec(backend, ctx, func(ctx *context.Context, q, k, v *graph.Node) *graph.Node {
		return TurboGemmaAttention(ctx.In("turbo"), q, k, v, numHeads, headDim)
	})
	if err != nil {
		b.Fatalf("Failed to create turbo execution: %v", err)
	}

	q_val := tensors.FromScalarAndDimensions(float32(0.5), batchSize, seqLen, hiddenDim)
	k_val := tensors.FromScalarAndDimensions(float32(0.1), batchSize, seqLen, hiddenDim)
	v_val := tensors.FromScalarAndDimensions(float32(0.2), batchSize, seqLen, hiddenDim)

	b.Run("Standard", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := execStandard.Exec1(q_val, k_val, v_val)
			if err != nil {
				b.Fatalf("Standard execution failed: %v", err)
			}
		}
	})

	b.Run("TurboQuant", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := execTurbo.Exec1(q_val, k_val, v_val)
			if err != nil {
				b.Fatalf("Turbo execution failed: %v", err)
			}
		}
	})
}
