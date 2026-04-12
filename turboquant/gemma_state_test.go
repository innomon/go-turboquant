package turboquant

import (
	"fmt"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

func TestKVCacheStatefulness(t *testing.T) {
	backend, err := InitializeBackend()
	if err != nil {
		t.Fatalf("Failed to initialize backend: %v", err)
	}

	ctx := context.New()
	batchSize := 1
	numHeads := 2
	headDim := 8
	hiddenDim := numHeads * headDim
	maxSeqLen := 10
	packedDim := hiddenDim / 2 // TurboQuant packs pairs

	cache := NewKVCache("test_layer_0")
	cache.InitializeVariables(ctx, batchSize, maxSeqLen, packedDim, dtypes.Uint8)
	
	// Define the graph for one step.
	exec, err := context.NewExec(backend, ctx, func(ctx *context.Context, q, k, v *Node) *Node {
		return TurboGemmaAttention(ctx, q, k, v, cache, numHeads, headDim)
	})
	if err != nil {
		t.Fatalf("Failed to create execution: %v", err)
	}

	// Initialize variables in backend before first run
	err = ctx.InitializeVariables(backend, nil)
	if err != nil {
		t.Fatalf("Failed to initialize variables: %v", err)
	}

	// Step 1: Prompt (SeqLen = 4)
	seqLen1 := 4
	q1 := tensors.FromScalarAndDimensions(float32(0.1), batchSize, seqLen1, hiddenDim)
	k1 := tensors.FromScalarAndDimensions(float32(0.2), batchSize, seqLen1, hiddenDim)
	v1 := tensors.FromScalarAndDimensions(float32(0.3), batchSize, seqLen1, hiddenDim)

	_, err = exec.Exec(q1, k1, v1)
	if err != nil {
		t.Fatalf("Step 1 failed: %v", err)
	}

	// Verify current length
	absScope := "/" + cache.Name
	currLenVar := ctx.GetVariableByScopeAndName(absScope, "current_len")
	if currLenVar == nil {
		t.Fatalf("Variable 'current_len' not found in scope %s (abs: %s)", cache.Name, absScope)
	}
	vTensor, _ := currLenVar.Value()
	currLen := vTensor.Value().([]int32)[0]
	if currLen != 4 {
		t.Errorf("Expected current_len 4 after Step 1, got %d", currLen)
	}

	// Step 2: Decode (SeqLen = 1)
	seqLen2 := 1
	q2 := tensors.FromScalarAndDimensions(float32(0.4), batchSize, seqLen2, hiddenDim)
	k2 := tensors.FromScalarAndDimensions(float32(0.5), batchSize, seqLen2, hiddenDim)
	v2 := tensors.FromScalarAndDimensions(float32(0.6), batchSize, seqLen2, hiddenDim)

	_, err = exec.Exec(q2, k2, v2)
	if err != nil {
		t.Fatalf("Step 2 failed: %v", err)
	}

	// Verify current length again
	vTensor, _ = currLenVar.Value()
	currLen = vTensor.Value().([]int32)[0]


	if currLen != 5 {
		t.Errorf("Expected current_len 5 after Step 2, got %d", currLen)
	}
	
	fmt.Printf("✅ KVCache correctly maintained state across multiple calls. Final Length: %d\n", currLen)
}
