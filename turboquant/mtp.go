package turboquant

import (
	"fmt"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
)

// MTPHead implements a single Multi-Token Prediction head.
// It consists of a ResNet-style skip connection, a dense layer, SiLU activation,
// and a final logits projection.
func MTPHead(ctx *context.Context, hiddenStates *Node, vocabSize int, unembeddingWeight *Node) *Node {
	ctx = ctx.In("mtp_head")
	hiddenDim := hiddenStates.Shape().Dimensions[hiddenStates.Rank()-1]

	// 1. ResNet-style skip connection + Dense layer
	// We project hiddenStates to the same dimension, apply SiLU, and add back.
	projected := layers.Dense(ctx.In("proj"), hiddenStates, true, hiddenDim)
	activated := activations.Swish(projected)
	
	// Residual connection
	h := Add(hiddenStates, activated)

	// 2. Logits projection
	// If unembeddingWeight is provided, we use it (weight sharing).
	// Otherwise, we create a new dense layer.
	var logits *Node
	if unembeddingWeight != nil {
		// logits = h @ unembeddingWeight.T
		// Assuming unembeddingWeight is [vocabSize, hiddenDim]
		logits = MatMul(h, Transpose(unembeddingWeight, 0, 1))
	} else {
		logits = layers.Dense(ctx.In("logits"), h, false, vocabSize)
	}

	return logits
}

// BuildGemma4MTP integrates MTP heads into the Gemma 4 backbone.
func BuildGemma4MTP(ctx *context.Context, lastHiddenState *Node, vocabSize int, unembeddingWeight *Node, numMTPHeads int) []*Node {
	ctx = ctx.In("mtp")
	
	// results[0] is the standard LM head (predicting t+1)
	// results[1...numMTPHeads] are the MTP heads (predicting t+2, t+3, ...)
	results := make([]*Node, numMTPHeads+1)
	
	// Base head
	if unembeddingWeight != nil {
		results[0] = MatMul(lastHiddenState, Transpose(unembeddingWeight, 0, 1))
	} else {
		results[0] = layers.Dense(ctx.In("base_logits"), lastHiddenState, false, vocabSize)
	}

	// MTP heads
	for i := 1; i <= numMTPHeads; i++ {
		results[i] = MTPHead(ctx.In(fmt.Sprintf("head_%d", i)), lastHiddenState, vocabSize, unembeddingWeight)
	}

	return results
}
