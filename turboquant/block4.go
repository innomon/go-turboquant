package turboquant

import (
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
)

// TurboGemma4Block implements a single transformer block for Gemma 4.
func TurboGemma4Block(ctx *context.Context, x, ple *Node, cache *KVCache, isEntryLayer bool, numHeads, headDim int, useSWA bool, maxWindow float64, isReasoning, isAudio bool, includeTurbo bool) *Node {
	ctx = ctx.In("gemma4_block")
	hiddenDim := x.Shape().Dimensions[x.Rank()-1]

	// ... (Norm and Projections logic same as before)
	if ple != nil {
		ple_proj := layers.Dense(ctx.In("ple_proj"), ple, true, hiddenDim)
		x = Add(x, ple_proj)
	}

	norm_x := RMSNorm(ctx.In("pre_attn_norm"), x, 1e-6)

	var k, v *Node
	if isEntryLayer {
		k = layers.Dense(ctx.In("k_proj"), norm_x, false, hiddenDim)
		v = layers.Dense(ctx.In("v_proj"), norm_x, false, hiddenDim)
	}
	q := layers.Dense(ctx.In("q_proj"), norm_x, false, hiddenDim)

	attn_out := TurboGemma4Attention(ctx.In("attn"), q, k, v, cache, isEntryLayer, numHeads, headDim, useSWA, maxWindow, isReasoning, isAudio, includeTurbo)

	// 4. Residual 1
	x = Add(x, attn_out)

	// 5. Pre-MLP RMSNorm
	norm_x2 := RMSNorm(ctx.In("pre_mlp_norm"), x, 1e-6)

	// 6. MLP (Gated MLP / GGLU)
	// Gemma uses GeGLU: (Dense(x) * GeLU(Dense(x))) * Dense(x)
	mlp_hidden := hiddenDim * 4 // Typical expansion
	gate := layers.Dense(ctx.In("mlp_gate"), norm_x2, false, mlp_hidden)
	up := layers.Dense(ctx.In("mlp_up"), norm_x2, false, mlp_hidden)
	activated := Mul(activations.Gelu(gate), up)
	down := layers.Dense(ctx.In("mlp_down"), activated, false, hiddenDim)

	// 7. Residual 2
	x = Add(x, down)

	return x
}
