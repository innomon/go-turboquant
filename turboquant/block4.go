package turboquant

import (
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
)

// TurboGemma4Block implements a single transformer block for Gemma 4.
func TurboGemma4Block(ctx *context.Context, x, ple, k, v *Node, cache *KVCache, isEntryLayer bool, numHeads, headDim int, useSWA bool, maxWindow float64, isReasoning, isAudio *Node, includeTurbo bool) *Node {
	ctx = ctx.In("gemma4_block")
	hiddenDim := x.Shape().Dimensions[x.Rank()-1]

	// 1. PLE (Per-Layer Embeddings)
	if ple != nil {
		ple_proj := layers.Dense(ctx.In("ple_proj"), ple, true, hiddenDim)
		x = Add(x, ple_proj)
	}

	// 2. Pre-Attention RMSNorm
	norm_x := layers.RMSNorm(ctx.In("pre_attn_norm"), x).Done()

	// 3. QKV Projections
	var k_proj, v_proj *Node
	if isEntryLayer {
		k_proj = layers.Dense(ctx.In("k_proj"), norm_x, false, hiddenDim)
		v_proj = layers.Dense(ctx.In("v_proj"), norm_x, false, hiddenDim)
	}
	q_proj := layers.Dense(ctx.In("q_proj"), norm_x, false, hiddenDim)

	// 4. TurboGemma4Attention
	attn_out := TurboGemma4Attention(ctx.In("attn"), q_proj, k_proj, v_proj, cache, isEntryLayer, numHeads, headDim, useSWA, maxWindow, isReasoning, isAudio, includeTurbo)

	// 5. Residual 1
	x = Add(x, attn_out)

	// 6. Pre-MLP RMSNorm
	norm_x2 := layers.RMSNorm(ctx.In("pre_mlp_norm"), x).Done()

	// 7. MLP (GeGLU)
	mlp_hidden := hiddenDim * 4
	gate := layers.Dense(ctx.In("mlp_gate"), norm_x2, false, mlp_hidden)
	up := layers.Dense(ctx.In("mlp_up"), norm_x2, false, mlp_hidden)
	activated := Mul(activations.Gelu(gate), up)
	down := layers.Dense(ctx.In("mlp_down"), activated, false, hiddenDim)

	// 8. Residual 2
	x = Add(x, down)

	return x
}
