package turboquant

import (
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
)

// RMSNorm implements Root Mean Square Layer Normalization.
func RMSNorm(ctx *context.Context, x *Node, epsilon float64) *Node {
	ctx = ctx.In("rms_norm")
	g := x.Graph()
	ms := ReduceMean(Square(x), -1)
	ms = ExpandDims(ms, -1) // Keep rank for broadcasting
	invRms := Inverse(Sqrt(AddScalar(ms, epsilon)))
	normalized := Mul(x, invRms)
	hiddenDim := x.Shape().Dimensions[x.Rank()-1]
	gamma := ctx.VariableWithShape("weight", shapes.Make(x.DType(), hiddenDim)).ValueGraph(g)
	// Reshape gamma to [1, 1, ..., hiddenDim] to match rank of x
	gammaDims := make([]int, x.Rank())
	for i := 0; i < len(gammaDims)-1; i++ {
		gammaDims[i] = 1
	}
	gammaDims[len(gammaDims)-1] = hiddenDim
	gamma = Reshape(gamma, gammaDims...)
	return Mul(normalized, gamma)
}

// MLP block for Gemma 3.
func MLP(ctx *context.Context, x *Node, intermediateDim int) *Node {
	ctx = ctx.In("mlp")
	gate := layers.Dense(ctx.In("gate_proj"), x, true, intermediateDim)
	gate = activations.Gelu(gate)
	up := layers.Dense(ctx.In("up_proj"), x, true, intermediateDim)
	intermediate := Mul(gate, up)
	hiddenDim := x.Shape().Dimensions[x.Rank()-1]
	return layers.Dense(ctx.In("down_proj"), intermediate, true, hiddenDim)
}

// TurboGemmaBlock represents a single Gemma 3 transformer layer with TurboQuant attention.
func TurboGemmaBlock(ctx *context.Context, x *Node, cache *KVCache, numHeads, headDim, intermediateDim int, isAudio, isMedical *Node) *Node {
	// 1. Pre-Attention Norm
	normX := RMSNorm(ctx.In("pre_attention_norm"), x, 1e-6)

	// 2. TurboQuant Attention
	// For this block, we assume k and v are projected from the same normX
	attn := TurboGemmaAttentionAdaptive(ctx.In("attention"), normX, normX, normX, cache, numHeads, headDim, isAudio, isMedical)
	x = Add(x, attn)

	// 3. Pre-MLP Norm
	normX = RMSNorm(ctx.In("pre_mlp_norm"), x, 1e-6)

	// 4. MLP
	mlpOut := MLP(ctx, normX, intermediateDim)
	x = Add(x, mlpOut)

	return x
}
