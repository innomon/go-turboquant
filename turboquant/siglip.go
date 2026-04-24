package turboquant

import (
	"fmt"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"
)

// SigLIPConfig holds the hyperparameters for the SigLIP vision encoder.
type SigLIPConfig struct {
	PatchSize       int
	HiddenDim       int
	NumHeads        int
	NumLayers       int
	IntermediateDim int
}

// DefaultMedGemmaSigLIPConfig returns the default configuration for MedGemma 1.5 SigLIP encoder.
func DefaultMedGemmaSigLIPConfig() SigLIPConfig {
	return SigLIPConfig{
		PatchSize:       56,   // 896 / 16 = 56, resulting in 16x16 = 256 tokens.
		HiddenDim:       1152, // SigLIP-Large hidden dimension.
		NumHeads:        16,
		NumLayers:       27,
		IntermediateDim: 4304,
	}
}

// BuildMedGemmaVisionEncoder builds the SigLIP vision encoder graph.
func BuildMedGemmaVisionEncoder(ctx *context.Context, images *Node, config SigLIPConfig) *Node {
	ctx = ctx.In("siglip")
	g := images.Graph()

	// 1. Patchify + Linear Projection
	// images shape: [batch, 896, 896, 3]
	x := layers.Convolution(ctx.In("patch_embed"), images).
		Filters(config.HiddenDim).
		KernelSize(config.PatchSize).
		Strides(config.PatchSize).
		UseBias(false).
		Done()
	// x shape: [batch, 16, 16, hidden_dim]

	// Flatten spatial dimensions
	batchSize := x.Shape().Dimensions[0]
	x = Reshape(x, batchSize, 256, config.HiddenDim)

	// 2. Positional Embeddings
	posEmbed := ctx.VariableWithShape("pos_embed", shapes.Make(images.DType(), 256, config.HiddenDim)).ValueGraph(g)
	// Reshape to [1, 256, config.HiddenDim] for broadcasting
	posEmbed = Reshape(posEmbed, 1, 256, config.HiddenDim)
	x = Add(x, posEmbed)

	// 3. SigLIP Transformer Blocks
	for i := 0; i < config.NumLayers; i++ {
		x = SigLIPBlock(ctx.In(fmt.Sprintf("layer_%d", i)), x, config)
	}

	// 4. Final Layer Norm
	x = layers.LayerNormalization(ctx.In("post_norm"), x).Done()

	return x
}

// SigLIPBlock represents a single SigLIP transformer layer.
func SigLIPBlock(ctx *context.Context, x *Node, config SigLIPConfig) *Node {
	// Pre-Norm Attention
	normX := layers.LayerNormalization(ctx.In("pre_attention_norm"), x).Done()
	headDim := config.HiddenDim / config.NumHeads
	attn := attention.MultiHeadAttention(ctx.In("attention"), normX, normX, normX, config.NumHeads, headDim).
		Dropout(0).
		Done()
	x = Add(x, attn)

	// Pre-Norm MLP
	normX = layers.LayerNormalization(ctx.In("pre_mlp_norm"), x).Done()
	mlpOut := SigLIPMLP(ctx.In("mlp"), normX, config.IntermediateDim)
	x = Add(x, mlpOut)

	return x
}

// SigLIPMLP represents the MLP part of the SigLIP block.
func SigLIPMLP(ctx *context.Context, x *Node, intermediateDim int) *Node {
	hiddenDim := x.Shape().Dimensions[x.Rank()-1]
	gate := layers.Dense(ctx.In("fc1"), x, true, intermediateDim)
	gate = activations.Gelu(gate)
	return layers.Dense(ctx.In("fc2"), gate, true, hiddenDim)
}
