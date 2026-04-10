package turboquant

import (
	"fmt"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
)

// Gemma4Config holds the hyperparameters for the Gemma 4 model.
type Gemma4Config struct {
	VocabSize    int     `yaml:"vocab_size"`
	NumLayers    int     `yaml:"num_layers"`
	NumHeads     int     `yaml:"num_heads"`
	HeadDim      int     `yaml:"head_dim"`
	HiddenDim    int     `yaml:"hidden_dim"`
	PLEDim       int     `yaml:"ple_dim"`
	UseSWA       bool    `yaml:"use_swa"`
	MaxWindow    float64 `yaml:"max_window"`
	NumMTPHeads  int     `yaml:"num_mtp_heads"`
	IncludeMTP   bool    `yaml:"include_mtp"`
	IncludeTurbo bool    `yaml:"include_turbo"`
	ThinkTokenID int     `yaml:"think_token_id"`
	AudioTokenID int     `yaml:"audio_token_id"`
}

// DefaultGemma4E4BConfig returns a default configuration for Gemma 4 E4B.
func DefaultGemma4E4BConfig() Gemma4Config {
	return Gemma4Config{
		VocabSize:    256000,
		NumLayers:    24,
		NumHeads:     16,
		HeadDim:      128,
		HiddenDim:    2048,
		PLEDim:       512,
		UseSWA:       true,
		MaxWindow:    8192.0,
		NumMTPHeads:  3,
		IncludeMTP:   true,
		IncludeTurbo: true,
		ThinkTokenID: 5001, // <|think|>
		AudioTokenID: 5004, // <|audio|>
	}
}

// BuildGemma4Model builds the full Gemma 4 model graph.
// It returns a single Tuple node containing the base logits and MTP heads.
func BuildGemma4Model(ctx *context.Context, tokens, ple *Node, config Gemma4Config) *Node {
	ctx = ctx.In("gemma4")
	g := tokens.Graph()

	// 0. Adaptive State Detection (Reasoning / Audio)
	// Check if any token in the current sequence triggers high-precision mode.
	isReasoning := Any(Equal(tokens, Scalar(g, tokens.DType(), config.ThinkTokenID)))
	isAudio := Any(Equal(tokens, Scalar(g, tokens.DType(), config.AudioTokenID)))
	
	// 1. Embedding
	// embedWeight shape: [VocabSize, HiddenDim]
	embedVar := ctx.In("embedding").VariableWithShape("weight", config.VocabSize, config.HiddenDim)
	embedWeight := embedVar.Node(tokens.Graph())
	
	// Use the built-in embedding layer (assuming it handles weights internally via context)
	x := layers.Embedding(ctx.In("embedding"), tokens, config.VocabSize, config.HiddenDim)
	
	// Scaling factor for embeddings (Gemma style)
	x = MulScalar(x, 1.0) // Gemma 2/4 often scales by sqrt(hiddenDim)

	// 2. Transformer Blocks
	var caches []*KVCache
	for i := 0; i < config.NumLayers; i++ {
		caches = append(caches, NewSharedKVCache(tokens.Graph()))
	}

	for i := 0; i < config.NumLayers; i++ {
		layerCtx := ctx.In(fmt.Sprintf("layer_%d", i))
		isEntryLayer := (i == 0) 
		
		x = TurboGemma4Block(layerCtx, x, ple, caches[i], isEntryLayer, config.NumHeads, config.HeadDim, config.UseSWA, config.MaxWindow, isReasoning, isAudio, config.IncludeTurbo)
	}

	// 3. Final Norm
	x = layers.RMSNorm(ctx.In("final_norm"), x, -1).Done()

	// 4. MTP / LM Heads
	var logits []*Node
	if config.IncludeMTP {
		// Use embedWeight for unembedding (weight sharing)
		logits = BuildGemma4MTP(ctx, x, config.VocabSize, embedWeight, config.NumMTPHeads)
	} else {
		// Standard LM Head
		logits = []*Node{MatMul(x, Transpose(embedWeight))}
	}
	
	return Tuple(logits...)
}
