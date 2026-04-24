package turboquant

import (
	"fmt"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
)
// Gemma4Config holds the hyperparameters for the Gemma models.
type Gemma4Config struct {
	ModelType    string  `yaml:"model_type"` // "gemma3" or "gemma4"
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
	IsMedical    bool    `yaml:"is_medical"`
	ThinkTokenID int     `yaml:"think_token_id"`
	AudioTokenID int     `yaml:"audio_token_id"`
	ImageTokenID int     `yaml:"image_token_id"`
}

// DefaultGemma4E4BConfig returns a default configuration for Gemma 4 E4B.
func DefaultGemma4E4BConfig() Gemma4Config {
	return Gemma4Config{
		ModelType:    "gemma4",
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
		ImageTokenID: 5005, // <|image|>
	}
}

// DefaultGemma3Config returns a default configuration for Gemma 3.
func DefaultGemma3Config() Gemma4Config {
	return Gemma4Config{
		ModelType:    "gemma3",
		VocabSize:    256000,
		NumLayers:    28,
		NumHeads:     16,
		HeadDim:      128,
		HiddenDim:    2048,
		IncludeTurbo: true,
		ImageTokenID: 5005,
	}
}

// BuildGemma3Model builds the full Gemma 3 model graph.
func BuildGemma3Model(ctx *context.Context, tokens, visualTokens *Node, config Gemma4Config) []*Node {
	ctx = ctx.In("gemma3")
	g := tokens.Graph()

	// 1. Embedding
	embedVar := ctx.In("embedding").VariableWithShape("weight", shapes.Make(dtypes.Float32, config.VocabSize, config.HiddenDim))
	embedWeight := embedVar.ValueGraph(g)
	x := layers.Embedding(ctx.In("embedding"), tokens, dtypes.Float32, config.VocabSize, config.HiddenDim)

	// 2. Multimodal Interleaving
	var isMedical *Node
	if config.IsMedical {
		isMedical = Const(g, true)
	}

	x = InterleaveTokens(ctx, x, visualTokens, tokens, config.ImageTokenID)

	// 3. Transformer Blocks
	var caches []*KVCache
	maxSeqLen := 8192
	packedDim := config.HiddenDim / 2
	if !config.IncludeTurbo {
		packedDim = config.HiddenDim
	}
	batchSize := tokens.Shape().Dimensions[0]

	for i := 0; i < config.NumLayers; i++ {
		cacheName := fmt.Sprintf("kv_cache_%d", i)
		cache := NewKVCache(cacheName)
		if ctx.GetVariableByScopeAndName("/"+cacheName, "k_cache") == nil {
			cacheDType := dtypes.Uint8
			if !config.IncludeTurbo {
				cacheDType = dtypes.Float32
			}
			cache.InitializeVariables(ctx, batchSize, maxSeqLen, packedDim, cacheDType)
		}
		caches = append(caches, cache)
	}

	intermediateDim := config.HiddenDim * 4 // Generic intermediate dim
	for i := 0; i < config.NumLayers; i++ {
		layerCtx := ctx.In(fmt.Sprintf("layer_%d", i))
		x = TurboGemmaBlock(layerCtx, x, caches[i], config.NumHeads, config.HeadDim, intermediateDim, nil, isMedical)
	}

	// 3. Final Norm
	x = RMSNorm(ctx.In("final_norm"), x, 1e-6)

	// 4. Head
	logits := MatMul(x, Transpose(embedWeight, 0, 1))
	return []*Node{logits}
}

// InterleaveTokens merges text embeddings and visual tokens.
// In this prototype, it prepends visual tokens if they are provided.
func InterleaveTokens(ctx *context.Context, textEmbeddings, visualTokens, tokens *Node, imageTokenID int) *Node {
	if visualTokens == nil {
		return textEmbeddings
	}
	// For MedGemma 1.5 prototype, we assume visual tokens are prepended.
	// A more complex implementation would use the imageTokenID to find the insertion point.
	return Concatenate([]*Node{visualTokens, textEmbeddings}, 1)
}

// BuildGemma4Model builds the full Gemma 4 model graph.
// It returns a slice of nodes containing the base logits and MTP heads.
func BuildGemma4Model(ctx *context.Context, tokens, ple *Node, config Gemma4Config) []*Node {
	ctx = ctx.In("gemma4")
	g := tokens.Graph()

	// 0. Adaptive State Detection (Reasoning / Audio / Medical)
	// Check if any token in the current sequence triggers high-precision mode.
	isReasoning := LogicalAny(Equal(tokens, Scalar(g, tokens.DType(), config.ThinkTokenID)))
	isAudio := LogicalAny(Equal(tokens, Scalar(g, tokens.DType(), config.AudioTokenID)))
	var isMedical *Node
	if config.IsMedical {
		isMedical = Const(g, true)
	}
	
	// 1. Embedding
	// embedWeight shape: [VocabSize, HiddenDim]
	embedVar := ctx.In("embedding").VariableWithShape("weight", shapes.Make(dtypes.Float32, config.VocabSize, config.HiddenDim))
	embedWeight := embedVar.ValueGraph(tokens.Graph())
	
	// Use the built-in embedding layer (assuming it handles weights internally via context)
	x := layers.Embedding(ctx.In("embedding"), tokens, dtypes.Float32, config.VocabSize, config.HiddenDim)
	
	// 2. Transformer Blocks Initialization
	// We use a shared KV cache every 8 layers (example for Gemma 4).
	var caches []*KVCache
	maxSeqLen := 8192 // Fixed for this graph
	packedDim := config.HiddenDim / 2
	if !config.IncludeTurbo {
		packedDim = config.HiddenDim
	}
	batchSize := tokens.Shape().Dimensions[0]

	for i := 0; i < config.NumLayers; i++ {
		cacheName := fmt.Sprintf("kv_cache_group_%d", i/8)
		cache := NewKVCache(cacheName)
		if i%8 == 0 {
			cacheDType := dtypes.Uint8
			if !config.IncludeTurbo {
				cacheDType = dtypes.Float32
			}
			cache.InitializeVariables(ctx, batchSize, maxSeqLen, packedDim, cacheDType)
		}
		caches = append(caches, cache)
	}

	for i := 0; i < config.NumLayers; i++ {
		layerCtx := ctx.In(fmt.Sprintf("layer_%d", i))
		isEntryLayer := (i % 8 == 0)
		
		// K and V projections are only needed at the entry layer of a shared group.
		// These will be calculated inside the block if isEntryLayer is true.
		x = TurboGemma4Block(layerCtx, x, ple, nil, nil, caches[i], isEntryLayer, config.NumHeads, config.HeadDim, config.UseSWA, config.MaxWindow, isReasoning, isAudio, isMedical, config.IncludeTurbo)
	}

	// 3. Final Norm
	x = layers.RMSNorm(ctx.In("final_norm"), x).WithNormalizationAxes(-1).Done()

	// 4. MTP / LM Heads
	var logits []*Node
	if config.IncludeMTP {
		logits = BuildGemma4MTP(ctx, x, config.VocabSize, embedWeight, config.NumMTPHeads)
	} else {
		// Standard LM Head
		logits = []*Node{MatMul(x, Transpose(embedWeight, 0, 1))}
	}
	
	return logits
}
