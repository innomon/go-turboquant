package turboquant

import (
	"fmt"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"
)

// KVCache represents a compressed KV cache storage backed by persistent variables.
type KVCache struct {
	Name        string
	IsShared    bool         // If true, this cache is shared across multiple layers in a single pass.
	StoredDType dtypes.DType // DType of the stored cache (Uint8 for packed, Float32 for raw)
}

// NewKVCache initializes a KV cache handle.
func NewKVCache(name string) *KVCache {
	return &KVCache{Name: name, StoredDType: dtypes.Uint8} // Default to Uint8 for Turbo
}

// NewSharedKVCache initializes a shared KV cache handle.
func NewSharedKVCache(name string) *KVCache {
	return &KVCache{Name: name, IsShared: true, StoredDType: dtypes.Uint8}
}

// InitializeVariables pre-allocates the persistent variables in the context.
func (cache *KVCache) InitializeVariables(ctx *context.Context, batchSize, maxSeqLen, packedDim int, dtype dtypes.DType) {
	cache.StoredDType = dtype
	cacheCtx := ctx.In(cache.Name)
	cacheCtx.VariableWithShape("k_cache", shapes.Make(dtype, batchSize, maxSeqLen, packedDim)).SetTrainable(false)
	cacheCtx.VariableWithShape("v_cache", shapes.Make(dtype, batchSize, maxSeqLen, packedDim)).SetTrainable(false)
	cacheCtx.VariableWithShape("current_len", shapes.Make(dtypes.Int32, 1)).SetTrainable(false)
}

func (cache *KVCache) DType() dtypes.DType {
	return cache.StoredDType
}

// Update updates the persistent KV cache with new packed tensors using DynamicUpdateSlice.
func (cache *KVCache) Update(ctx *context.Context, kPacked, vPacked *Node) {
	g := kPacked.Graph()
	// Access variables relative to the cache name with absolute scope.
	absScope := "/" + cache.Name
	kCacheVar := ctx.GetVariableByScopeAndName(absScope, "k_cache")
	vCacheVar := ctx.GetVariableByScopeAndName(absScope, "v_cache")
	lenVar := ctx.GetVariableByScopeAndName(absScope, "current_len")

	if kCacheVar == nil {
		panic(fmt.Sprintf("KV Cache variable 'k_cache' not found in scope %s (abs: %s)", cache.Name, absScope))
	}

	currentLen := Reshape(lenVar.ValueGraph(g)) // []

	startIndices := []*Node{
		Scalar(g, dtypes.Int32, 0),
		ConvertType(currentLen, dtypes.Int32),
		Scalar(g, dtypes.Int32, 0),
	}

	newKCache := DynamicUpdateSlice(kCacheVar.ValueGraph(g), kPacked, startIndices)
	newVCache := DynamicUpdateSlice(vCacheVar.ValueGraph(g), vPacked, startIndices)

	kCacheVar.SetValueGraph(newKCache)
	vCacheVar.SetValueGraph(newVCache)

	numNewTokens := Scalar(g, dtypes.Int32, kPacked.Shape().Dimensions[1])
	newLen := Add(currentLen, numNewTokens)
	lenVar.SetValueGraph(Reshape(newLen, 1))
}

// GetContents returns the KV cache and a mask for valid tokens.
func (cache *KVCache) GetContents(ctx *context.Context, g *Graph) (kPacked, vPacked, mask *Node) {
	absScope := "/" + cache.Name
	kCacheVarObj := ctx.GetVariableByScopeAndName(absScope, "k_cache")
	if kCacheVarObj == nil {
		panic(fmt.Sprintf("KV Cache variable 'k_cache' not found in scope %s (abs: %s)", cache.Name, absScope))
	}

	kCacheVar := kCacheVarObj.ValueGraph(g)
	vCacheVar := ctx.GetVariableByScopeAndName(absScope, "v_cache").ValueGraph(g)
	lenVar := ctx.GetVariableByScopeAndName(absScope, "current_len").ValueGraph(g)
	currentLen := Reshape(lenVar) // []

	// Create mask: [batch, total_seq]
	seqLen := kCacheVar.Shape().Dimensions[1]
	indices := Iota(g, shapes.Make(dtypes.Int32, seqLen), 0)
	mask = LessThan(indices, ConvertType(currentLen, dtypes.Int32)) // [total_seq]
	batchSize := kCacheVar.Shape().Dimensions[0]
	mask = Reshape(mask, 1, seqLen)
	mask = BroadcastToDims(mask, batchSize, seqLen)

	kPacked = kCacheVar
	vPacked = vCacheVar
	return
}

// TurboGemmaAttention implements a Gemma 3 attention mechanism with integrated
func TurboGemmaAttention(ctx *context.Context, q, k, v *Node, cache *KVCache, numHeads, headDim int) *Node {
	g := q.Graph()
	// 1. Apply RoPE to Q and K
	batchSize := q.Shape().Dimensions[0]
	qSeqLen := q.Shape().Dimensions[1]

	// Access cache with absolute scope
	absScope := "/" + cache.Name
	lenVarObj := ctx.GetVariableByScopeAndName(absScope, "current_len")
	if lenVarObj == nil {
		panic(fmt.Sprintf("Variable 'current_len' not found in scope %s (abs: %s)", cache.Name, absScope))
	}
	lenVar := lenVarObj.ValueGraph(g)
	currentLen := Reshape(lenVar) // []

	q = Reshape(q, batchSize, qSeqLen, numHeads, headDim)
	k = Reshape(k, batchSize, qSeqLen, numHeads, headDim)

	q = ApplyRoPEWithOffset(q, currentLen, 10000.0)
	k = ApplyRoPEWithOffset(k, currentLen, 10000.0)

	q = Reshape(q, batchSize, qSeqLen, numHeads*headDim)
	k = Reshape(k, batchSize, qSeqLen, numHeads*headDim)

	// 2. KV Quantization (Polar + QJL)
	hiddenAxis := k.Rank() - 1
	hiddenDim := k.Shape().Dimensions[hiddenAxis]
	mid := hiddenDim / 2

	k_x := Slice(k, AxisRange(), AxisRange(), AxisRange(0, mid))
	k_y := Slice(k, AxisRange(), AxisRange(), AxisRange(mid, hiddenDim))
	v_x := Slice(v, AxisRange(), AxisRange(), AxisRange(0, mid))
	v_y := Slice(v, AxisRange(), AxisRange(), AxisRange(mid, hiddenDim))

	k_packed := TurboQuantize(k_x, k_y)
	v_packed := TurboQuantize(v_x, v_y)
	cache.Update(ctx, k_packed, v_packed)

	// 3. Retrieve and Dequantize full cache
	k_packed_full, v_packed_full, mask := cache.GetContents(ctx, g)

	k_x_recon, k_y_recon := TurboDequantize(k_packed_full)
	v_x_recon, v_y_recon := TurboDequantize(v_packed_full)

	k_prime := Concatenate([]*Node{k_x_recon, k_y_recon}, 2)
	v_prime := Concatenate([]*Node{v_x_recon, v_y_recon}, 2)

	// 4. Standard Multi-Head Attention using dequantized values
	mha := attention.MultiHeadAttention(ctx.In("attention"), q, k_prime, v_prime, numHeads, numHeads*headDim)
	return mha.SetKeyMask(mask).Done()
}
