package turboquant

import (
	"fmt"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention"
)

// TurboGemma4Attention implements a Gemma 4 attention mechanism with persistent KV cache.
func TurboGemma4Attention(ctx *context.Context, q, k, v *Node, cache *KVCache, isEntryLayer bool, numHeads, headDim int, useSWA bool, maxWindow float64, isReasoning, isAudio *Node, includeTurbo bool) *Node {
	ctx = ctx.In("turbo_attention_g4")
	g := q.Graph()
	windowSize := 4096 // Default SWA window size for Gemma 4

	// 1. RoPE for Q
	origQShape := q.Shape()
	batchSize := origQShape.Dimensions[0]
	seqLen := origQShape.Dimensions[1]
	
	// Retrieve current length from cache to use as RoPE offset.
	absScope := "/" + cache.Name
	lenVarObj := ctx.GetVariableByScopeAndName(absScope, "current_len")
	if lenVarObj == nil {
		panic(fmt.Sprintf("Variable 'current_len' not found in scope %s (abs: %s)", cache.Name, absScope))
	}
	lenVar := lenVarObj.ValueGraph(g)
	currentLen := Reshape(lenVar) // []
	
	q = Reshape(q, batchSize, seqLen, numHeads, headDim)
	if useSWA {
		q = ApplyRoPEWithOffset(q, currentLen, 10000.0)
	} else {
		// Global layer uses Proportional RoPE
		q = ApplyProportionalRoPE(q, currentLen, 10000.0, currentLen, maxWindow)
	}
	q = Reshape(q, batchSize, seqLen, numHeads*headDim)

	// 2. Shared KV Cache logic
	if isEntryLayer {
		// RoPE for K (applied before storage)
		k = Reshape(k, batchSize, seqLen, numHeads, headDim)
		if useSWA {
			k = ApplyRoPEWithOffset(k, currentLen, 10000.0)
		} else {
			k = ApplyProportionalRoPE(k, currentLen, 10000.0, currentLen, maxWindow)
		}
		k = Reshape(k, batchSize, seqLen, numHeads*headDim)

		if includeTurbo {
			// Quantize and update cache
			hiddenDim := k.Shape().Dimensions[k.Rank()-1]
			mid := hiddenDim / 2
			
			k_x := Slice(k, AxisRange(), AxisRange(), AxisRange(0, mid))
			k_y := Slice(k, AxisRange(), AxisRange(), AxisRange(mid, hiddenDim))
			v_x := Slice(v, AxisRange(), AxisRange(), AxisRange(0, mid))
			v_y := Slice(v, AxisRange(), AxisRange(), AxisRange(mid, hiddenDim))
			
			k_packed := TurboQuantizeAdaptive(k_x, k_y, isReasoning, isAudio)
			v_packed := TurboQuantizeAdaptive(v_x, v_y, isReasoning, isAudio)
			cache.Update(ctx, k_packed, v_packed)
		} else {
			// No TurboQuant: Store raw float32
			cache.Update(ctx, k, v)
		}
	}

	// 3. Dequantize or use raw contents
	var k_prime, v_prime *Node
	k_full, v_full, mask := cache.GetContents(ctx, g)
	
	if cache.DType() == dtypes.Uint8 {
		k_x_recon, k_y_recon := TurboDequantizeAdaptive(k_full, isReasoning, isAudio)
		v_x_recon, v_y_recon := TurboDequantizeAdaptive(v_full, isReasoning, isAudio)
		
		k_prime = Concatenate([]*Node{k_x_recon, k_y_recon}, 2)
		v_prime = Concatenate([]*Node{v_x_recon, v_y_recon}, 2)
	} else {
		k_prime = k_full
		v_prime = v_full
	}

	// SWA Slicing
	if useSWA {
		cacheSeqLen := k_prime.Shape().Dimensions[1]
		if cacheSeqLen > windowSize {
			k_prime = Slice(k_prime, AxisRange(), AxisRange(cacheSeqLen-windowSize, cacheSeqLen), AxisRange())
			v_prime = Slice(v_prime, AxisRange(), AxisRange(cacheSeqLen-windowSize, cacheSeqLen), AxisRange())
			// We should also slice the mask if we are using SWA
			mask = Slice(mask, AxisRange(), AxisRange(cacheSeqLen-windowSize, cacheSeqLen))
		}
	}

	mha := attention.MultiHeadAttention(ctx, q, k_prime, v_prime, numHeads, numHeads*headDim)
	return mha.SetKeyMask(mask).Done()
}
