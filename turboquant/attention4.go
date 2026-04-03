package turboquant

import (
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
)

// TurboGemma4Attention implements a Gemma 4 attention mechanism.
// It supports Shared KV Cache, Sliding Window Attention (SWA), Dual RoPE, Adaptive QJL, and Audio-optimized codebooks.
func TurboGemma4Attention(ctx *context.Context, q, k, v *Node, cache *KVCache, isEntryLayer bool, numHeads, headDim int, useSWA bool, maxWindow float64, isReasoning, isAudio bool) *Node {
	ctx = ctx.In("turbo_attention_g4")
	windowSize := 4096 // Default SWA window size for Gemma 4

	// 1. Dual RoPE
	// Reshape Q to [batch, seq, numHeads, head_dim] for RoPE
	origQShape := q.Shape()
	batchSize := origQShape.Dimensions[0]
	seqLen := origQShape.Dimensions[1]
	q = Reshape(q, batchSize, seqLen, numHeads, headDim)

	if useSWA {
		q = ApplyRoPE(q, 10000.0)
	} else {
		// Global layer uses Proportional RoPE
		currentSeqLen := float64(seqLen)
		q = ApplyProportionalRoPE(q, 10000.0, currentSeqLen, maxWindow)
	}

	// Flatten Q back to [batch, seq, hidden]
	q = Reshape(q, batchSize, seqLen, numHeads*headDim)

	// 2. Shared KV Cache logic
	if isEntryLayer || cache.KPacked == nil {
		// Quantize and update cache
		hiddenDim := k.Shape().Dimensions[k.Rank()-1]
		mid := hiddenDim / 2
		
		k_x := Slice(k, AxisRange(), AxisRange(), AxisRange(0, mid))
		k_y := Slice(k, AxisRange(), AxisRange(), AxisRange(mid, hiddenDim))
		
		v_x := Slice(v, AxisRange(), AxisRange(), AxisRange(0, mid))
		v_y := Slice(v, AxisRange(), AxisRange(), AxisRange(mid, hiddenDim))
		
		k_packed := TurboQuantizeAdaptive(k_x, k_y, isReasoning, isAudio)
		v_packed := TurboQuantizeAdaptive(v_x, v_y, isReasoning, isAudio)
		
		// Update the persistent cache object
		cache.Update(k_packed, v_packed)
	}

	// 3. Dequantize
	k_x_recon, k_y_recon := TurboDequantizeAdaptive(cache.KPacked, isReasoning, isAudio)
	v_x_recon, v_y_recon := TurboDequantizeAdaptive(cache.VPacked, isReasoning, isAudio)
	
	k_prime := Concatenate([]*Node{k_x_recon, k_y_recon}, 2)
	v_prime := Concatenate([]*Node{v_x_recon, v_y_recon}, 2)

	// SWA Slicing
	if useSWA {
		cacheSeqLen := k_prime.Shape().Dimensions[1]
		if cacheSeqLen > windowSize {
			k_prime = Slice(k_prime, AxisRange(), AxisRange(cacheSeqLen-windowSize, cacheSeqLen), AxisRange())
			v_prime = Slice(v_prime, AxisRange(), AxisRange(cacheSeqLen-windowSize, cacheSeqLen), AxisRange())
		}
	}

	// Reshape K to [batch, seq, numHeads, headDim] before RoPE
	kBatchSize := k_prime.Shape().Dimensions[0]
	kSeqLen := k_prime.Shape().Dimensions[1]
	k_prime = Reshape(k_prime, kBatchSize, kSeqLen, numHeads, headDim)

	// Apply RoPE to K
	if useSWA {
		k_prime = ApplyRoPE(k_prime, 10000.0)
	} else {
		currentKSeqLen := float64(kSeqLen)
		k_prime = ApplyProportionalRoPE(k_prime, 10000.0, currentKSeqLen, maxWindow)
	}

	// Reshape K and V back to [batch, seq, hidden] for MultiHeadAttention
	k_prime = Reshape(k_prime, kBatchSize, kSeqLen, numHeads*headDim)
	v_prime = Reshape(v_prime, kBatchSize, kSeqLen, numHeads*headDim)

	// 4. Attention
	mha := layers.MultiHeadAttention(ctx, q, k_prime, v_prime, numHeads, numHeads*headDim)
	return mha.Done()
}
