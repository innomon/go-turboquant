package turboquant

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"math"
)

// ApplyRoPE applies Rotary Positional Embeddings to a tensor.
// x shape: [batch, seq_len, num_heads, head_dim]
// headDim must be even.
func ApplyRoPE(x *Node, base float64) *Node {
	g := x.Graph()
	shape := x.Shape()
	seqLen := shape.Dimensions[1]
	headDim := shape.Dimensions[3]
	
	// 1. Generate frequencies
	// frequencies = base ^ (-2 * (0..headDim/2-1) / headDim)
	halfDim := headDim / 2
	indices := Iota(g, shapes.Make(dtypes.Int32, halfDim), 0)
	indicesF := ConvertType(indices, shape.DType)
	
	exponent := Mul(indicesF, Scalar(g, shape.DType, -2.0/float64(headDim)))
	freqs := Exp(Mul(Log(Scalar(g, shape.DType, base)), exponent))
	
	// 2. Generate positions [0..seqLen-1]
	t := Iota(g, shapes.Make(dtypes.Int32, seqLen), 0)
	tF := ConvertType(t, shape.DType)
	
	// 3. Compute phases [seqLen, headDim/2]
	// phases[t, i] = t * freqs[i]
	phases := Mul(Reshape(tF, seqLen, 1), Reshape(freqs, 1, halfDim))
	
	// 4. Compute cos and sin
	cos := Cos(phases)
	sin := Sin(phases)
	
	// Reshape to [1, seq_len, 1, head_dim/2] for broadcasting
	cos = Reshape(cos, 1, seqLen, 1, halfDim)
	sin = Reshape(sin, 1, seqLen, 1, halfDim)
	
	// 5. Apply RoPE
	// x_split: [batch, seq_len, num_heads, head_dim/2] x 2
	x1 := Slice(x, AxisRange(), AxisRange(), AxisRange(), AxisRange(0, halfDim))
	x2 := Slice(x, AxisRange(), AxisRange(), AxisRange(), AxisRange(halfDim, headDim))
	
	// x_rope1 = x1 * cos - x2 * sin
	// x_rope2 = x1 * sin + x2 * cos
	x_rope1 := Sub(Mul(x1, cos), Mul(x2, sin))
	x_rope2 := Add(Mul(x1, sin), Mul(x2, cos))
	
	return Concatenate([]*Node{x_rope1, x_rope2}, 3)
}

// ApplyProportionalRoPE applies RoPE with a frequency scaling factor
// proportional to the sequence length, as used in Gemma 4 Global layers.
func ApplyProportionalRoPE(x *Node, base float64, currentSeqLen, maxWindow float64) *Node {
	// Proportional scaling factor: scale = max(1, currentSeqLen / maxWindow)
	scale := math.Max(1.0, currentSeqLen/maxWindow)
	scaledBase := base * math.Pow(scale, float64(x.Shape().Dimensions[3])/(float64(x.Shape().Dimensions[3])-2.0))
	
	return ApplyRoPE(x, scaledBase)
}
