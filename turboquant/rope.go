package turboquant

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// ApplyRoPEWithOffset applies Rotary Positional Embeddings with a starting sequence offset.
func ApplyRoPEWithOffset(x *Node, offset *Node, base float64) *Node {
	g := x.Graph()
	shape := x.Shape()
	seqLen := shape.Dimensions[1]
	headDim := shape.Dimensions[3]
	
	// 1. Generate frequencies
	halfDim := headDim / 2
	indices := Iota(g, shapes.Make(dtypes.Int32, halfDim), 0)
	indicesF := ConvertType(indices, shape.DType)
	exponent := Mul(indicesF, Scalar(g, shape.DType, -2.0/float64(headDim)))
	freqs := Exp(Mul(Log(Scalar(g, shape.DType, base)), exponent))
	
	// 2. Generate positions [offset..offset+seqLen-1]
	t := Iota(g, shapes.Make(dtypes.Int32, seqLen), 0)
	t = Add(t, ConvertType(offset, dtypes.Int32))
	tF := ConvertType(t, shape.DType)
	
	// 3. Compute phases [seqLen, headDim/2]
	phases := Mul(Reshape(tF, seqLen, 1), Reshape(freqs, 1, halfDim))
	cos := Cos(phases)
	sin := Sin(phases)
	cos = Reshape(cos, 1, seqLen, 1, halfDim)
	sin = Reshape(sin, 1, seqLen, 1, halfDim)
	
	// 4. Apply RoPE
	x1 := Slice(x, AxisRange(), AxisRange(), AxisRange(), AxisRange(0, halfDim))
	x2 := Slice(x, AxisRange(), AxisRange(), AxisRange(), AxisRange(halfDim, headDim))
	x_rope1 := Sub(Mul(x1, cos), Mul(x2, sin))
	x_rope2 := Add(Mul(x1, sin), Mul(x2, cos))
	return Concatenate([]*Node{x_rope1, x_rope2}, 3)
}

// ApplyProportionalRoPE applies RoPE with a frequency scaling factor
// proportional to the sequence length, as used in Gemma 4 Global layers.
func ApplyProportionalRoPE(x *Node, offset *Node, base float64, currentSeqLen *Node, maxWindow float64) *Node {
	g := x.Graph()
	dtype := x.Shape().DType
	
	// Convert currentSeqLen to match dtype for division
	currentSeqLenF := ConvertType(currentSeqLen, dtype)
	
	// Proportional scaling factor: scale = max(1, currentSeqLen / maxWindow)
	maxWindowNode := Scalar(g, dtype, maxWindow)
	scale := Max(Scalar(g, dtype, 1.0), Div(currentSeqLenF, maxWindowNode))
	
	// For Proportional RoPE, we scale the base frequency:
	// scaled_base = base * scale^(head_dim / (head_dim - 2))
	headDim := float64(x.Shape().Dimensions[3])
	powExp := headDim / (headDim - 2.0)
	
	// scaledBase = base * Pow(scale, powExp)
	scaledBase := Mul(Scalar(g, dtype, base), Pow(scale, Scalar(g, dtype, powExp)))
	
	// We can't pass a Node as the 'base' float64 to ApplyRoPEWithOffset directly 
	// without refactoring it to accept a Node-based base.
	return ApplyRoPEWithNodeBase(x, offset, scaledBase)
}

// ApplyRoPEWithNodeBase is a helper that accepts a Node for the base frequency.
func ApplyRoPEWithNodeBase(x *Node, offset *Node, base *Node) *Node {
	g := x.Graph()
	shape := x.Shape()
	seqLen := shape.Dimensions[1]
	headDim := shape.Dimensions[3]
	
	halfDim := headDim / 2
	indices := Iota(g, shapes.Make(dtypes.Int32, halfDim), 0)
	indicesF := ConvertType(indices, shape.DType)
	exponent := Mul(indicesF, Scalar(g, shape.DType, -2.0/float64(headDim)))
	freqs := Exp(Mul(Log(base), exponent))
	
	t := Iota(g, shapes.Make(dtypes.Int32, seqLen), 0)
	t = Add(t, ConvertType(offset, dtypes.Int32))
	tF := ConvertType(t, shape.DType)
	
	phases := Mul(Reshape(tF, seqLen, 1), Reshape(freqs, 1, halfDim))
	cos := Cos(phases)
	sin := Sin(phases)
	cos = Reshape(cos, 1, seqLen, 1, halfDim)
	sin = Reshape(sin, 1, seqLen, 1, halfDim)
	
	x1 := Slice(x, AxisRange(), AxisRange(), AxisRange(), AxisRange(0, halfDim))
	x2 := Slice(x, AxisRange(), AxisRange(), AxisRange(), AxisRange(halfDim, headDim))
	x_rope1 := Sub(Mul(x1, cos), Mul(x2, sin))
	x_rope2 := Add(Mul(x1, sin), Mul(x2, cos))
	return Concatenate([]*Node{x_rope1, x_rope2}, 3)
}
