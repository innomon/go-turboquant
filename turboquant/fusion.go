package turboquant

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// TurboQuantize unifies the quantization process into a single computational graph.
func TurboQuantize(x, y *Node) *Node {
	return TurboQuantizeAdaptive(x, y, nil, nil)
}

// TurboDequantize unifies the dequantization process into a single computational graph.
func TurboDequantize(packed *Node) (x, y *Node) {
	return TurboDequantizeAdaptive(packed, nil, nil)
}

// TurboQuantizeAdaptive supports switching bit-depth and codebooks.
func TurboQuantizeAdaptive(x, y *Node, isReasoning, isAudio *Node) *Node {
	r, theta := CartesianToPolar(x, y)
	r_idx := QuantizeRadiusAdaptive(r, isAudio)
	theta_idx := QuantizeAngle(theta)

	// Dequantize temporarily for residual
	r_polar := DequantizeRadiusAdaptive(r_idx, isAudio)
	theta_polar := DequantizeAngle(theta_idx)
	x_polar, y_polar := PolarToCartesian(r_polar, theta_polar)

	sx, sy := QJLProjection(x, y, x_polar, y_polar)
	sx_bit := GreaterThan(sx, Scalar(x.Graph(), dtypes.Float32, 0.0))
	sy_bit := GreaterThan(sy, Scalar(x.Graph(), dtypes.Float32, 0.0))

	resStd := Pack8Bit(r_idx, theta_idx, sx_bit)
	resReasoning := Pack8BitReasoning(r_idx, theta_idx, sx_bit, sy_bit)

	if isReasoning == nil {
		return resStd
	}
	return Where(isReasoning, resReasoning, resStd)
}

// TurboDequantizeAdaptive reconstructs based on reasoning and audio mode.
func TurboDequantizeAdaptive(packed *Node, isReasoning, isAudio *Node) (x, y *Node) {
	g := packed.Graph()
	
	r_idx_std, theta_idx_std, sx_std := Unpack8Bit(packed)
	sy_std := Scalar(g, dtypes.Float32, 0.0)

	r_idx_reas, theta_idx_reas, sx_reas, sy_reas := Unpack8BitReasoning(packed)

	var r_idx, theta_idx, sx, sy *Node
	if isReasoning == nil {
		r_idx, theta_idx, sx, sy = r_idx_std, theta_idx_std, sx_std, sy_std
	} else {
		r_idx = Where(isReasoning, r_idx_reas, r_idx_std)
		theta_idx = Where(isReasoning, theta_idx_reas, theta_idx_std)
		sx = Where(isReasoning, sx_reas, sx_std)
		sy = Where(isReasoning, sy_reas, sy_std)
	}

	r_polar := DequantizeRadiusAdaptive(r_idx, isAudio)
	theta_polar := DequantizeAngle(theta_idx)
	x_polar, y_polar := PolarToCartesian(r_polar, theta_polar)

	// QJL Correction
	sx_val := Sub(Mul(sx, Scalar(g, dtypes.Float32, 2.0)), Scalar(g, dtypes.Float32, 1.0))
	sy_val := Sub(Mul(sy, Scalar(g, dtypes.Float32, 2.0)), Scalar(g, dtypes.Float32, 1.0))

	scale := 0.05
	x, y = ApplyQJLCorrection(x_polar, y_polar, sx_val, sy_val, scale)
	return
}
