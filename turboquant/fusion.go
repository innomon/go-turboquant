package turboquant

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// TurboQuantize unifies the quantization process into a single computational graph.
func TurboQuantize(x, y *Node) *Node {
	return TurboQuantizeAdaptive(x, y, false, false)
}

// TurboDequantize unifies the dequantization process into a single computational graph.
func TurboDequantize(packed *Node) (x, y *Node) {
	return TurboDequantizeAdaptive(packed, false, false)
}

// TurboQuantizeAdaptive supports switching bit-depth and codebooks.
func TurboQuantizeAdaptive(x, y *Node, isReasoning, isAudio bool) *Node {
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

	if isReasoning {
		// Use 2-bit QJL, 3-bit Radius, 3-bit Angle
		return Pack8BitReasoning(r_idx, theta_idx, sx_bit, sy_bit)
	}
	// Standard: 1-bit QJL, 4-bit Radius, 3-bit Angle
	return Pack8Bit(r_idx, theta_idx, sx_bit)
}

// TurboDequantizeAdaptive reconstructs based on reasoning and audio mode.
func TurboDequantizeAdaptive(packed *Node, isReasoning, isAudio bool) (x, y *Node) {
	g := packed.Graph()
	var r_idx, theta_idx, sx, sy *Node

	if isReasoning {
		r_idx, theta_idx, sx, sy = Unpack8BitReasoning(packed)
	} else {
		r_idx, theta_idx, sx = Unpack8Bit(packed)
		sy = Scalar(g, dtypes.Float32, 0.0)
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
