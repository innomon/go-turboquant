package turboquant

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// TurboQuantize unifies the quantization process into a single computational graph.
// It takes Cartesian (x, y) coordinates and returns a single packed 8-bit tensor.
func TurboQuantize(x, y *Node) *Node {
	// 1. Cartesian to Polar
	r, theta := CartesianToPolar(x, y)
	
	// 2. Polar Quantization
	r_idx := QuantizeRadius(r)
	theta_idx := QuantizeAngle(theta)
	
	// 3. QJL Projection
	// For residual calculation, we dequantize temporarily
	r_polar := DequantizeRadius(r_idx)
	theta_polar := DequantizeAngle(theta_idx)
	x_polar, y_polar := PolarToCartesian(r_polar, theta_polar)
	
	// We only use 1-bit for x-component sign in this simplified 8-bit version
	sign_x, _ := QJLProjection(x, y, x_polar, y_polar)
	
	// Convert sign to 0/1 for packing
	qjl_sign := GreaterThan(sign_x, Scalar(x.Graph(), dtypes.Float32, 0.0))
	
	// 4. Bit-Packing
	return Pack8Bit(r_idx, theta_idx, qjl_sign)
}

// TurboDequantize unifies the dequantization process into a single computational graph.
// It takes a packed 8-bit tensor and returns reconstructed (x, y) Cartesian coordinates.
func TurboDequantize(packed *Node) (x, y *Node) {
	// 1. Unpacking
	r_idx, theta_idx, qjl_sign := Unpack8Bit(packed)
	
	// 2. Polar Reconstruction
	r_polar := DequantizeRadius(r_idx)
	theta_polar := DequantizeAngle(theta_idx)
	x_polar, y_polar := PolarToCartesian(r_polar, theta_polar)
	
	// 3. QJL Correction
	// map 0/1 back to -1/1 sign
	g := packed.Graph()
	sign := Sub(Mul(qjl_sign, Scalar(g, dtypes.Float32, 2.0)), Scalar(g, dtypes.Float32, 1.0))
	
	// Apply scale (approximate average residual)
	scale := 0.05 
	x, y = ApplyQJLCorrection(x_polar, y_polar, sign, Scalar(g, dtypes.Float32, 0.0), scale)
	
	return
}
