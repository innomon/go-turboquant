package turboquant

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// QJLProjection applies a 1-bit residual correction to the reconstructed (x_polar, y_polar).
// It returns a sign bit for x and y components.
func QJLProjection(x_orig, y_orig, x_polar, y_polar *Node) (sign_x, sign_y *Node) {
	// Error = Cartesian_original - Dequantized_polar
	err_x := Sub(x_orig, x_polar)
	err_y := Sub(y_orig, y_polar)

	// sign = Sign(Error)
	// We use 1-bit per component for simplicity, but in TurboQuant 
	// this is often a 1-bit projection through a rotation matrix Omega.
	sign_x = Sign(err_x)
	sign_y = Sign(err_y)
	
	return
}

// ApplyQJLCorrection applies the 1-bit sign correction to the reconstructed (x_polar, y_polar).
func ApplyQJLCorrection(x_polar, y_polar, sign_x, sign_y *Node, scale float64) (x_final, y_final *Node) {
	g := x_polar.Graph()
	
	// x_final = x_polar + scale * sign_x
	// scale is typically the standard deviation or mean of the quantization error.
	s := Scalar(g, dtypes.Float32, scale)
	x_final = Add(x_polar, Mul(s, sign_x))
	y_final = Add(y_polar, Mul(s, sign_y))
	
	return
}
