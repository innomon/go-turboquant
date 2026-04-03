package turboquant

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// Atan2Node approximates atan2(y, x) using available ops.
// Uses a common rational approximation for atan(z).
func Atan2Node(y, x *Node) *Node {
	g := y.Graph()
	// eps to avoid div by zero
	eps := Scalar(g, dtypes.Float32, 1e-7)
	absX := Abs(x)
	absY := Abs(y)
	
	// z = y / x (but only for x != 0)
	// We handle the case where x is small or zero by checking |x| vs |y|
	
	// Quadrant adjustments:
	// pi = 3.14159265
	pi := Scalar(g, dtypes.Float32, 3.14159265)
	halfPi := Scalar(g, dtypes.Float32, 1.57079632)
	
	// if absX >= absY:
	//   atan(y/x)
	// else:
	//   sign(y) * pi/2 - atan(x/y)
	
	cond := GreaterOrEqual(absX, absY)
	
	// atan_approx(z) = z / (1 + 0.280872 * z^2)
	atan_approx := func(z *Node) *Node {
		coeff := Scalar(g, dtypes.Float32, 0.280872)
		return Div(z, AddScalar(Mul(coeff, Square(z)), 1.0))
	}
	
	z1 := Div(y, Add(x, eps))
	a1 := atan_approx(z1)
	
	z2 := Div(x, Add(y, eps))
	a2 := Sub(Mul(Sign(y), halfPi), atan_approx(z2))
	
	theta := Where(cond, a1, a2)
	
	// Handle negative X quadrants (atan only covers -pi/2 to pi/2)
	// if x < 0:
	//   if y >= 0: theta += pi
	//   else: theta -= pi
	
	isNegX := LessThan(x, Scalar(g, dtypes.Float32, 0.0))
	isNegY := LessThan(y, Scalar(g, dtypes.Float32, 0.0))
	
	adj := Where(isNegY, Neg(pi), pi)
	theta = Where(LogicalAnd(isNegX, cond), Add(theta, adj), theta)
	
	return theta
}

// CartesianToPolar transforms (x, y) coordinates into (r, theta).
// It returns r (radius) and theta (angle).
func CartesianToPolar(x, y *Node) (r, theta *Node) {
	// r = sqrt(x^2 + y^2)
	r = Sqrt(Add(Square(x), Square(y)))
	// theta = atan2(y, x)
	theta = Atan2Node(y, x)
	return
}

// PolarToCartesian transforms (r, theta) coordinates back into (x, y).
func PolarToCartesian(r, theta *Node) (x, y *Node) {
	// x = r * cos(theta)
	x = Mul(r, Cos(theta))
	// y = r * sin(theta)
	y = Mul(r, Sin(theta))
	return
}

// QuantizeAngle maps theta to one of 8 circular grid indices (3-bit).
func QuantizeAngle(theta *Node) *Node {
	// theta is in [-pi, pi]
	// map to [0, 8)
	// indices = floor((theta + pi) / (2 * pi / 8))
	pi := Scalar(theta.Graph(), dtypes.Float32, 3.14159265358979323846)
	twoPi := Scalar(theta.Graph(), dtypes.Float32, 2*3.14159265358979323846)
	sectors := Scalar(theta.Graph(), dtypes.Float32, 8.0)

	normalized := Div(Add(theta, pi), twoPi)
	indices := Floor(Mul(normalized, sectors))
	
	// Ensure indices are in [0, 7]
	indices = Mod(indices, sectors)
	return indices
}

// DequantizeAngle reconstructs theta from 3-bit circular grid indices.
func DequantizeAngle(indices *Node) *Node {
	pi := Scalar(indices.Graph(), dtypes.Float32, 3.14159265358979323846)
	twoPi := Scalar(indices.Graph(), dtypes.Float32, 2*3.14159265358979323846)
	sectors := Scalar(indices.Graph(), dtypes.Float32, 8.0)
	
	// Add 0.5 to center the angle in the sector
	centerOffset := Scalar(indices.Graph(), dtypes.Float32, 0.5)
	normalized := Div(Add(indices, centerOffset), sectors)
	theta := Sub(Mul(normalized, twoPi), pi)
	return theta
}
