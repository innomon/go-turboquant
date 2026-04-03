package turboquant

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// Pack8Bit packs (r_idx, theta_idx, qjl_sign) into a single 8-bit unsigned integer.
// r_idx (4 bits), theta_idx (3 bits), qjl_sign (1 bit).
func Pack8Bit(r_idx, theta_idx, qjl_sign *Node) *Node {
	// Convert all inputs to Uint8 for packing
	r8 := ConvertType(r_idx, dtypes.Uint8)
	t8 := ConvertType(theta_idx, dtypes.Uint8)
	s8 := ConvertType(qjl_sign, dtypes.Uint8)

	// Shift bits into position: r (bits 4-7), theta (bits 1-3), sign (bit 0)
	packed := BitwiseOr(BitwiseShiftLeftScalar(r8, 4),
		BitwiseOr(BitwiseShiftLeftScalar(t8, 1),
			s8))
	return packed
}

// Unpack8Bit extracts (r_idx, theta_idx, qjl_sign) from a packed 8-bit integer.
func Unpack8Bit(packed *Node) (r_idx, theta_idx, qjl_sign *Node) {
	g := packed.Graph()
	// Radius (bits 4-7): mask 11110000 (0xF0) and shift right 4
	r_idx = BitwiseShiftRightLogicalScalar(BitwiseAnd(packed, Scalar(g, dtypes.Uint8, 0xF0)), 4)
	
	// Theta (bits 1-3): mask 00001110 (0x0E) and shift right 1
	theta_idx = BitwiseShiftRightLogicalScalar(BitwiseAnd(packed, Scalar(g, dtypes.Uint8, 0x0E)), 1)
	
	// QJL Sign (bit 0): mask 00000001 (0x01)
	qjl_sign = BitwiseAnd(packed, Scalar(g, dtypes.Uint8, 0x01))
	
	// Convert outputs back to float32
	r_idx = ConvertType(r_idx, dtypes.Float32)
	theta_idx = ConvertType(theta_idx, dtypes.Float32)
	qjl_sign = ConvertType(qjl_sign, dtypes.Float32)
	return
}

// Pack8BitReasoning packs (r_idx, theta_idx, sign_x, sign_y) into 8 bits.
// r_idx (3 bits), theta_idx (3 bits), sign_x (1 bit), sign_y (1 bit).
func Pack8BitReasoning(r_idx, theta_idx, sx, sy *Node) *Node {
	r8 := ConvertType(r_idx, dtypes.Uint8)
	t8 := ConvertType(theta_idx, dtypes.Uint8)
	sx8 := ConvertType(sx, dtypes.Uint8)
	sy8 := ConvertType(sy, dtypes.Uint8)

	// r (bits 5-7), theta (bits 2-4), sx (bit 1), sy (bit 0)
	packed := BitwiseOr(BitwiseShiftLeftScalar(BitwiseAnd(r8, Scalar(r8.Graph(), dtypes.Uint8, 0x07)), 5),
		BitwiseOr(BitwiseShiftLeftScalar(BitwiseAnd(t8, Scalar(t8.Graph(), dtypes.Uint8, 0x07)), 2),
			BitwiseOr(BitwiseShiftLeftScalar(sx8, 1), sy8)))
	return packed
}

// Unpack8BitReasoning extracts (r_idx, theta_idx, sx, sy) from a reasoning-packed 8-bit integer.
func Unpack8BitReasoning(packed *Node) (r_idx, theta_idx, sx, sy *Node) {
	g := packed.Graph()
	r_idx = BitwiseShiftRightLogicalScalar(BitwiseAnd(packed, Scalar(g, dtypes.Uint8, 0xE0)), 5)
	theta_idx = BitwiseShiftRightLogicalScalar(BitwiseAnd(packed, Scalar(g, dtypes.Uint8, 0x1C)), 2)
	sx = BitwiseShiftRightLogicalScalar(BitwiseAnd(packed, Scalar(g, dtypes.Uint8, 0x02)), 1)
	sy = BitwiseAnd(packed, Scalar(g, dtypes.Uint8, 0x01))

	r_idx = ConvertType(r_idx, dtypes.Float32)
	theta_idx = ConvertType(theta_idx, dtypes.Float32)
	sx = ConvertType(sx, dtypes.Float32)
	sy = ConvertType(sy, dtypes.Float32)
	return
}
