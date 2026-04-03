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
	// indices = (r_idx << 4) | (theta_idx << 1) | qjl_sign
	packed := BitwiseOr(BitwiseShiftLeftScalar(r8, 4),
		BitwiseOr(BitwiseShiftLeftScalar(t8, 1),
			s8))
	return packed
}

// Unpack8Bit extracts (r_idx, theta_idx, qjl_sign) from a packed 8-bit integer.
func Unpack8Bit(packed *Node) (r_idx, theta_idx, qjl_sign *Node) {
	// Radius (bits 4-7): mask 11110000 (0xF0) and shift right 4
	r_idx = BitwiseShiftRightLogicalScalar(BitwiseAnd(packed, Scalar(packed.Graph(), dtypes.Uint8, 0xF0)), 4)
	
	// Theta (bits 1-3): mask 00001110 (0x0E) and shift right 1
	theta_idx = BitwiseShiftRightLogicalScalar(BitwiseAnd(packed, Scalar(packed.Graph(), dtypes.Uint8, 0x0E)), 1)
	
	// QJL Sign (bit 0): mask 00000001 (0x01)
	qjl_sign = BitwiseAnd(packed, Scalar(packed.Graph(), dtypes.Uint8, 0x01))
	
	// Convert outputs back to float32/64 for further calculation
	r_idx = ConvertType(r_idx, dtypes.Float32)
	theta_idx = ConvertType(theta_idx, dtypes.Float32)
	qjl_sign = ConvertType(qjl_sign, dtypes.Float32)
	
	return
}
