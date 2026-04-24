package turboquant

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// Pack8Bit packs (r_idx, theta_idx, qjl_sign) into a single 8-bit unsigned integer.
// r_idx (4 bits), theta_idx (3 bits), qjl_sign (1 bit).
// This version uses arithmetic to support backends without bitwise ops (e.g. SimpleGo).
func Pack8Bit(r_idx, theta_idx, qjl_sign *Node) *Node {
	g := r_idx.Graph()
	// Convert all inputs to Uint8 for packing
	r8 := ConvertType(r_idx, dtypes.Uint8)
	t8 := ConvertType(theta_idx, dtypes.Uint8)
	s8 := ConvertType(qjl_sign, dtypes.Uint8)

	// r (bits 4-7), theta (bits 1-3), sign (bit 0)
	// r_idx should be < 16, theta_idx < 8, qjl_sign < 2
	r_masked := Mod(r8, Scalar(g, dtypes.Uint8, 16))
	t_masked := Mod(t8, Scalar(g, dtypes.Uint8, 8))
	s_masked := Mod(s8, Scalar(g, dtypes.Uint8, 2))

	r_shifted := Mul(r_masked, Scalar(g, dtypes.Uint8, 16))
	t_shifted := Mul(t_masked, Scalar(g, dtypes.Uint8, 2))

	return Add(r_shifted, Add(t_shifted, s_masked))
}

// Unpack8Bit extracts (r_idx, theta_idx, qjl_sign) from a packed 8-bit integer.
func Unpack8Bit(packed *Node) (r_idx, theta_idx, qjl_sign *Node) {
	g := packed.Graph()
	// packed = r*16 + t*2 + s
	
	// r_idx = packed / 16
	r_u8 := Div(packed, Scalar(g, dtypes.Uint8, 16))
	
	// remain = packed % 16 = t*2 + s
	remain := Mod(packed, Scalar(g, dtypes.Uint8, 16))
	
	// theta_idx = remain / 2
	t_u8 := Div(remain, Scalar(g, dtypes.Uint8, 2))
	
	// qjl_sign = remain % 2
	s_u8 := Mod(remain, Scalar(g, dtypes.Uint8, 2))
	
	// Convert outputs back to float32
	r_idx = ConvertType(r_u8, dtypes.Float32)
	theta_idx = ConvertType(t_u8, dtypes.Float32)
	qjl_sign = ConvertType(s_u8, dtypes.Float32)
	return
}

// Pack8BitReasoning packs (r_idx, theta_idx, sign_x, sign_y) into 8 bits.
// r_idx (3 bits), theta_idx (3 bits), sign_x (1 bit), sign_y (1 bit).
func Pack8BitReasoning(r_idx, theta_idx, sx, sy *Node) *Node {
	r8 := ConvertType(r_idx, dtypes.Uint8)
	t8 := ConvertType(theta_idx, dtypes.Uint8)
	sx8 := ConvertType(sx, dtypes.Uint8)
	sy8 := ConvertType(sy, dtypes.Uint8)
	g := r8.Graph()

	// r (bits 5-7), theta (bits 2-4), sx (bit 1), sy (bit 0)
	r_masked := Mod(r8, Scalar(g, dtypes.Uint8, 8))
	t_masked := Mod(t8, Scalar(g, dtypes.Uint8, 8))
	sx_masked := Mod(sx8, Scalar(g, dtypes.Uint8, 2))
	sy_masked := Mod(sy8, Scalar(g, dtypes.Uint8, 2))

	r_shifted := Mul(r_masked, Scalar(g, dtypes.Uint8, 32))
	t_shifted := Mul(t_masked, Scalar(g, dtypes.Uint8, 4))
	sx_shifted := Mul(sx_masked, Scalar(g, dtypes.Uint8, 2))

	return Add(r_shifted, Add(t_shifted, Add(sx_shifted, sy_masked)))
}

// Unpack8BitReasoning extracts (r_idx, theta_idx, sx, sy) from a reasoning-packed 8-bit integer.
func Unpack8BitReasoning(packed *Node) (r_idx, theta_idx, sx, sy *Node) {
	g := packed.Graph()
	// packed = r*32 + t*4 + sx*2 + sy
	
	r_u8 := Div(packed, Scalar(g, dtypes.Uint8, 32))
	remain := Mod(packed, Scalar(g, dtypes.Uint8, 32))
	
	t_u8 := Div(remain, Scalar(g, dtypes.Uint8, 4))
	remain = Mod(remain, Scalar(g, dtypes.Uint8, 4))
	
	sx_u8 := Div(remain, Scalar(g, dtypes.Uint8, 2))
	sy_u8 := Mod(remain, Scalar(g, dtypes.Uint8, 2))

	r_idx = ConvertType(r_u8, dtypes.Float32)
	theta_idx = ConvertType(t_u8, dtypes.Float32)
	sx = ConvertType(sx_u8, dtypes.Float32)
	sy = ConvertType(sy_u8, dtypes.Float32)
	return
}
