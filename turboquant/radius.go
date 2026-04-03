package turboquant

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// RadiusCodebook represents a 4-bit (16 levels) Lloyd-Max codebook for radius quantization.
// In a production environment, these levels would be pre-calculated based on a data distribution.
var defaultRadiusLevels = []float64{
	0.0, 0.1, 0.25, 0.4, 0.6, 0.85, 1.15, 1.5, 1.9, 2.4, 3.0, 3.7, 4.6, 5.7, 7.1, 9.0,
}

// audioRadiusLevels is optimized for USM-conformer outputs (higher variance).
var audioRadiusLevels = []float64{
	0.0, 0.2, 0.5, 0.9, 1.4, 2.0, 2.8, 3.8, 5.0, 6.5, 8.5, 11.0, 14.5, 19.0, 25.0, 35.0,
}

// QuantizeRadius maps radius r to one of 16 Lloyd-Max levels (4-bit).
func QuantizeRadius(r *Node) *Node {
	return QuantizeRadiusAdaptive(r, false)
}

// QuantizeRadiusAdaptive supports audio-optimized codebooks.
func QuantizeRadiusAdaptive(r *Node, isAudio bool) *Node {
	g := r.Graph()
	levels := defaultRadiusLevels
	if isAudio {
		levels = audioRadiusLevels
	}
	
	thresholds := make([]float64, len(levels)-1)
	for i := 0; i < len(thresholds); i++ {
		thresholds[i] = (levels[i] + levels[i+1]) / 2.0
	}

	indices := Scalar(g, dtypes.Float32, 0.0)
	for _, t := range thresholds {
		condition := GreaterThan(r, Scalar(g, dtypes.Float32, t))
		indices = Add(indices, ConvertType(condition, dtypes.Float32))
	}
	return indices
}

// DequantizeRadius reconstructs r from 4-bit indices.
func DequantizeRadius(indices *Node) *Node {
	return DequantizeRadiusAdaptive(indices, false)
}

// DequantizeRadiusAdaptive reconstructs r using optional audio codebook.
func DequantizeRadiusAdaptive(indices *Node, isAudio bool) *Node {
	g := indices.Graph()
	levels := defaultRadiusLevels
	if isAudio {
		levels = audioRadiusLevels
	}
	codebook := Const(g, levels)
	
	intIndices := ConvertType(indices, dtypes.Int64)
	expandedIndices := ExpandDims(intIndices, -1)
	r := Gather(codebook, expandedIndices)
	return ConvertType(r, dtypes.Float32)
}
