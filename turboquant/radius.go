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

// QuantizeRadius maps radius r to one of 16 Lloyd-Max levels (4-bit).
// It returns the indices in [0, 15].
func QuantizeRadius(r *Node) *Node {
	g := r.Graph()
	
	// thresholds are midpoints between levels
	thresholds := make([]float64, len(defaultRadiusLevels)-1)
	for i := 0; i < len(thresholds); i++ {
		thresholds[i] = (defaultRadiusLevels[i] + defaultRadiusLevels[i+1]) / 2.0
	}

	indices := Scalar(g, dtypes.Float32, 0.0)
	for _, t := range thresholds {
		// increment index if r > threshold
		condition := GreaterThan(r, Scalar(g, dtypes.Float32, t))
		indices = Add(indices, ConvertType(condition, dtypes.Float32))
	}
	
	return indices
}

// DequantizeRadius reconstructs r from 4-bit indices using the Lloyd-Max codebook.
func DequantizeRadius(indices *Node) *Node {
	g := indices.Graph()
	
	// Create a constant node for the codebook
	codebook := Const(g, defaultRadiusLevels)
	
	// Gather from the codebook
	// For Gather, indices must have shape [..., N] where N is the rank of params being indexed.
	// Since codebook is rank 1, indices must have shape [..., 1].
	intIndices := ConvertType(indices, dtypes.Int64)
	expandedIndices := ExpandDims(intIndices, -1)
	r := Gather(codebook, expandedIndices)
	return ConvertType(r, dtypes.Float32)
}
