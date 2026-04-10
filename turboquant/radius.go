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
	return QuantizeRadiusAdaptive(r, nil)
}

// QuantizeRadiusAdaptive supports audio-optimized codebooks.
func QuantizeRadiusAdaptive(r *Node, isAudio *Node) *Node {
	g := r.Graph()
	
	// Pre-calculate both options
	levelsStd := defaultRadiusLevels
	levelsAudio := audioRadiusLevels
	
	thresholdsStd := make([]float64, len(levelsStd)-1)
	for i := 0; i < len(thresholdsStd); i++ {
		thresholdsStd[i] = (levelsStd[i] + levelsStd[i+1]) / 2.0
	}

	thresholdsAudio := make([]float64, len(levelsAudio)-1)
	for i := 0; i < len(thresholdsAudio); i++ {
		thresholdsAudio[i] = (levelsAudio[i] + levelsAudio[i+1]) / 2.0
	}

	indicesStd := Scalar(g, dtypes.Float32, 0.0)
	for _, t := range thresholdsStd {
		indicesStd = Add(indicesStd, ConvertType(GreaterThan(r, Scalar(g, dtypes.Float32, t)), dtypes.Float32))
	}

	indicesAudio := Scalar(g, dtypes.Float32, 0.0)
	for _, t := range thresholdsAudio {
		indicesAudio = Add(indicesAudio, ConvertType(GreaterThan(r, Scalar(g, dtypes.Float32, t)), dtypes.Float32))
	}

	if isAudio == nil {
		return indicesStd
	}
	return Where(isAudio, indicesAudio, indicesStd)
}

// DequantizeRadius reconstructs r from 4-bit indices.
func DequantizeRadius(indices *Node) *Node {
	return DequantizeRadiusAdaptive(indices, nil)
}

// DequantizeRadiusAdaptive reconstructs r using optional audio codebook.
func DequantizeRadiusAdaptive(indices *Node, isAudio *Node) *Node {
	g := indices.Graph()
	
	codebookStd := Const(g, defaultRadiusLevels)
	codebookAudio := Const(g, audioRadiusLevels)
	
	intIndices := ConvertType(indices, dtypes.Int64)
	shape := intIndices.Shape()
	flatIndices := Reshape(intIndices, -1)
	expanded := ExpandDims(flatIndices, -1)
	
	resStd := Reshape(Gather(codebookStd, expanded), shape.Dimensions...)
	resAudio := Reshape(Gather(codebookAudio, expanded), shape.Dimensions...)
	
	if isAudio == nil {
		return ConvertType(resStd, dtypes.Float32)
	}
	return ConvertType(Where(isAudio, resAudio, resStd), dtypes.Float32)
}
