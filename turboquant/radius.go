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

// medicalRadiusLevels is optimized for MedGemma 1.5 latent manifold.
var medicalRadiusLevels = []float64{
	0.0, 0.05, 0.12, 0.22, 0.35, 0.52, 0.75, 1.05, 1.45, 2.0, 2.7, 3.6, 4.8, 6.4, 8.5, 12.0,
}

// QuantizeRadius maps radius r to one of 16 Lloyd-Max levels (4-bit).
func QuantizeRadius(r *Node) *Node {
	return QuantizeRadiusAdaptive(r, nil, nil)
}

// QuantizeRadiusAdaptive supports audio and medical optimized codebooks.
func QuantizeRadiusAdaptive(r *Node, isAudio, isMedical *Node) *Node {
	// Codebooks
	levelsStd := defaultRadiusLevels
	levelsAudio := audioRadiusLevels
	levelsMed := medicalRadiusLevels
	
	indicesStd := getQuantizedIndices(r, levelsStd)
	indicesAudio := getQuantizedIndices(r, levelsAudio)
	indicesMed := getQuantizedIndices(r, levelsMed)

	res := indicesStd
	if isAudio != nil {
		res = Where(isAudio, indicesAudio, res)
	}
	if isMedical != nil {
		res = Where(isMedical, indicesMed, res)
	}
	return res
}

func getQuantizedIndices(r *Node, levels []float64) *Node {
	g := r.Graph()
	thresholds := make([]float64, len(levels)-1)
	for i := 0; i < len(thresholds); i++ {
		thresholds[i] = (levels[i] + levels[i+1]) / 2.0
	}
	indices := Scalar(g, dtypes.Float32, 0.0)
	for _, t := range thresholds {
		indices = Add(indices, ConvertType(GreaterThan(r, Scalar(g, dtypes.Float32, t)), dtypes.Float32))
	}
	return indices
}

// DequantizeRadius reconstructs r from 4-bit indices.
func DequantizeRadius(indices *Node) *Node {
	return DequantizeRadiusAdaptive(indices, nil, nil)
}

// DequantizeRadiusAdaptive reconstructs r using optional audio/medical codebook.
func DequantizeRadiusAdaptive(indices *Node, isAudio, isMedical *Node) *Node {
	g := indices.Graph()
	
	codebookStd := Const(g, defaultRadiusLevels)
	codebookAudio := Const(g, audioRadiusLevels)
	codebookMed := Const(g, medicalRadiusLevels)
	
	intIndices := ConvertType(indices, dtypes.Int64)
	shape := intIndices.Shape()
	flatIndices := Reshape(intIndices, -1)
	expanded := ExpandDims(flatIndices, -1)
	
	resStd := Reshape(Gather(codebookStd, expanded), shape.Dimensions...)
	resAudio := Reshape(Gather(codebookAudio, expanded), shape.Dimensions...)
	resMed := Reshape(Gather(codebookMed, expanded), shape.Dimensions...)
	
	res := resStd
	if isAudio != nil {
		res = Where(isAudio, resAudio, res)
	}
	if isMedical != nil {
		res = Where(isMedical, resMed, res)
	}
	return ConvertType(res, dtypes.Float32)
}
