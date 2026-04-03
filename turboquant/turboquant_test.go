package turboquant

import (
	"fmt"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/graph"
)

func TestTurboQuantRoundTrip(t *testing.T) {
	backend, err := InitializeBackend()
	if err != nil {
		t.Fatalf("Failed to initialize backend: %v", err)
	}

	// We'll use a simpler manual test because RunTestGraphFn might be too restrictive for this complex logic.
	g := graph.NewGraph(backend, "TurboQuant RoundTrip")
	
	// Create dummy Cartesian data (x, y pairs)
	x_orig := []float32{1.0, -2.0, 0.5, 3.0}
	y_orig := []float32{1.0, 2.0, -0.5, -3.0}
	x := graph.Const(g, x_orig)
	y := graph.Const(g, y_orig)
	
	// Quantize
	packed := TurboQuantize(x, y)
	
	// Dequantize
	x_recon_node, y_recon_node := TurboDequantize(packed)
	
	g.Compile(x_recon_node, y_recon_node)
	outputs := g.Run()
	
	x_recon := outputs[0].Value().([]float32)
	y_recon := outputs[1].Value().([]float32)

	for i := 0; i < len(x_orig); i++ {
		fmt.Printf("Pair %d: Original(%.3f, %.3f) -> Reconstructed(%.3f, %.3f)\n", 
			i, x_orig[i], y_orig[i], x_recon[i], y_recon[i])
	}
}

func TestPacking(t *testing.T) {
	backend, err := InitializeBackend()
	if err != nil {
		t.Fatalf("Failed to initialize backend: %v", err)
	}

	g := graph.NewGraph(backend, "Bit Packing")
	
	r_in := []float32{15.0, 0.0, 7.0}
	t_in := []float32{7.0, 0.0, 3.0}
	s_in := []float32{1.0, 0.0, 1.0}
	
	r_idx := graph.Const(g, r_in)
	theta_idx := graph.Const(g, t_in)
	qjl_sign := graph.Const(g, s_in)
	
	packed := Pack8Bit(r_idx, theta_idx, qjl_sign)
	r_out_node, t_out_node, s_out_node := Unpack8Bit(packed)
	
	g.Compile(r_out_node, t_out_node, s_out_node)
	outputs := g.Run()
	
	r_out := outputs[0].Value().([]float32)
	t_out := outputs[1].Value().([]float32)
	s_out := outputs[2].Value().([]float32)

	for i := 0; i < len(r_in); i++ {
		if r_in[i] != r_out[i] || t_in[i] != t_out[i] || s_in[i] != s_out[i] {
			t.Errorf("Mismatch at index %d: expected (%.1f, %.1f, %.1f), got (%.1f, %.1f, %.1f)", 
				i, r_in[i], t_in[i], s_in[i], r_out[i], t_out[i], s_out[i])
		}
	}
}
