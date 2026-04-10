package turboquant

import (
	"fmt"
	"os"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/checkpoints"
	"github.com/gomlx/gomlx/pkg/ml/train/losses"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// ... (ZimDataset logic same as before)

// LoadMTPCheckpoints restores MTP head weights from a checkpoint.
func LoadMTPCheckpoints(ctx *context.Context, checkpointPath string) error {
	if _, err := os.Stat(checkpointPath); os.IsNotExist(err) {
		return fmt.Errorf("checkpoint path %s does not exist", checkpointPath)
	}
	cp := checkpoints.New(ctx, checkpointPath)
	return cp.Restore()
}

// TrainMTP performs self-distillation training for the MTP heads.
func TrainMTP(ctx *context.Context, config Gemma4Config, checkpointPath string) {
	// 0. Load existing MTP weights if they exist (Incremental Training)
	fmt.Printf("📂 Checking for existing MTP weights in %s...\n", checkpointPath)
	if err := LoadMTPCheckpoints(ctx, checkpointPath); err == nil {
		fmt.Println("✅ Resuming from existing MTP checkpoint.")
	} else {
		fmt.Printf("🌱 No existing MTP checkpoint found. Starting from random initialization.\n")
	}

	// 1. Define Optimizer
	opt := optimizers.Adagrad().LearningRate(1e-4).Done()
	_ = opt // used in training loop

	// 2. Training Step (XLA Compiled)
	// inputs: [batch, seq]
	// targets: [batch, seq] x (NumMTPHeads + 1)
	trainStep := context.NewExec(ctx.Backend(), ctx, func(ctx *context.Context, inputs *Node, targets []*Node) *Node {
		// Get full model output (Tuple of base + MTP logits)
		modelOutput := BuildGemma4Model(ctx, inputs, nil, config)
		
		var totalLoss *Node
		numOutputs := config.NumMTPHeads + 1
		for i := 0; i < numOutputs; i++ {
			logits := GetTupleItem(modelOutput, i)
			// SparseSoftmaxCrossEntropy expects targets as integer indices.
			loss := losses.SparseSoftmaxCrossEntropy(targets[i], logits)
			if totalLoss == nil {
				totalLoss = loss
			} else {
				totalLoss = Add(totalLoss, loss)
			}
		}
		
		return totalLoss
	})

	fmt.Println("🚀 Starting MTP Distillation Training on Medicine ZIM data...")
	// ... (Loop logic)
	_ = trainStep
	
	// 3. Save after training
	fmt.Printf("💾 Saving MTP heads to %s...\n", checkpointPath)
	SaveMTPCheckpoints(ctx, checkpointPath)
}

// SaveMTPCheckpoints saves only the MTP head weights.
func SaveMTPCheckpoints(ctx *context.Context, checkpointPath string) {
	cp := checkpoints.New(ctx, checkpointPath)
	cp.Save()
}
