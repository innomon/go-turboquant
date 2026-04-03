package main

import (
	"log"

	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/go-turboquant/turboquant"
	"github.com/gomlx/go-turboquant/internal/api"

	// Register XLA backend
	_ "github.com/gomlx/gomlx/backends/xla"
)

func main() {
	// 1. Initialize TurboQuant Backend
	backend, err := turboquant.InitializeBackend()
	if err != nil {
		log.Fatalf("❌ Failed to initialize GoMLX backend: %v", err)
	}

	// 2. Initialize ML Context
	ctx := context.New()

	// 3. Setup and Start API Server
	server := &api.Server{
		Backend: backend,
		Context: ctx,
		Port:    8080,
	}

	if err := server.Start(); err != nil {
		log.Fatalf("❌ Server failed: %v", err)
	}
}
