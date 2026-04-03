package turboquant

import (
	"log"
	"os"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/simplego" // Registers SimpleGo
	_ "github.com/gomlx/gomlx/backends/xla"      // Registers XLA
)

// InitializeBackend initializes the GoMLX backend.
func InitializeBackend() (backends.Backend, error) {
	backendName := os.Getenv("GOMLX_BACKEND")
	if backendName == "" {
		backendName = "xla"
	}

	backend, err := backends.New()
	if err != nil {
		log.Printf("⚠️ Failed to initialize %s backend: %v. Falling back to simplego.", backendName, err)
		os.Setenv("GOMLX_BACKEND", "simplego")
		backend, err = backends.New()
		if err != nil {
			return nil, err
		}
	}

	log.Printf("🚀 GoMLX successfully running on: %s\n", backend.Name())
	return backend, nil
}
