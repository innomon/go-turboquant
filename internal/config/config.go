package config

import (
	"os"
	"gopkg.in/yaml.v3"
	"github.com/gomlx/go-turboquant/turboquant"
)

// LoadGemma4Config reads a YAML file and returns a Gemma4Config.
func LoadGemma4Config(path string) (turboquant.Gemma4Config, error) {
	config := turboquant.DefaultGemma4E4BConfig() // Start with defaults

	f, err := os.Open(path)
	if err != nil {
		return config, err
	}
	defer f.Close()

	decoder := yaml.NewDecoder(f)
	err = decoder.Decode(&config)
	return config, err
}
