package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

// Server handles OpenAI-compatible requests for the TurboQuant engine.
type Server struct {
	Backend backends.Backend
	Context *context.Context
	Port    int
}

// Model represents a simple OpenAI model object.
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ChatCompletionRequest follows the OpenAI chat completions schema.
type ChatCompletionRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionResponse follows the OpenAI chat completions response schema.
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Choices []Choice `json:"choices"`
}

type Choice struct {
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
	Index        int     `json:"index"`
}

// InitializeServer sets up the HTTP handlers.
func (s *Server) InitializeServer() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/models", s.handleListModels)
	mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
	return mux
}

func (s *Server) handleListModels(w http.ResponseWriter, r *http.Request) {
	models := []Model{
		{ID: "gemma-3-4b-turboquant", Object: "model", Created: time.Now().Unix(), OwnedBy: "turboquant"},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"data": models})
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// This is where TurboQuant inference would happen.
	// For this prototype, we simulate a response.
	resp := ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().Unix()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Choices: []Choice{
			{
				Index: 0,
				Message: Message{
					Role:    "assistant",
					Content: "[TurboQuant Simulation] Hello! I am running with 4-bit KV cache compression.",
				},
				FinishReason: "stop",
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// Start launches the HTTP server.
func (s *Server) Start() error {
	mux := s.InitializeServer()
	fmt.Printf("🚀 TurboQuant API server listening on :%d\n", s.Port)
	return http.ListenAndServe(fmt.Sprintf(":%d", s.Port), mux)
}
