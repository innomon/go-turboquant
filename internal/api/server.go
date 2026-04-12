package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unsafe"

	"github.com/gomlx/go-turboquant/internal/config"
	"github.com/gomlx/go-turboquant/turboquant"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/nlpodyssey/safetensors"
)

// Server handles OpenAI-compatible requests for the TurboQuant engine.
type Server struct {
	Backend       backends.Backend
	Context       *context.Context
	Port          int
	WeightsDir    string
	MTPCheckpoint string
	Config        turboquant.Gemma4Config
	exec          *context.Exec
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
		{ID: "gemma-4-4b-turboquant", Object: "model", Created: time.Now().Unix(), OwnedBy: "turboquant"},
	}
	
	if s.WeightsDir != "" {
		if files, _ := filepath.Glob(filepath.Join(s.WeightsDir, "*.safetensors")); len(files) > 0 {
			// Weights detected.
		}
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

	if len(req.Messages) == 0 {
		http.Error(w, "no messages provided", http.StatusBadRequest)
		return
	}

	// 1. Tokenize input
	inputContent := req.Messages[len(req.Messages)-1].Content
	inputTokens := s.tokenize(inputContent)

	// 2. Initialize Execution if needed
	if s.exec == nil {
		var err error
		s.exec, err = context.NewExec(s.Backend, s.Context, func(ctx *context.Context, tokens *Node) []*Node {
			if s.Config.ModelType == "gemma3" {
				return turboquant.BuildGemma3Model(ctx, tokens, s.Config)
			}
			// Default to Gemma 4
			g := tokens.Graph()
			// Create a zero PLE tensor for initial call
			pleShape := shapes.Make(dtypes.Float32, 1, tokens.Shape().Dimensions[1], s.Config.PLEDim)
			ple := Const(g, tensors.FromScalarAndDimensions(float32(0), pleShape.Dimensions...))
			return turboquant.BuildGemma4Model(ctx, tokens, ple, s.Config)
		})
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to build model: %v", err), http.StatusInternalServerError)
			return
		}
	}

	// 3. Inference Loop (with MTP)
	var generatedTokens []int32
	maxNewTokens := 50
	
	// Prepare initial tensor
	currentTokens := tensors.FromFlatDataAndDimensions(inputTokens, 1, len(inputTokens))
	
	for len(generatedTokens) < maxNewTokens {
		// Run inference
		results, err := s.exec.Exec(currentTokens)
		if err != nil {
			http.Error(w, fmt.Sprintf("inference error: %v", err), http.StatusInternalServerError)
			return
		}

		baseLogits := results[0]
		mtpLogits := results[1:]

		// Verify and Accept (MTP Speculative Decoding)
		specResult := turboquant.VerifyMTP(s.Context, baseLogits, mtpLogits)
		
		// Convert AcceptedTokens to int32 for our loop
		accepted32 := make([]int32, len(specResult.AcceptedTokens))
		for i, t := range specResult.AcceptedTokens {
			accepted32[i] = int32(t)
		}
		generatedTokens = append(generatedTokens, accepted32...)

		// Stop if we see an EOS token (simulated as 1)
		if contains32(accepted32, 1) {
			break
		}

		// Update input for next step
		currentTokens = tensors.FromFlatDataAndDimensions(accepted32, 1, len(accepted32))
	}

	// 4. De-tokenize
	responseContent := s.detokenize(generatedTokens)

	resp := ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().Unix()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Choices: []Choice{
			{
				Index: 0,
				Message: Message{
					Role:    "assistant",
					Content: responseContent,
				},
				FinishReason: "stop",
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// tokenize is a simple placeholder for a real tokenizer.
func (s *Server) tokenize(text string) []int32 {
	tokens := []int32{}
	for _, r := range text {
		tokens = append(tokens, int32(r)%int32(s.Config.VocabSize))
	}
	if len(tokens) == 0 {
		tokens = append(tokens, 0)
	}
	return tokens
}

// detokenize is a simple placeholder for a real detokenizer.
func (s *Server) detokenize(tokens []int32) string {
	var sb strings.Builder
	for _, t := range tokens {
		if t > 32 && t < 127 {
			sb.WriteRune(rune(t))
		} else {
			sb.WriteString(" ")
		}
	}
	return sb.String()
}

func contains32(slice []int32, val int32) bool {
	for _, s := range slice {
		if s == val {
			return true
		}
	}
	return false
}

// Start launches the HTTP server.
func (s *Server) Start() error {
	// Load config
	cfg, err := config.LoadGemma4Config("config.yaml")
	if err != nil {
		fmt.Printf("⚠️ Failed to load config.yaml, using defaults: %v\n", err)
		s.Config = turboquant.DefaultGemma4E4BConfig()
	} else {
		s.Config = cfg
	}

	if s.WeightsDir != "" {
		fmt.Printf("📦 Loading weights from %s...\n", s.WeightsDir)
		if err := s.LoadWeights(); err != nil {
			return fmt.Errorf("failed to load weights: %w", err)
		}
	}
	
	if s.MTPCheckpoint != "" {
		fmt.Printf("📂 Loading MTP checkpoints from %s...\n", s.MTPCheckpoint)
		if err := turboquant.LoadMTPCheckpoints(s.Context, s.MTPCheckpoint); err != nil {
			fmt.Printf("⚠️ Failed to load MTP checkpoints: %v\n", err)
		}
	}

	// Initialize variables in the context (needed for GoMLX backends)
	if err := s.Context.InitializeVariables(s.Backend, nil); err != nil {
		return fmt.Errorf("failed to initialize variables: %w", err)
	}

	mux := s.InitializeServer()
	fmt.Printf("🚀 TurboQuant API server listening on :%d\n", s.Port)
	return http.ListenAndServe(fmt.Sprintf(":%d", s.Port), mux)
}

// LoadWeights loads safetensors from the WeightsDir.
func (s *Server) LoadWeights() error {
	files, err := filepath.Glob(filepath.Join(s.WeightsDir, "*.safetensors"))
	if err != nil {
		return err
	}
	for _, file := range files {
		data, err := os.ReadFile(file)
		if err != nil {
			return err
		}
		st, err := safetensors.Deserialize(data)
		if err != nil {
			return err
		}
		for _, name := range st.Names() {
			t, _ := st.Tensor(name)
			gomlxScope := mapHuggingFaceToGoMLX(name)

			dims := make([]int, len(t.Shape()))
			for i, dim := range t.Shape() {
				dims[i] = int(dim)
			}

			if t.DType() == safetensors.F32 {
				tData := t.Data()
				floatData := *(*[]float32)(unsafe.Pointer(&tData))
				floatData = floatData[:len(tData)/4]
				gmlxt := tensors.FromFlatDataAndDimensions(floatData, dims...)
				s.Context.In(gomlxScope).VariableWithShape("weight", shapes.Make(gmlxt.DType(), dims...)).MustSetValue(gmlxt)
			}
		}
	}
	return nil
}

func mapHuggingFaceToGoMLX(hfName string) string {
	name := strings.ReplaceAll(hfName, ".", "/")
	if !strings.HasPrefix(name, "/") {
		name = "/" + name
	}
	return name
}
