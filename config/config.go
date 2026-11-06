// Package config loads runtime configuration from environment
// variables, with a .env file in the working directory as a fallback
// (real environment variables win).
package config

import (
	"errors"
	"os"
	"strconv"
	"strings"

	"github.com/joho/godotenv"
)

type Config struct {
	// BaseURL, APIKey and Model address any OpenAI-compatible chat
	// endpoint (OpenAI, Ollama at :11434/v1, LM Studio, vLLM, ...).
	BaseURL string
	APIKey  string
	Model   string

	// SystemPromptFile is optional; a missing file means no system prompt.
	SystemPromptFile string

	// DatabaseURL is the Postgres DSN. Empty disables retrieval; chat
	// still works.
	DatabaseURL string

	// EmbeddingDim must match EmbeddingModel (nomic-embed-text: 768,
	// text-embedding-3-small: 1536). It is fixed into the column type on
	// first migration; changing it means recreating the table.
	EmbeddingDim int

	// EmbeddingBaseURL/APIKey let embeddings use a different endpoint
	// than chat. When the URL is empty both fall back to the chat ones.
	EmbeddingBaseURL string
	EmbeddingAPIKey  string
	EmbeddingModel   string

	// EmbeddingQueryPrefix and EmbeddingDocumentPrefix are prepended to
	// text before embedding (stored text is unchanged). nomic-embed-text
	// was trained with "search_query: " / "search_document: ", but on this
	// corpus they gave no significant gain (see README), so the default is
	// none. Re-test with cmd/eval -compare after retrieval changes.
	EmbeddingQueryPrefix    string
	EmbeddingDocumentPrefix string

	// MinScore is the cosine-similarity floor on a question's best chunk;
	// see rag.Options.MinScore. Tuned with cmd/eval.
	MinScore float64

	IngestDir    string
	ProcessedDir string
	HTTPAddr     string
	ImageDir     string
	VisionModel  string
}

// DefaultMinScore was chosen from the eval run recorded in
// eval/baseline.json (nomic-embed-text over corpus/).
const DefaultMinScore = 0.55

// Validate checks settings that would otherwise fail later at runtime.
// No API key is required: local servers accept a placeholder.
func (c Config) Validate() error {
	switch {
	case strings.TrimSpace(c.BaseURL) == "":
		return errors.New("OPENAI_BASE_URL must not be empty")
	case strings.TrimSpace(c.Model) == "":
		return errors.New("OPENAI_MODEL must not be empty")
	case c.EmbeddingDim <= 0:
		return errors.New("EMBEDDING_DIM must be greater than zero")
	case c.EmbeddingModel == "":
		return errors.New("EMBEDDING_MODEL must not be empty")
	case c.MinScore < 0 || c.MinScore >= 1:
		return errors.New("RETRIEVAL_MIN_SCORE must be in [0, 1)")
	case c.HTTPAddr == "":
		return errors.New("HTTP_ADDR must not be empty")
	case c.IngestDir == c.ProcessedDir:
		return errors.New("INGEST_DIR and PROCESSED_DIR must differ")
	}
	return nil
}

// Load reads the environment (after .env, if present) and applies
// defaults for unset values.
func Load() Config {
	_ = godotenv.Load() // a missing .env is fine

	cfg := Config{
		BaseURL:                 envOr("OPENAI_BASE_URL", "https://api.openai.com/v1"),
		APIKey:                  os.Getenv("OPENAI_API_KEY"),
		Model:                   envOr("OPENAI_MODEL", "gpt-4o-mini"),
		SystemPromptFile:        os.Getenv("SYSTEM_PROMPT_FILE"),
		DatabaseURL:             os.Getenv("DATABASE_URL"),
		EmbeddingDim:            atoiOr(os.Getenv("EMBEDDING_DIM"), 768),
		EmbeddingBaseURL:        os.Getenv("EMBEDDING_BASE_URL"),
		EmbeddingAPIKey:         os.Getenv("EMBEDDING_API_KEY"),
		EmbeddingModel:          envOr("EMBEDDING_MODEL", "nomic-embed-text"),
		EmbeddingQueryPrefix:    os.Getenv("EMBEDDING_QUERY_PREFIX"),
		EmbeddingDocumentPrefix: os.Getenv("EMBEDDING_DOCUMENT_PREFIX"),
		MinScore:                floatOr(os.Getenv("RETRIEVAL_MIN_SCORE"), DefaultMinScore),
		IngestDir:               envOr("INGEST_DIR", "./documents"),
		ProcessedDir:            envOr("PROCESSED_DIR", "./documents/processed"),
		HTTPAddr:                envOr("HTTP_ADDR", "127.0.0.1:8080"),
		ImageDir:                envOr("IMAGES_DIR", "./documents/images"),
		VisionModel:             os.Getenv("VISION_MODEL"),
	}

	// Reuse the chat endpoint for embeddings unless one is set. The chat
	// key is only borrowed in that case: sending it to a different host
	// would leak it.
	if cfg.EmbeddingBaseURL == "" {
		cfg.EmbeddingBaseURL = cfg.BaseURL
		if cfg.EmbeddingAPIKey == "" {
			cfg.EmbeddingAPIKey = cfg.APIKey
		}
	}
	return cfg
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// atoiOr parses s, returning fallback when s is empty or invalid.
func atoiOr(s string, fallback int) int {
	n, err := strconv.Atoi(s)
	if err != nil {
		return fallback
	}
	return n
}

func floatOr(s string, fallback float64) float64 {
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return fallback
	}
	return f
}
