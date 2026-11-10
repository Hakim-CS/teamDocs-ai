// Package rag is the query side: rewrite a follow-up into a standalone
// query, embed it, fetch the nearest chunks, refuse the question when
// even the best one is weak, and format the hits as prompt context.
package rag

import (
	"context"
	"fmt"
	"godocs-rag/llm"
	"godocs-rag/vector"
	"slices"
)

const defaultTopK = 5

type Options struct {
	TopK int
	// MinScore is the out-of-scope gate: when no hit's cosine similarity
	// reaches it, Search returns nothing, the model gets no excerpts and
	// the system prompt tells it to say the docs don't cover the question.
	// Otherwise every hit is returned. The value is chosen from the eval
	// (see cmd/eval); 0 disables it.
	MinScore float64
	Rewriter *Rewriter
}

// SearchResult is one retrieved chunk, independent of prompt format.
type SearchResult struct {
	ChunkID   string
	Source    string
	Content   string
	ImagePath string
	Score     float64
}

type Retriever struct {
	embedder llm.Embedder
	store    vector.Store
	rewriter *Rewriter
	topK     int
	minScore float64
}

func New(embedder llm.Embedder, store vector.Store, opts Options) *Retriever {
	topK := opts.TopK
	if topK <= 0 {
		topK = defaultTopK
	}
	return &Retriever{
		embedder: embedder,
		store:    store,
		rewriter: opts.Rewriter,
		topK:     topK,
		minScore: opts.MinScore,
	}
}

// Retrieve returns formatted context for the latest user turn, or ""
// when nothing relevant was found.
func (r *Retriever) Retrieve(ctx context.Context, history []llm.Message) (string, error) {
	hits, err := r.Search(ctx, history)
	if err != nil {
		return "", err
	}
	return formatContext(hits), nil
}

// Search returns the chunks relevant to the latest user turn, best
// first, or nothing when even the best one is below MinScore.
func (r *Retriever) Search(ctx context.Context, history []llm.Message) ([]SearchResult, error) {
	query := r.buildQuery(ctx, history)
	if query == "" {
		return nil, nil
	}

	vec, err := r.embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}

	hits, err := r.store.Query(ctx, vec, r.topK)
	if err != nil {
		return nil, fmt.Errorf("vector query: %w", err)
	}

	// The floor gates the question, not each hit: weak hits that sit next
	// to a strong one stay in the context (a keyword-only match from hybrid
	// search scores low on cosine but is still wanted).
	if !slices.ContainsFunc(hits, func(h vector.Result) bool { return float64(h.Score) >= r.minScore }) {
		return nil, nil
	}

	var results []SearchResult
	for _, h := range hits {
		res := SearchResult{
			ChunkID: h.ID,
			Source:  h.Metadata["source"],
			Content: h.Content,
			Score:   float64(h.Score),
		}
		if h.Metadata["type"] == "image" {
			res.ImagePath = h.Metadata["image_path"]
		}
		results = append(results, res)
	}
	return results, nil
}

func (r *Retriever) buildQuery(ctx context.Context, history []llm.Message) string {
	if r.rewriter != nil {
		if q, err := r.rewriter.Rewrite(ctx, history); err == nil && q != "" {
			return q
		}
	}
	return lastUserMessage(history)
}

func lastUserMessage(history []llm.Message) string {
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role == "user" {
			return history[i].Content
		}
	}
	return ""
}
