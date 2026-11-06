// Package ingest turns documents into embedded chunks in a
// vector.Store: read → chunk → embed → atomically replace the source's
// previous chunks. Watch runs it continuously over a directory.
package ingest

import (
	"context"
	"errors"
	"fmt"
	"godocs-rag/llm"
	"godocs-rag/vector"
	"maps"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// Default chunking parameters, in bytes. ~1000 bytes holds a paragraph
// or two of prose; the 100-byte overlap keeps a sentence that straddles
// a cut intact in at least one chunk.
const (
	defaultChunkSize    = 1000
	defaultChunkOverlap = 100
)

type Options struct {
	// SourceDir is watched (non-recursively) for new files.
	SourceDir string
	// ProcessedDir receives originals after a successful ingest. It
	// must differ from SourceDir or moved files would loop forever.
	ProcessedDir string
	// ChunkSize and ChunkOverlap default when zero.
	ChunkSize    int
	ChunkOverlap int
}

// ProcessFile reads path and ingests it under its basename.
func ProcessFile(ctx context.Context, path string, opts Options, embedder llm.Embedder, store vector.Store) (int, error) {
	if !IsSupported(path) {
		return 0, fmt.Errorf("unsupported format: %s", filepath.Ext(path))
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0, fmt.Errorf("read: %w", err)
	}
	return ProcessContent(ctx, filepath.Base(path), raw, opts, embedder, store)
}

// ProcessContent ingests an in-memory document under the basename of
// source and returns the number of chunks stored. Re-ingesting the same
// source replaces its previous chunks atomically.
func ProcessContent(ctx context.Context, source string, content []byte, opts Options, embedder llm.Embedder, store vector.Store) (int, error) {
	base := filepath.Base(source)
	if !IsSupported(base) {
		return 0, fmt.Errorf("unsupported format: %s", filepath.Ext(base))
	}
	return index(ctx, base, string(content), nil, opts, embedder, store)
}

// index is the shared chunk → embed → replace pipeline for text and
// image descriptions. extra is merged into every chunk's metadata.
func index(ctx context.Context, source, text string, extra map[string]string, opts Options, embedder llm.Embedder, store vector.Store) (int, error) {
	if embedder == nil || store == nil {
		return 0, errors.New("embedder and vector store are required")
	}

	size, overlap := opts.ChunkSize, opts.ChunkOverlap
	if size <= 0 {
		size = defaultChunkSize
	}
	if overlap <= 0 {
		overlap = defaultChunkOverlap
	}

	split := chunk
	if isGoDoc(text) {
		split = chunkGoDoc
	}
	chunks := split(text, size, overlap)
	if len(chunks) == 0 {
		return 0, errors.New("document is empty")
	}

	vectors, err := embedder.EmbedDocuments(ctx, chunks)
	if err != nil {
		return 0, fmt.Errorf("embed: %w", err)
	}
	if len(vectors) != len(chunks) {
		return 0, fmt.Errorf("embed: got %d vectors for %d chunks", len(vectors), len(chunks))
	}

	ingestedAt := time.Now().UTC().Format(time.RFC3339)
	docs := make([]vector.Document, len(chunks))
	for i, c := range chunks {
		meta := map[string]string{
			"source":       source,
			"chunk_index":  strconv.Itoa(i),
			"chunks":       strconv.Itoa(len(chunks)),
			"ingested_at":  ingestedAt,
			"embed_scheme": embedder.Scheme(),
		}
		maps.Copy(meta, extra)
		docs[i] = vector.Document{
			ID:        fmt.Sprintf("%s#%d", source, i),
			Content:   c,
			Metadata:  meta,
			Embedding: vectors[i],
		}
	}

	if err := store.ReplaceSource(ctx, source, docs); err != nil {
		return 0, err
	}
	return len(chunks), nil
}

// IsSupported reports whether name is a text format ingest can read.
func IsSupported(name string) bool {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".txt", ".md", ".markdown":
		return true
	}
	return false
}
