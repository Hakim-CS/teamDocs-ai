// Package app wires the components together and owns their lifecycle:
// LLM clients, the vector store, the ingest watcher, the web server and
// the foreground REPL.
package app

import (
	"context"
	"godocs-rag/chat"
	"godocs-rag/config"
	"godocs-rag/ingest"
	"godocs-rag/llm"
	"godocs-rag/rag"
	"godocs-rag/vector"
	"godocs-rag/vector/pgvector"
	"godocs-rag/web"
	"log"
	"os"
	"sync"
)

// Run starts the background services and blocks on the REPL. When the
// REPL exits (or ctx is cancelled) everything is shut down before Run
// returns.
func Run(parent context.Context, cfg config.Config) error {
	// Status lines go to stderr so they don't interleave with chat on stdout.
	logger := log.New(os.Stderr, "[rag] ", log.LstdFlags)

	ctx, cancel := context.WithCancel(parent)
	// close the database and finish the thread
	defer cancel()

	client := llm.New(cfg)
	embedder := llm.NewEmbedder(cfg)

	// Without a store the app still chats, just without retrieval.
	store, err := openStore(ctx, cfg)
	if err != nil {
		logger.Printf("vector store disabled: %v", err)
	}

	var wg sync.WaitGroup
	var retriever *rag.Retriever
	if store != nil {
		defer store.Close()
		logger.Printf("vector store ready")
		// A mismatch still returns results, just worse ones, so say so.
		if pg, ok := store.(*pgvector.Store); ok {
			if err := pg.CheckScheme(ctx, embedder.Scheme()); err != nil {
				logger.Printf("warning: %v", err)
			}
		}

		wg.Go(func() {
			opts := ingest.Options{SourceDir: cfg.IngestDir, ProcessedDir: cfg.ProcessedDir}
			if err := ingest.Watch(ctx, opts, embedder, store, logger); err != nil && ctx.Err() == nil {
				logger.Printf("watcher stopped: %v", err)
			}
		})
		logger.Printf("watching %s for new documents", cfg.IngestDir)

		retriever = rag.New(embedder, store, rag.Options{
			MinScore: cfg.MinScore,
			Rewriter: rag.NewRewriter(client),
		})
	}

	srv, err := web.New(client, embedder, retriever, web.Options{
		SystemPromptFile: cfg.SystemPromptFile,
		Store:            store,
		ProcessedDir:     cfg.ProcessedDir,
		ImagesDir:        cfg.ImageDir,
		Ready:            store != nil,
	})
	if err != nil {
		logger.Printf("web server disabled: %v", err)
	} else {
		wg.Go(func() {
			if err := srv.Run(ctx, cfg.HTTPAddr); err != nil && ctx.Err() == nil {
				logger.Printf("web server stopped: %v", err)
			}
		})
		logger.Printf("web chat at http://%s/chat", cfg.HTTPAddr)
	}

	replErr := chat.RunREPL(ctx, client, retriever, chat.Options{
		SystemPromptFile: cfg.SystemPromptFile,
	})

	cancel()
	wg.Wait()
	return replErr
}

// openStore returns (nil, nil) when DATABASE_URL is unset: retrieval is
// disabled, not broken. It returns an untyped nil on error so the
// caller's "store != nil" check works (a typed nil *Store boxed in the
// interface would be non-nil).
func openStore(ctx context.Context, cfg config.Config) (vector.Store, error) {
	if cfg.DatabaseURL == "" {
		return nil, nil
	}
	s, err := pgvector.New(ctx, pgvector.Options{
		DSN:          cfg.DatabaseURL,
		EmbeddingDim: cfg.EmbeddingDim,
	})
	if err != nil {
		return nil, err
	}
	return s, nil
}
