// Package pgvector implements vector.Store on Postgres with the
// pgvector extension: one documents table, an HNSW cosine index, and
// JSONB metadata. The schema is created idempotently on startup; the
// embedding dimension is baked into the column type, so changing
// models means recreating the table.
package pgvector

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"godocs-rag/vector"
	"slices"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
	pgxvec "github.com/pgvector/pgvector-go/pgx"
)

// efSearch is hnsw.ef_search on every connection. pgvector's default of 40
// returned a different top-5 than exact search for 7-10% of the eval
// questions whenever the index was used, and which questions changed with
// how the graph was built. 400 matched exact search on every index state
// tried; 800 keeps a margin. See README.
const efSearch = 800

type Options struct {
	// DSN is a libpq-style connection string.
	DSN          string
	EmbeddingDim int
}

type Store struct {
	pool *pgxpool.Pool
}

// New connects, registers the vector type on every pooled connection,
// and ensures the schema exists.
func New(ctx context.Context, opts Options) (*Store, error) {
	if opts.DSN == "" {
		return nil, errors.New("pgvector: DSN is required")
	}
	if opts.EmbeddingDim <= 0 {
		return nil, errors.New("pgvector: EmbeddingDim must be > 0")
	}

	cfg, err := pgxpool.ParseConfig(opts.DSN)
	if err != nil {
		return nil, fmt.Errorf("parse DSN: %w", err)
	}

	// The extension must exist before the pool connects: RegisterTypes
	// in AfterConnect looks up the vector OID and fails on a fresh DB.
	if err := ensureExtension(ctx, opts.DSN); err != nil {
		return nil, fmt.Errorf("install extension: %w", err)
	}
	cfg.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
		if err := pgxvec.RegisterTypes(ctx, conn); err != nil {
			return err
		}
		_, err := conn.Exec(ctx, fmt.Sprintf("SET hnsw.ef_search = %d", efSearch))
		return err
	}

	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("connect: %w", err)
	}

	s := &Store{pool: pool}
	if err := s.migrate(ctx, opts.EmbeddingDim); err != nil {
		pool.Close()
		return nil, fmt.Errorf("migrate: %w", err)
	}
	return s, nil
}

func ensureExtension(ctx context.Context, dsn string) error {
	conn, err := pgx.Connect(ctx, dsn)
	if err != nil {
		return err
	}
	defer conn.Close(ctx)
	_, err = conn.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector")
	return err
}

func (s *Store) migrate(ctx context.Context, dim int) error {
	stmts := []string{
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS documents (
			id         TEXT PRIMARY KEY,
			content    TEXT NOT NULL,
			metadata   JSONB NOT NULL DEFAULT '{}'::jsonb,
			embedding  vector(%d) NOT NULL,
			created_at TIMESTAMPTZ NOT NULL DEFAULT now())`, dim),
		`CREATE INDEX IF NOT EXISTS documents_embedding_idx
			ON documents USING hnsw (embedding vector_cosine_ops)`,
		`CREATE INDEX IF NOT EXISTS documents_source_idx
			ON documents ((metadata->>'source'))`,
	}
	for _, q := range stmts {
		if _, err := s.pool.Exec(ctx, q); err != nil {
			return fmt.Errorf("exec: %w", err)
		}
	}
	return nil
}

// ReplaceSource deletes the source's old chunks and inserts docs in one
// transaction, so a failed insert leaves the previous version intact.
func (s *Store) ReplaceSource(ctx context.Context, source string, docs []vector.Document) error {
	if source == "" {
		return errors.New("pgvector: source is required")
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)

	if _, err := tx.Exec(ctx, `DELETE FROM documents WHERE metadata->>'source' = $1`, source); err != nil {
		return fmt.Errorf("delete %s: %w", source, err)
	}

	const insert = `
		INSERT INTO documents (id, content, metadata, embedding)
		VALUES ($1, $2, $3, $4)
		ON CONFLICT (id) DO UPDATE SET
			content = EXCLUDED.content,
			metadata = EXCLUDED.metadata,
			embedding = EXCLUDED.embedding`

	batch := &pgx.Batch{}
	for _, d := range docs {
		meta, err := json.Marshal(d.Metadata)
		if err != nil {
			return fmt.Errorf("metadata for %s: %w", d.ID, err)
		}
		if d.Metadata == nil {
			meta = []byte("{}")
		}
		batch.Queue(insert, d.ID, d.Content, meta, pgvector.NewVector(d.Embedding))
	}
	if err := tx.SendBatch(ctx, batch).Close(); err != nil {
		return fmt.Errorf("insert %s: %w", source, err)
	}

	return tx.Commit(ctx)
}

// Query returns the topK rows by cosine similarity (1 - cosine distance).
// ORDER BY uses the same <=> operator the HNSW index was built for, so
// the planner can use the index.
func (s *Store) Query(ctx context.Context, embedding []float32, topK int) ([]vector.Result, error) {
	if topK <= 0 {
		return nil, nil
	}
	return topByCosine(ctx, s.pool, embedding, topK)
}

// QueryExact is Query without the HNSW index: a sequential scan, which
// cannot miss a neighbour. Eval-only: it checks that retrieval does not
// depend on how the index was built.
func (s *Store) QueryExact(ctx context.Context, embedding []float32, topK int) ([]vector.Result, error) {
	if topK <= 0 {
		return nil, nil
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback(ctx)
	if _, err := tx.Exec(ctx, "SET LOCAL enable_indexscan = off"); err != nil {
		return nil, err
	}
	return topByCosine(ctx, tx, embedding, topK)
}

// querier is what a pool and a transaction have in common.
type querier interface {
	Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error)
}

func topByCosine(ctx context.Context, q querier, embedding []float32, topK int) ([]vector.Result, error) {
	rows, err := q.Query(ctx, `
		SELECT id, content, metadata, embedding <=> $1 AS distance
		FROM documents
		ORDER BY embedding <=> $1
		LIMIT $2`, pgvector.NewVector(embedding), topK)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []vector.Result
	for rows.Next() {
		var (
			r        vector.Result
			metaRaw  []byte
			distance float64
		)
		if err := rows.Scan(&r.ID, &r.Content, &metaRaw, &distance); err != nil {
			return nil, err
		}
		if err := json.Unmarshal(metaRaw, &r.Metadata); err != nil {
			return nil, fmt.Errorf("metadata for %s: %w", r.ID, err)
		}
		r.Score = float32(1 - distance)
		results = append(results, r)
	}
	return results, rows.Err()
}

// CheckScheme reports an error when stored chunks were embedded with a
// scheme other than want. Mixing schemes doesn't fail loudly: queries
// still return results, just worse ones.
func (s *Store) CheckScheme(ctx context.Context, want string) error {
	counts, err := s.embedSchemes(ctx)
	if err != nil {
		return err
	}
	return schemeErr(counts, want)
}

// embedSchemes counts rows by embed_scheme; rows without one count as "".
func (s *Store) embedSchemes(ctx context.Context) (map[string]int, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT coalesce(metadata->>'embed_scheme', ''), count(*)
		FROM documents GROUP BY 1`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	counts := map[string]int{}
	for rows.Next() {
		var scheme string
		var n int
		if err := rows.Scan(&scheme, &n); err != nil {
			return nil, err
		}
		counts[scheme] = n
	}
	return counts, rows.Err()
}

func schemeErr(counts map[string]int, want string) error {
	var other []string
	for scheme, n := range counts {
		if scheme != want {
			other = append(other, fmt.Sprintf("%d chunks as %q", n, scheme))
		}
	}
	if len(other) == 0 {
		return nil
	}
	slices.Sort(other)
	return fmt.Errorf("index has %s, but this run embeds as %q; re-ingest the documents", strings.Join(other, ", "), want)
}

// Reset deletes every row. Used by the eval harness, which owns its
// database; never called by the app.
func (s *Store) Reset(ctx context.Context) error {
	_, err := s.pool.Exec(ctx, `TRUNCATE documents`)
	return err
}

func (s *Store) Close() error {
	s.pool.Close()
	return nil
}
