package rag

import (
	"context"
	"godocs-rag/llm"
	"godocs-rag/vector"
	"slices"
	"testing"
)

type stubEmbedder struct{}

func (stubEmbedder) EmbedDocuments(_ context.Context, texts []string) ([][]float32, error) {
	return make([][]float32, len(texts)), nil
}
func (stubEmbedder) EmbedQuery(context.Context, string) ([]float32, error) { return []float32{1}, nil }
func (stubEmbedder) Scheme() string                                        { return "stub" }

type stubStore struct{ hits []vector.Result }

func (stubStore) ReplaceSource(context.Context, string, []vector.Document) error { return nil }
func (s stubStore) Query(context.Context, []float32, int) ([]vector.Result, error) {
	return s.hits, nil
}
func (stubStore) Close() error { return nil }

func hit(id string, score float32) vector.Result {
	return vector.Result{Document: vector.Document{ID: id}, Score: score}
}

// MinScore decides whether the question is in scope at all, from its best
// hit. It must not drop weak hits next to a strong one: a hybrid ranker's
// keyword-only matches have a low cosine score but belong in the context.
func TestSearchGatesOnBestHit(t *testing.T) {
	for _, tc := range []struct {
		name string
		hits []vector.Result
		want []string
	}{
		{"best hit passes: keep every hit", []vector.Result{hit("a", 0.7), hit("b", 0.4)}, []string{"a", "b"}},
		{"best hit need not come first", []vector.Result{hit("a", 0.4), hit("b", 0.7)}, []string{"a", "b"}},
		{"best hit fails: nothing", []vector.Result{hit("a", 0.5), hit("b", 0.4)}, nil},
		{"no hits", nil, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := New(stubEmbedder{}, stubStore{hits: tc.hits}, Options{MinScore: 0.6})
			got, err := r.Search(context.Background(), []llm.Message{{Role: "user", Content: "q"}})
			if err != nil {
				t.Fatal(err)
			}
			var ids []string
			for _, h := range got {
				ids = append(ids, h.ChunkID)
			}
			if !slices.Equal(ids, tc.want) {
				t.Fatalf("got %q, want %q", ids, tc.want)
			}
		})
	}
}
