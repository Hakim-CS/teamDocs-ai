package ingest

import (
	"context"
	"godocs-rag/vector"
	"testing"
)

type fakeEmbedder struct{ scheme string }

func (f fakeEmbedder) EmbedDocuments(_ context.Context, texts []string) ([][]float32, error) {
	return make([][]float32, len(texts)), nil
}
func (f fakeEmbedder) EmbedQuery(context.Context, string) ([]float32, error) { return nil, nil }
func (f fakeEmbedder) Scheme() string                                        { return f.scheme }

type recordingStore struct{ docs []vector.Document }

func (s *recordingStore) ReplaceSource(_ context.Context, _ string, docs []vector.Document) error {
	s.docs = docs
	return nil
}
func (s *recordingStore) Query(context.Context, []float32, int) ([]vector.Result, error) {
	return nil, nil
}
func (s *recordingStore) Close() error { return nil }

// go doc output is split per declaration, not into byte windows.
func TestIndexSplitsGoDocOnDeclarations(t *testing.T) {
	store := &recordingStore{}
	n, err := ProcessContent(context.Background(), "demo.txt", []byte(sampleGoDoc), Options{}, fakeEmbedder{"m"}, store)
	if err != nil {
		t.Fatal(err)
	}
	if n != 5 || len(store.docs) != 5 {
		t.Fatalf("stored %d chunks (returned %d), want 5: overview, A, B, T, M", len(store.docs), n)
	}
}

// Every chunk records how it was embedded, so a mixed index is detectable.
func TestIndexRecordsEmbedScheme(t *testing.T) {
	store := &recordingStore{}
	if _, err := ProcessContent(context.Background(), "a.txt", []byte("hello"), Options{}, fakeEmbedder{"m|d: "}, store); err != nil {
		t.Fatal(err)
	}
	if len(store.docs) != 1 || store.docs[0].Metadata["embed_scheme"] != "m|d: " {
		t.Fatalf("docs = %+v, want one chunk with embed_scheme %q", store.docs, "m|d: ")
	}
}
