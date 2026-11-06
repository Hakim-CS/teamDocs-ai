package main

import (
	"context"
	"godocs-rag/rag"
	"godocs-rag/vector"
	"math"
	"slices"
	"testing"
)

func TestJudgeAndScore(t *testing.T) {
	cases := []Case{
		{Question: "a", Source: "sync.txt", Contains: "WaitGroup"},
		{Question: "b", Source: "io.txt", Contains: "EOF"},
		{Question: "c", Source: "os.txt", Contains: "Getenv"},
		{Question: "out of scope"},
		{Question: "also out of scope"},
	}
	hits := [][]rag.SearchResult{
		{{Source: "sync.txt", Content: "a waitgroup waits"}},                     // rank 1 (case-insensitive)
		{{Source: "os.txt", Content: "EOF"}, {Source: "io.txt", Content: "EOF"}}, // wrong source first: rank 2
		{{Source: "os.txt", Content: "Setenv"}},                                  // right file, wrong passage: miss
		nil,                                                                      // abstained
		{{Source: "sync.txt"}},                                                   // answered out of scope
	}
	outs := make([]outcome, len(cases))
	for i := range cases {
		outs[i] = judge(cases[i], hits[i])
	}
	m := score(cases, outs)
	want := Metrics{RecallAt1: 1.0 / 3, RecallAtK: 2.0 / 3, MRR: 1.5 / 3, Abstention: 0.5}
	for name, pair := range map[string][2]float64{
		"recall@1": {m.RecallAt1, want.RecallAt1}, "recall@k": {m.RecallAtK, want.RecallAtK},
		"mrr": {m.MRR, want.MRR}, "abstention": {m.Abstention, want.Abstention},
	} {
		if math.Abs(pair[0]-pair[1]) > 1e-9 {
			t.Errorf("%s = %v, want %v", name, pair[0], pair[1])
		}
	}
}

// A miss where the right file was retrieved is a ranking problem, not a
// coverage problem; the report counts the two separately.
func TestJudgeMarksRightFileWrongPassage(t *testing.T) {
	c := Case{Question: "q", Source: "os.txt", Contains: "Getenv"}
	o := judge(c, []rag.SearchResult{{Source: "io.txt", Content: "Getenv"}, {Source: "os.txt", Content: "Setenv"}})
	if o.rank != 0 || !o.sourceHit {
		t.Fatalf("rank=%d sourceHit=%v, want 0 and true", o.rank, o.sourceHit)
	}
	if o := judge(c, []rag.SearchResult{{Source: "io.txt", Content: "Getenv"}}); o.sourceHit {
		t.Fatal("sourceHit = true for a wrong-file miss")
	}
}

// stubEmbedder tags each query with its position in questions, and
// stubStore answers with that question's canned hits.
type stubEmbedder struct{ questions []string }

func (stubEmbedder) EmbedDocuments(_ context.Context, texts []string) ([][]float32, error) {
	return make([][]float32, len(texts)), nil
}
func (e stubEmbedder) EmbedQuery(_ context.Context, q string) ([]float32, error) {
	return []float32{float32(slices.Index(e.questions, q))}, nil
}
func (stubEmbedder) Scheme() string { return "stub" }

type stubStore struct{ hits [][]vector.Result }

func (stubStore) ReplaceSource(context.Context, string, []vector.Document) error { return nil }
func (s stubStore) Query(_ context.Context, vec []float32, _ int) ([]vector.Result, error) {
	return s.hits[int(vec[0])], nil
}
func (stubStore) Close() error { return nil }

// The sweep gates one unfiltered run instead of querying again at every
// floor. That only measures what ships while it agrees with the retriever,
// so check the two against each other at floors on both sides of every score.
func TestGatedMatchesRetrieverAtEveryFloor(t *testing.T) {
	cases := []Case{
		{Question: "strong", Source: "a.txt", Contains: "answer"},
		{Question: "right passage is weak", Source: "a.txt", Contains: "answer"},
		{Question: "out of scope"},
	}
	hit := func(id, source, content string, score float32) vector.Result {
		return vector.Result{
			Document: vector.Document{ID: id, Content: content, Metadata: map[string]string{"source": source}},
			Score:    score,
		}
	}
	store := stubStore{hits: [][]vector.Result{
		{hit("a#0", "a.txt", "the answer", 0.8), hit("b#0", "b.txt", "noise", 0.6)},
		{hit("b#1", "b.txt", "noise", 0.6), hit("a#5", "a.txt", "the answer", 0.5)},
		{hit("c#0", "c.txt", "noise", 0.52)},
	}}
	embedder := stubEmbedder{questions: []string{cases[0].Question, cases[1].Question, cases[2].Question}}
	ctx := context.Background()

	raw, scores, err := runCases(ctx, rag.New(embedder, store, rag.Options{}), cases)
	if err != nil {
		t.Fatal(err)
	}
	for _, floor := range []float64{0, 0.45, 0.51, 0.55, 0.61, 0.7, 0.81} {
		real, _, err := runCases(ctx, rag.New(embedder, store, rag.Options{MinScore: floor}), cases)
		if err != nil {
			t.Fatal(err)
		}
		got := gated(raw, scores, floor)
		for i, want := range real {
			if got[i].rank != want.rank || got[i].hits != want.hits {
				t.Errorf("floor %.2f, %q: gated rank/hits = %d/%d, retriever = %d/%d",
					floor, cases[i].Question, got[i].rank, got[i].hits, want.rank, want.hits)
			}
		}
	}
}
