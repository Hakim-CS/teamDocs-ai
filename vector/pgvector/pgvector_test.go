package pgvector

import (
	"context"
	"fmt"
	"godocs-rag/vector"
	"math"
	"os"
	"slices"
	"strconv"
	"testing"
)

const testDim = 768

// TEST_DATABASE_URL must point at a disposable database (CI uses the
// eval service). The test only touches rows of its own source.
func openTest(t *testing.T) *Store {
	t.Helper()
	dsn := os.Getenv("TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("TEST_DATABASE_URL not set")
	}
	s, err := New(context.Background(), Options{DSN: dsn, EmbeddingDim: testDim})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func doc(id, source string, dim int) vector.Document {
	v := make([]float32, dim)
	v[0] = 1
	return vector.Document{ID: id, Content: id, Metadata: map[string]string{"source": source}, Embedding: v}
}

func countSource(t *testing.T, s *Store, source string) int {
	t.Helper()
	var n int
	if err := s.pool.QueryRow(context.Background(),
		`SELECT count(*) FROM documents WHERE metadata->>'source' = $1`, source).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

// A failed re-ingest must leave the previous version in place.
func TestReplaceSourceIsAtomic(t *testing.T) {
	s := openTest(t)
	ctx := context.Background()
	const src = "__test_replace__.txt"
	t.Cleanup(func() { s.ReplaceSource(ctx, src, nil) })

	if err := s.ReplaceSource(ctx, src, []vector.Document{doc(src+"#0", src, testDim), doc(src+"#1", src, testDim)}); err != nil {
		t.Fatal(err)
	}

	// The second document has the wrong dimension, so the insert fails
	// after the delete has already run inside the transaction.
	err := s.ReplaceSource(ctx, src, []vector.Document{doc(src+"#0", src, testDim), doc(src+"#1", src, 3)})
	if err == nil {
		t.Fatal("expected dimension error")
	}
	if n := countSource(t, s, src); n != 2 {
		t.Fatalf("after failed replace: %d chunks, want the original 2", n)
	}

	if err := s.ReplaceSource(ctx, src, []vector.Document{doc(src+"#0", src, testDim)}); err != nil {
		t.Fatal(err)
	}
	if n := countSource(t, s, src); n != 1 {
		t.Fatalf("after replace: %d chunks, want 1", n)
	}
}

// Rows are counted by the scheme they were embedded with; rows from before
// schemes were recorded count under "".
func TestEmbedSchemesCountsRows(t *testing.T) {
	s := openTest(t)
	ctx := context.Background()
	const src = "__test_scheme__.txt"
	t.Cleanup(func() { s.ReplaceSource(ctx, src, nil) })
	d0, d1 := doc(src+"#0", src, testDim), doc(src+"#1", src, testDim)
	d0.Metadata["embed_scheme"] = "__scheme__"
	d1.Metadata["embed_scheme"] = "__scheme__"
	if err := s.ReplaceSource(ctx, src, []vector.Document{d0, d1}); err != nil {
		t.Fatal(err)
	}
	counts, err := s.embedSchemes(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if counts["__scheme__"] != 2 {
		t.Fatalf("counts = %v, want 2 rows for __scheme__", counts)
	}
}

// A query embedded one way against documents embedded another still
// returns results, just worse ones, so a mismatch has to be reported.
func TestSchemeErrRejectsMixedIndex(t *testing.T) {
	for _, tc := range []struct {
		counts map[string]int
		ok     bool
	}{
		{map[string]int{"a": 3}, true},
		{map[string]int{}, true}, // empty index: nothing to mismatch
		{map[string]int{"a": 3, "": 1}, false},
		{map[string]int{"b": 3}, false},
	} {
		if err := schemeErr(tc.counts, "a"); (err == nil) != tc.ok {
			t.Errorf("schemeErr(%v, a) = %v, want ok=%v", tc.counts, err, tc.ok)
		}
	}
}

// QueryExact is the eval's ground truth, so it must return the true nearest
// neighbours in order. The oracle is cosine similarity computed here: chunk i
// points (1, 0.1i, 0, ...), so a smaller i is closer to the query (1, 0, ...)
// and every corpus row in a shared database is far behind them.
func TestQueryExactReturnsTrueNearestNeighbours(t *testing.T) {
	s := openTest(t)
	ctx := context.Background()
	const src = "__test_exact__.txt"
	t.Cleanup(func() { s.ReplaceSource(ctx, src, nil) })

	var docs []vector.Document
	for i := 12; i >= 1; i-- { // inserted far-to-near, so insertion order is no help
		d := doc(fmt.Sprintf("%s#%d", src, i), src, testDim)
		d.Embedding = make([]float32, testDim)
		d.Embedding[0], d.Embedding[1] = 1, float32(i)/10
		docs = append(docs, d)
	}
	if err := s.ReplaceSource(ctx, src, docs); err != nil {
		t.Fatal(err)
	}
	query := make([]float32, testDim)
	query[0] = 1

	got, err := s.QueryExact(ctx, query, 5)
	if err != nil {
		t.Fatal(err)
	}
	var wantIDs, gotIDs []string
	for i := 1; i <= 5; i++ {
		wantIDs = append(wantIDs, fmt.Sprintf("%s#%d", src, i))
	}
	for _, r := range got {
		gotIDs = append(gotIDs, r.ID)
	}
	if !slices.Equal(gotIDs, wantIDs) {
		t.Fatalf("got %v, want %v", gotIDs, wantIDs)
	}
	for i, r := range got {
		want := 1 / math.Sqrt(1+math.Pow(float64(i+1)/10, 2))
		if math.Abs(float64(r.Score)-want) > 1e-4 {
			t.Errorf("%s: score %v, want cosine %v", r.ID, r.Score, want)
		}
	}
}

// pgvector's default hnsw.ef_search of 40 let HNSW miss the true nearest
// neighbour for some questions (see README), so every connection in the
// pool must search wider.
func TestConnectionsSearchWiderThanPgvectorDefault(t *testing.T) {
	s := openTest(t)
	var got string
	if err := s.pool.QueryRow(context.Background(), `SHOW hnsw.ef_search`).Scan(&got); err != nil {
		t.Fatal(err)
	}
	if want := strconv.Itoa(efSearch); got != want {
		t.Fatalf("hnsw.ef_search = %s, want %s", got, want)
	}
}
