// Command eval measures retrieval quality against a hand-labelled
// question set and fails when it regresses below a committed baseline.
//
// It ingests corpus/ into a dedicated database (EVAL_DATABASE_URL — it is
// truncated on every run), then runs each question in eval/golden.jsonl
// through the same rag.Retriever the app uses. A retrieved chunk counts
// as relevant when it comes from the case's source file and contains the
// case's phrase; questions with no source are out of scope and should
// retrieve nothing (abstention).
//
//	go run ./cmd/eval                  # report + gate against eval/baseline.json
//	go run ./cmd/eval -write-baseline  # accept current numbers as the baseline
//	go run ./cmd/eval -sweep           # recall/abstention per MinScore, for tuning
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"godocs-rag/config"
	"godocs-rag/ingest"
	"godocs-rag/llm"
	"godocs-rag/rag"
	"godocs-rag/vector"
	"godocs-rag/vector/pgvector"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"
)

// Case is one line of the golden set.
type Case struct {
	Question string `json:"q"`
	// Source is the corpus file that answers the question; empty means
	// the corpus does not answer it.
	Source string `json:"source"`
	// Contains is a phrase (case-insensitive) that the answering
	// passage includes. Labelling by phrase rather than chunk ID keeps
	// the set valid when chunking changes.
	Contains string `json:"contains"`
	// Kind optionally tags a slice of the set to report on its own, where
	// keyword search should help and embeddings may not: "identifier"
	// questions name an API whose doc describes what it does, "opaque" ones
	// name a variable, constant or error whose name says little.
	Kind string `json:"kind,omitempty"`
}

const (
	kindIdentifier = "identifier"
	kindOpaque     = "opaque"
)

var kinds = []string{kindIdentifier, kindOpaque}

// Metrics are all higher-is-better fractions in [0, 1].
type Metrics struct {
	RecallAt1  float64 `json:"recall_at_1"`
	RecallAtK  float64 `json:"recall_at_k"`
	MRR        float64 `json:"mrr"`
	Abstention float64 `json:"abstention"`
}

type Baseline struct {
	Metrics
	K           int     `json:"k"`
	MinScore    float64 `json:"min_score"`
	Model       string  `json:"embedding_model"`
	Scheme      string  `json:"embed_scheme"`
	QueryPrefix string  `json:"query_prefix"`
	Cases       int     `json:"cases"`
}

// outcome is what retrieval returned for one case.
type outcome struct {
	rank int // 1-based rank of the first relevant hit; 0 = none
	hits int
	// sourceHit is true when any hit came from the labelled source, which
	// separates ranking misses (right file) from coverage misses.
	sourceHit bool
	top       string // source of the top hit, for the miss list
	latency   time.Duration
}

type options struct {
	corpus, golden, baseline, out, split string
	k                                    int
	minScore, tolerance                  float64
	writeBaseline, skipIngest, sweep     bool
}

func main() {
	var o options
	flag.StringVar(&o.corpus, "corpus", "corpus", "directory of documents to index")
	flag.StringVar(&o.golden, "golden", "eval/golden.jsonl", "golden question set")
	flag.StringVar(&o.baseline, "baseline", "eval/baseline.json", "baseline to gate against")
	flag.StringVar(&o.out, "out", "", "also write per-question results to this JSON file (for -compare)")
	flag.StringVar(&o.split, "split", "", `evaluate one half of the questions: "dev" (for tuning) or "test" (for reporting)`)
	flag.IntVar(&o.k, "k", 5, "chunks retrieved per question")
	flag.Float64Var(&o.minScore, "min-score", -1, "similarity floor (default: RETRIEVAL_MIN_SCORE / app default)")
	flag.Float64Var(&o.tolerance, "tolerance", 0.02, "allowed drop per metric before failing (never less than 2 questions' worth)")
	flag.BoolVar(&o.writeBaseline, "write-baseline", false, "write current metrics to -baseline instead of gating")
	flag.BoolVar(&o.skipIngest, "skip-ingest", false, "reuse the already-indexed corpus")
	flag.BoolVar(&o.sweep, "sweep", false, "print recall and abstention across MinScore values, then exit")
	compareRuns := flag.Bool("compare", false, "compare two -out files given as arguments (A then B), then exit")
	flag.Parse()

	var err error
	switch {
	case *compareRuns && flag.NArg() != 2:
		err = errors.New("-compare needs two result files: A.json B.json")
	case *compareRuns:
		err = runCompare(flag.Arg(0), flag.Arg(1))
	default:
		err = run(o)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "eval:", err)
		os.Exit(1)
	}
}

func run(o options) error {
	ctx := context.Background()
	cfg := config.Load()
	if o.minScore < 0 {
		o.minScore = cfg.MinScore
	}

	dsn := os.Getenv("EVAL_DATABASE_URL")
	if dsn == "" {
		return errors.New("EVAL_DATABASE_URL is required (a dedicated database; it is truncated on every run)")
	}
	if dsn == cfg.DatabaseURL {
		return errors.New("EVAL_DATABASE_URL must not be the app's DATABASE_URL")
	}

	cases, err := loadCases(o.golden)
	if err != nil {
		return err
	}
	if err := checkLabels(cases, o.corpus); err != nil {
		return err
	}
	switch o.split {
	case "":
	case "dev", "test":
		if o.writeBaseline {
			return errors.New("the baseline covers every question; drop -split")
		}
		cases = slices.DeleteFunc(cases, func(c Case) bool { return half(c.Question) != o.split })
	default:
		return errors.New(`-split must be "dev" or "test"`)
	}

	store, err := pgvector.New(ctx, pgvector.Options{DSN: dsn, EmbeddingDim: cfg.EmbeddingDim})
	if err != nil {
		return err
	}
	defer store.Close()
	embedder := llm.NewEmbedder(cfg)

	if !o.skipIngest {
		if err := indexCorpus(ctx, o.corpus, embedder, store); err != nil {
			return err
		}
	}
	// With -skip-ingest the index may come from a run with other prefixes;
	// comparing across schemes would measure the mismatch, not the change.
	if err := store.CheckScheme(ctx, embedder.Scheme()); err != nil {
		return err
	}

	// The unfiltered pass gives each question's best score, for the sweep
	// and for the floor-independent separation AUC.
	rawOuts, scores, err := runCases(ctx, rag.New(embedder, store, rag.Options{TopK: o.k}), cases)
	if err != nil {
		return err
	}
	if o.sweep {
		printSweep(cases, rawOuts, scores)
		return nil
	}
	agree, err := indexAgreement(ctx, store, embedder, cases, o.k)
	if err != nil {
		return err
	}

	retriever := rag.New(embedder, store, rag.Options{TopK: o.k, MinScore: o.minScore})
	outs, _, err := runCases(ctx, retriever, cases)
	if err != nil {
		return err
	}
	cur := Baseline{
		Metrics: score(cases, outs), K: o.k, MinScore: o.minScore, Model: cfg.EmbeddingModel,
		Scheme: embedder.Scheme(), QueryPrefix: cfg.EmbeddingQueryPrefix, Cases: len(cases),
	}
	results := Results{Settings: cur, Cases: caseResults(cases, outs, scores)}
	printReport(cur, cases, outs, separation(results.Cases), agree)

	if o.out != "" {
		if err := writeResults(o.out, results); err != nil {
			return err
		}
	}
	if o.writeBaseline {
		data, _ := json.MarshalIndent(cur, "", "  ")
		if err := os.WriteFile(o.baseline, append(data, '\n'), 0o644); err != nil {
			return err
		}
		fmt.Printf("\nBaseline written to %s\n", o.baseline)
		return nil
	}
	neg := 0
	for _, c := range cases {
		if c.Source == "" {
			neg++
		}
	}
	return gate(cur, o.baseline, o.tolerance, len(cases)-neg, neg)
}

// caseResults pairs each case's filtered outcome with its best unfiltered score.
func caseResults(cases []Case, outs []outcome, scores [][]float64) []CaseResult {
	rs := make([]CaseResult, len(cases))
	for i, c := range cases {
		rs[i] = CaseResult{Question: c.Question, Source: c.Source, Kind: c.Kind, Rank: outs[i].rank, Hits: outs[i].hits}
		if len(scores[i]) > 0 {
			rs[i].BestScore = slices.Max(scores[i])
		}
	}
	return rs
}

func loadCases(path string) ([]Case, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var cases []Case
	sc := bufio.NewScanner(f)
	for line := 1; sc.Scan(); line++ {
		text := strings.TrimSpace(sc.Text())
		if text == "" {
			continue
		}
		var c Case
		if err := json.Unmarshal([]byte(text), &c); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, line, err)
		}
		if c.Question == "" || (c.Source == "") != (c.Contains == "") {
			return nil, fmt.Errorf("%s:%d: need q, and source+contains together", path, line)
		}
		if c.Kind != "" && (!slices.Contains(kinds, c.Kind) || c.Source == "") {
			return nil, fmt.Errorf("%s:%d: kind must be one of %q and needs a source", path, line, kinds)
		}
		cases = append(cases, c)
	}
	return cases, sc.Err()
}

// checkLabels fails fast if a label can't be satisfied by the corpus, so
// a typo or a corpus regeneration can't silently turn a case into a
// guaranteed miss.
func checkLabels(cases []Case, corpusDir string) error {
	var bad []string
	for _, c := range cases {
		if c.Source == "" {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(corpusDir, c.Source))
		if err != nil {
			bad = append(bad, fmt.Sprintf("%q: %v", c.Question, err))
			continue
		}
		if !containsFold(string(raw), c.Contains) {
			bad = append(bad, fmt.Sprintf("%q: %s does not contain %q", c.Question, c.Source, c.Contains))
		}
	}
	if len(bad) > 0 {
		return fmt.Errorf("invalid labels:\n  %s", strings.Join(bad, "\n  "))
	}
	return nil
}

func indexCorpus(ctx context.Context, dir string, embedder llm.Embedder, store *pgvector.Store) error {
	if err := store.Reset(ctx); err != nil {
		return err
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	start, total := time.Now(), 0
	for _, e := range entries {
		if e.IsDir() || !ingest.IsSupported(e.Name()) {
			continue
		}
		n, err := ingest.ProcessFile(ctx, filepath.Join(dir, e.Name()), ingest.Options{}, embedder, store)
		if err != nil {
			return fmt.Errorf("ingest %s: %w", e.Name(), err)
		}
		total += n
	}
	fmt.Fprintf(os.Stderr, "indexed %d chunks in %s\n", total, time.Since(start).Round(time.Millisecond))
	return nil
}

// runCases runs every question through the retriever. scores holds each
// case's hit scores (best first), used by the sweep.
func runCases(ctx context.Context, r *rag.Retriever, cases []Case) ([]outcome, [][]float64, error) {
	outs := make([]outcome, len(cases))
	scores := make([][]float64, len(cases))
	for i, c := range cases {
		start := time.Now()
		hits, err := r.Search(ctx, []llm.Message{{Role: "user", Content: c.Question}})
		if err != nil {
			return nil, nil, fmt.Errorf("%q: %w", c.Question, err)
		}
		outs[i] = judge(c, hits)
		outs[i].latency = time.Since(start)
		for _, h := range hits {
			scores[i] = append(scores[i], h.Score)
		}
	}
	return outs, scores, nil
}

func judge(c Case, hits []rag.SearchResult) outcome {
	o := outcome{hits: len(hits)}
	if len(hits) > 0 {
		o.top = hits[0].Source
	}
	for i, h := range hits {
		if c.Source == "" || h.Source != c.Source {
			continue
		}
		o.sourceHit = true
		if containsFold(h.Content, c.Contains) {
			o.rank = i + 1
			break
		}
	}
	return o
}

func score(cases []Case, outs []outcome) Metrics {
	var m Metrics
	var pos, neg float64
	for i, c := range cases {
		o := outs[i]
		if c.Source == "" {
			neg++
			if o.hits == 0 {
				m.Abstention++
			}
			continue
		}
		pos++
		if o.rank == 1 {
			m.RecallAt1++
		}
		if o.rank > 0 {
			m.RecallAtK++
			m.MRR += 1 / float64(o.rank)
		}
	}
	if pos > 0 {
		m.RecallAt1 /= pos
		m.RecallAtK /= pos
		m.MRR /= pos
	}
	if neg > 0 {
		m.Abstention /= neg
	}
	return m
}

// slice scores only the cases of one kind and returns how many there are.
func slice(cases []Case, outs []outcome, kind string) (int, Metrics) {
	var cs []Case
	var sub []outcome
	for i, c := range cases {
		if c.Kind == kind {
			cs, sub = append(cs, c), append(sub, outs[i])
		}
	}
	return len(cs), score(cs, sub)
}

func printReport(b Baseline, cases []Case, outs []outcome, separationAUC float64, indexAgree int) {
	lat := make([]time.Duration, len(outs))
	for i, o := range outs {
		lat[i] = o.latency
	}
	slices.Sort(lat)
	pct := func(p float64) time.Duration { return lat[int(p*float64(len(lat)-1))].Round(time.Millisecond) }

	var misses []string
	var rightFile, wrongFile int
	for i, c := range cases {
		o := outs[i]
		switch {
		case c.Source != "" && o.rank == 0:
			kind := "wrong file"
			if o.sourceHit {
				kind = "right file"
				rightFile++
			} else {
				wrongFile++
			}
			misses = append(misses, fmt.Sprintf("- miss (%s): %q (want %s, top: %s)", kind, c.Question, c.Source, orNone(o.top)))
		case c.Source == "" && o.hits > 0:
			misses = append(misses, fmt.Sprintf("- answered out-of-scope: %q (top: %s)", c.Question, o.top))
		}
	}

	fmt.Printf("## Retrieval eval\n\n")
	fmt.Printf("%d cases · k=%d · min_score=%.2f · %s · query prefix %q\n\n", b.Cases, b.K, b.MinScore, b.Scheme, b.QueryPrefix)
	fmt.Printf("| Metric | Value |\n|---|---:|\n")
	fmt.Printf("| Recall@1 | %.3f |\n| Recall@%d | %.3f |\n| MRR@%d | %.3f |\n", b.RecallAt1, b.K, b.RecallAtK, b.K, b.MRR)
	fmt.Printf("| Abstention (out-of-scope) | %.3f |\n", b.Abstention)
	for _, kind := range kinds {
		if n, m := slice(cases, outs, kind); n > 0 {
			fmt.Printf("| %s questions (%d): Recall@1 / Recall@%d / MRR | %.3f / %.3f / %.3f |\n", kind, n, b.K, m.RecallAt1, m.RecallAtK, m.MRR)
		}
	}
	fmt.Printf("| Misses (right file / wrong file) | %d / %d |\n", rightFile, wrongFile)
	fmt.Printf("| Separation AUC (best score, in- vs out-of-scope) | %.3f |\n", separationAUC)
	fmt.Printf("| Index matches exact search (top-%d scores) | %d / %d |\n", b.K, indexAgree, len(cases))
	fmt.Printf("| Latency p50 / p95 | %s / %s |\n", pct(0.5), pct(0.95))
	if indexAgree < len(cases) {
		fmt.Printf("\n**Warning:** the index returned different top-%d scores than a sequential scan for %d questions, so the numbers above depend on the index graph, not only on the embeddings and chunking.\n", b.K, len(cases)-indexAgree)
	}

	if len(misses) > 0 {
		fmt.Printf("\n<details><summary>%d failures</summary>\n\n%s\n\n</details>\n", len(misses), strings.Join(misses, "\n"))
	}
}

// gated applies the retriever's per-query floor to an unfiltered run: a
// question keeps all its hits when its best score reaches floor, none
// otherwise. TestGatedMatchesRetrieverAtEveryFloor keeps it equal to
// rag.Retriever.Search.
func gated(outs []outcome, scores [][]float64, floor float64) []outcome {
	out := make([]outcome, len(outs))
	for i, o := range outs {
		if len(scores[i]) > 0 && slices.Max(scores[i]) >= floor {
			out[i] = o
		}
	}
	return out
}

// sameScores reports whether two result lists have the same scores in the
// same order. Equal scores mean equal results up to ties, which may come
// back in a different order.
func sameScores(a, b []vector.Result) bool {
	return slices.EqualFunc(a, b, func(x, y vector.Result) bool { return math.Abs(float64(x.Score-y.Score)) < 1e-6 })
}

// indexAgreement counts the questions for which the index returns the same
// top-k scores as an exact scan. Fewer than len(cases) means the numbers
// depend on the index graph, not only on the embeddings and chunking.
func indexAgreement(ctx context.Context, store *pgvector.Store, embedder llm.Embedder, cases []Case, k int) (int, error) {
	agree := 0
	for _, c := range cases {
		vec, err := embedder.EmbedQuery(ctx, c.Question)
		if err != nil {
			return 0, fmt.Errorf("%q: %w", c.Question, err)
		}
		got, err := store.Query(ctx, vec, k)
		if err != nil {
			return 0, err
		}
		want, err := store.QueryExact(ctx, vec, k)
		if err != nil {
			return 0, err
		}
		if sameScores(got, want) {
			agree++
		}
	}
	return agree, nil
}

// printSweep shows how recall@k and abstention trade off as MinScore rises.
func printSweep(cases []Case, outs []outcome, scores [][]float64) {
	fmt.Printf("| MinScore | Recall@k | Abstention |\n|---:|---:|---:|\n")
	for t := 0.30; t <= 0.801; t += 0.025 {
		m := score(cases, gated(outs, scores, t))
		fmt.Printf("| %.3f | %.3f | %.3f |\n", t, m.RecallAtK, m.Abstention)
	}
}

// slackQuestions is how many questions a metric may lose before the gate
// fails. A fraction over few questions moves in big steps (one out-of-scope
// question is 1/30), and 7 of 30 sit within 0.01 of the floor, so a slightly
// different embedding build flips a few without any regression.
const slackQuestions = 2

// gate fails when a metric drops below baseline by more than the larger of
// tolerance and slackQuestions questions. pos and neg are the in-scope and
// out-of-scope question counts that the metrics are fractions of.
func gate(cur Baseline, path string, tolerance float64, pos, neg int) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read baseline (create one with -write-baseline): %w", err)
	}
	var base Baseline
	if err := json.Unmarshal(raw, &base); err != nil {
		return fmt.Errorf("parse baseline: %w", err)
	}
	if base.K != cur.K || base.MinScore != cur.MinScore || base.Model != cur.Model || base.Scheme != cur.Scheme ||
		base.QueryPrefix != cur.QueryPrefix || base.Cases != cur.Cases {
		return fmt.Errorf("settings differ from baseline (k, min_score, model, prefixes or case count); re-baseline with -write-baseline")
	}

	var failed []string
	check := func(name string, got, want float64, n int) {
		allowed := tolerance
		if n > 0 {
			allowed = max(allowed, slackQuestions/float64(n))
		}
		if got < want-allowed-1e-9 {
			failed = append(failed, fmt.Sprintf("%s %.3f < baseline %.3f - %.3f", name, got, want, allowed))
		}
	}
	check("recall@1", cur.RecallAt1, base.RecallAt1, pos)
	check("recall@k", cur.RecallAtK, base.RecallAtK, pos)
	check("mrr", cur.MRR, base.MRR, pos)
	check("abstention", cur.Abstention, base.Abstention, neg)
	if len(failed) > 0 {
		return fmt.Errorf("regression:\n  %s", strings.Join(failed, "\n  "))
	}
	fmt.Printf("\nGate passed (tolerance %.2f vs %s).\n", tolerance, path)
	return nil
}

func containsFold(s, sub string) bool {
	return strings.Contains(strings.ToLower(s), strings.ToLower(sub))
}

func orNone(s string) string {
	if s == "" {
		return "nothing retrieved"
	}
	return s
}
