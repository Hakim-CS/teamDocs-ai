package main

import (
	"math"
	"slices"
	"testing"
)

func TestMcNemarExact(t *testing.T) {
	for _, tc := range []struct {
		b, c int
		want float64
	}{
		{0, 0, 1},           // no discordant pairs: no evidence either way
		{1, 9, 22.0 / 1024}, // 2 * (C(10,0)+C(10,1)) / 2^10
		{5, 5, 1},           // balanced flips: capped at 1
		{0, 6, 2.0 / 64},    // 2 * C(6,0) / 2^6
		{9, 1, 22.0 / 1024}, // symmetric in b and c
	} {
		if got := mcnemarExact(tc.b, tc.c); math.Abs(got-tc.want) > 1e-12 {
			t.Errorf("mcnemarExact(%d, %d) = %v, want %v", tc.b, tc.c, got, tc.want)
		}
	}
}

// With a constant per-question improvement every resample has the same
// mean, so the interval collapses onto it.
func TestBootstrapMeanDiffConstantShift(t *testing.T) {
	a := []float64{0.5, 0.5, 0.5, 0.5}
	b := []float64{1, 1, 1, 1}
	mean, lo, hi := bootstrapMeanDiff(a, b, 1000, 1)
	if mean != 0.5 || lo != 0.5 || hi != 0.5 {
		t.Fatalf("got mean %v CI [%v, %v], want 0.5 [0.5, 0.5]", mean, lo, hi)
	}
}

func TestBootstrapMeanDiffIsPairedAndOrdered(t *testing.T) {
	a := []float64{0, 1, 0, 1, 0, 0, 1, 0}
	b := []float64{1, 1, 0, 1, 1, 0, 1, 1}
	mean, lo, hi := bootstrapMeanDiff(a, b, 2000, 7)
	if mean != 0.375 || lo > mean || hi < mean || lo < 0 {
		t.Fatalf("got mean %v CI [%v, %v], want mean 0.375 inside a non-negative CI", mean, lo, hi)
	}
}

// AUC is the probability that a random in-scope question scores above a
// random out-of-scope one; ties count half.
func TestAUC(t *testing.T) {
	if got := auc([]float64{0.9, 0.8}, []float64{0.7, 0.85}); got != 0.75 {
		t.Errorf("auc = %v, want 0.75", got)
	}
	if got := auc([]float64{0.5}, []float64{0.5}); got != 0.5 {
		t.Errorf("auc with a tie = %v, want 0.5", got)
	}
}

func TestCompareListsFlips(t *testing.T) {
	a := Results{Cases: []CaseResult{
		{Question: "kept", Source: "a.txt", Rank: 1, BestScore: 0.8},
		{Question: "lost", Source: "a.txt", Rank: 2, BestScore: 0.7},
		{Question: "gained", Source: "a.txt", Rank: 0, BestScore: 0.6},
		{Question: "oos", BestScore: 0.5},
	}}
	b := Results{Cases: []CaseResult{
		{Question: "gained", Source: "a.txt", Rank: 3, BestScore: 0.7},
		{Question: "kept", Source: "a.txt", Rank: 1, BestScore: 0.8},
		{Question: "lost", Source: "a.txt", Rank: 0, BestScore: 0.6},
		{Question: "oos", BestScore: 0.4},
	}}
	cmp, err := compare(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(cmp.Lost, []string{"lost"}) || !slices.Equal(cmp.Gained, []string{"gained"}) {
		t.Fatalf("lost %q gained %q", cmp.Lost, cmp.Gained)
	}
	if _, err := compare(a, Results{Cases: b.Cases[:3]}); err == nil {
		t.Fatal("want an error when the question sets differ")
	}
}

// An out-of-scope question is abstained when the floor left it no hits.
// Its flips are paired like recall flips; in-scope questions never count.
func TestOnlyKeepsOneKind(t *testing.T) {
	r := Results{Cases: []CaseResult{
		{Question: "a", Source: "x.txt", Kind: kindIdentifier},
		{Question: "b", Source: "x.txt"},
		{Question: "c"},
	}}
	got := only(r, kindIdentifier)
	if len(got.Cases) != 1 || got.Cases[0].Question != "a" {
		t.Errorf("only = %+v, want just question a", got.Cases)
	}
}

func TestCompareListsAbstentionFlips(t *testing.T) {
	a := Results{Cases: []CaseResult{
		{Question: "refused 1", Hits: 3},
		{Question: "refused 2", Hits: 2},
		{Question: "refused 3", Hits: 1},
		{Question: "answered 1", Hits: 0},
		{Question: "still answered", Hits: 2},
		{Question: "still refused", Hits: 0},
		{Question: "in scope", Source: "a.txt", Rank: 1, Hits: 5},
	}}
	b := Results{Cases: []CaseResult{
		{Question: "refused 1", Hits: 0},
		{Question: "refused 2", Hits: 0},
		{Question: "refused 3", Hits: 0},
		{Question: "answered 1", Hits: 4},
		{Question: "still answered", Hits: 5},
		{Question: "still refused", Hits: 0},
		{Question: "in scope", Source: "a.txt", Rank: 0, Hits: 0},
	}}
	cmp, err := compare(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(cmp.Refused, []string{"refused 1", "refused 2", "refused 3"}) || !slices.Equal(cmp.Answered, []string{"answered 1"}) {
		t.Fatalf("refused %q, answered %q", cmp.Refused, cmp.Answered)
	}
	if want := mcnemarExact(3, 1); cmp.AbstainP != want {
		t.Fatalf("AbstainP = %v, want %v", cmp.AbstainP, want)
	}
}
