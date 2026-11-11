package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand/v2"
	"os"
	"slices"
)

// Two eval runs answer the same questions, so they are compared pairwise:
// McNemar's exact test on per-question recall hits and on out-of-scope
// abstentions, and a bootstrap over questions for MRR. AUC measures how
// well the best similarity separates in-scope from out-of-scope questions,
// independent of any floor.

// CaseResult is one question's outcome, written by -out.
type CaseResult struct {
	Question  string  `json:"q"`
	Source    string  `json:"source,omitempty"`
	Kind      string  `json:"kind,omitempty"`
	Rank     int     `json:"rank"`       // first relevant hit, 1-based; 0 = miss
	Hits      int     `json:"hits"`       // hits returned after the floor
	BestScore float64 `json:"best_score"` // top similarity before the floor
}

// Results is what -out writes and -compare reads.
type Results struct {
	Settings Baseline     `json:"settings"`
	Cases    []CaseResult `json:"cases"`
}

type comparison struct {
	Lost, Gained          []string // in-scope questions whose recall@k hit flipped
	P                     float64  // McNemar exact, two-sided
	MRRDiff, MRRLo, MRRHi float64  // b - a, paired bootstrap 95% CI
	Refused, Answered     []string // out-of-scope questions A answered and B refused, and the reverse
	AbstainP              float64  // McNemar exact on those flips
	AUCa, AUCb            float64
}

func compare(a, b Results) (comparison, error) {
	byQ := make(map[string]CaseResult, len(a.Cases))
	for _, c := range a.Cases {
		byQ[c.Question] = c
	}
	if len(b.Cases) != len(byQ) {
		return comparison{}, errors.New("the runs cover different questions")
	}
	var cmp comparison
	var rrA, rrB []float64
	for _, cb := range b.Cases {
		ca, ok := byQ[cb.Question]
		if !ok || ca.Source != cb.Source {
			return comparison{}, fmt.Errorf("%q is not in both runs with the same label", cb.Question)
		}
		if cb.Source == "" {
			switch answeredA, answeredB := ca.Hits > 0, cb.Hits > 0; {
			case answeredA && !answeredB:
				cmp.Refused = append(cmp.Refused, cb.Question)
			case !answeredA && answeredB:
				cmp.Answered = append(cmp.Answered, cb.Question)
			}
			continue
		}
		switch hitA, hitB := ca.Rank > 0, cb.Rank > 0; {
		case hitA && !hitB:
			cmp.Lost = append(cmp.Lost, cb.Question)
		case !hitA && hitB:
			cmp.Gained = append(cmp.Gained, cb.Question)
		}
		rrA = append(rrA, reciprocalRank(ca.Rank))
		rrB = append(rrB, reciprocalRank(cb.Rank))
	}
	cmp.P = mcnemarExact(len(cmp.Lost), len(cmp.Gained))
	cmp.AbstainP = mcnemarExact(len(cmp.Refused), len(cmp.Answered))
	cmp.MRRDiff, cmp.MRRLo, cmp.MRRHi = bootstrapMeanDiff(rrA, rrB, 10000, 1)
	cmp.AUCa, cmp.AUCb = separation(a.Cases), separation(b.Cases)
	return cmp, nil
}

// only keeps the questions of one kind, so a slice can be compared on its own.
func only(r Results, kind string) Results {
	out := Results{Settings: r.Settings}
	for _, c := range r.Cases {
		if c.Kind == kind {
			out.Cases = append(out.Cases, c)
		}
	}
	return out
}

func reciprocalRank(rank int) float64 {
	if rank == 0 {
		return 0
	}
	return 1 / float64(rank)
}

// mcnemarExact is the two-sided exact McNemar p-value: under "no
// difference", each of the b+c discordant questions is a fair coin flip.
func mcnemarExact(b, c int) float64 {
	n := b + c
	if n == 0 {
		return 1
	}
	var tail, choose float64 = 0, 1 // choose = C(n, i)
	for i := 0; i <= min(b, c); i++ {
		tail += choose
		choose = choose * float64(n-i) / float64(i+1)
	}
	return min(1, 2*tail/math.Ldexp(1, n))
}

// bootstrapMeanDiff returns mean(b-a) and a 95% percentile interval from
// resampling questions with replacement. The seed is fixed so a report
// can be reproduced exactly.
func bootstrapMeanDiff(a, b []float64, iters int, seed uint64) (mean, lo, hi float64) {
	n := len(a)
	if n == 0 {
		return 0, 0, 0
	}
	d := make([]float64, n)
	for i := range a {
		d[i] = b[i] - a[i]
		mean += d[i]
	}
	mean /= float64(n)

	rng := rand.New(rand.NewPCG(seed, seed))
	means := make([]float64, iters)
	for it := range means {
		var s float64
		for range n {
			s += d[rng.IntN(n)]
		}
		means[it] = s / float64(n)
	}
	slices.Sort(means)
	return mean, means[int(0.025*float64(iters))], means[int(0.975*float64(iters))]
}

// auc is the probability that a random positive outscores a random
// negative (ties count half): the Mann-Whitney form of ROC AUC.
func auc(pos, neg []float64) float64 {
	if len(pos) == 0 || len(neg) == 0 {
		return math.NaN()
	}
	var wins float64
	for _, p := range pos {
		for _, q := range neg {
			switch {
			case p > q:
				wins++
			case p == q:
				wins += 0.5
			}
		}
	}
	return wins / float64(len(pos)*len(neg))
}

// separation is the AUC of best score as an in-scope classifier.
func separation(cases []CaseResult) float64 {
	var pos, neg []float64
	for _, c := range cases {
		if c.Source != "" {
			pos = append(pos, c.BestScore)
		} else {
			neg = append(neg, c.BestScore)
		}
	}
	return auc(pos, neg)
}

func readResults(path string) (Results, error) {
	var r Results
	raw, err := os.ReadFile(path)
	if err != nil {
		return r, err
	}
	if err := json.Unmarshal(raw, &r); err != nil {
		return r, fmt.Errorf("%s: %w", path, err)
	}
	return r, nil
}

func runCompare(pathA, pathB string) error {
	a, err := readResults(pathA)
	if err != nil {
		return err
	}
	b, err := readResults(pathB)
	if err != nil {
		return err
	}
	cmp, err := compare(a, b)
	if err != nil {
		return err
	}
	fmt.Printf("## %s → %s\n\n", pathA, pathB)
	fmt.Printf("| | A | B |\n|---|---:|---:|\n")
	fmt.Printf("| Recall@%d | %.3f | %.3f |\n", a.Settings.K, a.Settings.RecallAtK, b.Settings.RecallAtK)
	fmt.Printf("| MRR | %.3f | %.3f |\n", a.Settings.MRR, b.Settings.MRR)
	fmt.Printf("| Abstention | %.3f | %.3f |\n", a.Settings.Abstention, b.Settings.Abstention)
	fmt.Printf("| Separation AUC | %.3f | %.3f |\n\n", cmp.AUCa, cmp.AUCb)
	fmt.Printf("Recall flips: %d gained, %d lost · McNemar exact p = %.4f\n", len(cmp.Gained), len(cmp.Lost), cmp.P)
	fmt.Printf("MRR difference (B-A): %+.3f, 95%% CI [%+.3f, %+.3f]\n", cmp.MRRDiff, cmp.MRRLo, cmp.MRRHi)
	fmt.Printf("Abstention flips (out-of-scope): %d now refused, %d now answered · McNemar exact p = %.4f\n",
		len(cmp.Refused), len(cmp.Answered), cmp.AbstainP)
	for _, kind := range kinds {
		if as, bs := only(a, kind), only(b, kind); len(as.Cases) > 0 {
			if sc, err := compare(as, bs); err == nil {
				fmt.Printf("%s questions (%d): %d gained, %d lost · McNemar exact p = %.4f · MRR %+.3f, 95%% CI [%+.3f, %+.3f]\n",
					kind, len(as.Cases), len(sc.Gained), len(sc.Lost), sc.P, sc.MRRDiff, sc.MRRLo, sc.MRRHi)
			}
		}
	}
	for _, q := range cmp.Gained {
		fmt.Printf("- gained: %s\n", q)
	}
	for _, q := range cmp.Lost {
		fmt.Printf("- lost: %s\n", q)
	}
	for _, q := range cmp.Refused {
		fmt.Printf("- now refused: %s\n", q)
	}
	for _, q := range cmp.Answered {
		fmt.Printf("- now answered: %s\n", q)
	}
	return nil
}

func writeResults(path string, r Results) error {
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

// half assigns a question to "dev" or "test" by a hash of its text, so the
// split is stable when cases are added and doesn't depend on file order.
func half(q string) string {
	h := fnv.New32a()
	h.Write([]byte(q))
	if h.Sum32()%2 == 0 {
		return "dev"
	}
	return "test"
}
