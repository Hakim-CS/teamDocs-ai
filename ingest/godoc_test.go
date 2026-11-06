package ingest

import (
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
)

const demoHeader = `package demo // import "example.com/demo"`

const sampleGoDoc = demoHeader + `

Package demo does things.

FUNCTIONS

func A()
    A does a.

func B()
    B does b.

TYPES

type T struct {
	// Has unexported fields.
}
    T is a thing.

func (t *T) M()
    M does m.
`

func TestChunkGoDocSplitsOnDeclarations(t *testing.T) {
	got := chunkGoDoc(sampleGoDoc, 1000, 100)
	want := []string{"Package demo does things.", "func A()", "func B()", "type T struct", "func (t *T) M()"}
	if len(got) != len(want) {
		t.Fatalf("got %d chunks, want %d: %q", len(got), len(want), got)
	}
	for i, c := range got {
		if !strings.HasPrefix(c, demoHeader) || !strings.Contains(c, want[i]) {
			t.Errorf("chunk %d = %q, want header + %q", i, c, want[i])
		}
		if strings.Contains(c, "FUNCTIONS") || strings.Contains(c, "TYPES") {
			t.Errorf("chunk %d keeps a section header: %q", i, c)
		}
	}
}

func TestChunkGoDocSplitsLongDeclarationKeepingSignature(t *testing.T) {
	text := demoHeader + "\n\nFUNCTIONS\n\nfunc Long()\n" + strings.Repeat("    Long does a lot of things.\n", 100)
	got := chunkGoDoc(text, 300, 30)
	if len(got) < 2 {
		t.Fatalf("got %d chunks, want several", len(got))
	}
	for i, c := range got {
		if !strings.HasPrefix(c, demoHeader) || !strings.Contains(c, "func Long()") {
			t.Errorf("chunk %d lost its header or signature: %q", i, c)
		}
		if len(c) > 300 {
			t.Errorf("chunk %d is %d bytes, want <= 300", i, len(c))
		}
	}
}

func TestChunkGoDocNormalizesCRLF(t *testing.T) {
	crlf := strings.ReplaceAll(sampleGoDoc, "\n", "\r\n")
	if got, want := chunkGoDoc(crlf, 1000, 100), chunkGoDoc(sampleGoDoc, 1000, 100); !slices.Equal(got, want) {
		t.Fatalf("CRLF chunks differ:\n%q\n%q", got, want)
	}
}

func TestChunkGoDocSkipsEmptyOverview(t *testing.T) {
	got := chunkGoDoc(demoHeader+"\n\nFUNCTIONS\n\nfunc A()\n    A does a.\n", 1000, 100)
	if len(got) != 1 || !strings.Contains(got[0], "func A()") {
		t.Fatalf("got %q, want one chunk for func A", got)
	}
}

// A size too small to hold the header must still chunk, not panic.
func TestChunkGoDocTinySizeFallsBackToPlainChunking(t *testing.T) {
	if got := chunkGoDoc(sampleGoDoc, 20, 5); len(got) == 0 {
		t.Fatal("got no chunks")
	}
}

func TestIsGoDocRequiresImportComment(t *testing.T) {
	if !isGoDoc(sampleGoDoc) || isGoDoc("package main\n\nfunc main() {}\n") || isGoDoc("Some notes.") {
		t.Fatal("isGoDoc must accept go doc output only")
	}
}

// The tests below run on the real corpus, where the odd cases live: link
// definitions at column 0, grouped const/var blocks, structs closed by a
// column-0 brace.

var (
	sectionLine = regexp.MustCompile(`(?m)^(CONSTANTS|VARIABLES|FUNCTIONS|TYPES)\n`)
	declLine    = regexp.MustCompile(`(?m)^(func|type|var|const) `)
)

func corpusTexts(t *testing.T) map[string]string {
	t.Helper()
	paths, _ := filepath.Glob(filepath.Join("..", "corpus", "*.txt"))
	if len(paths) == 0 {
		t.Skip("corpus/ not available")
	}
	texts := make(map[string]string, len(paths))
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		texts[filepath.Base(p)] = string(b)
	}
	return texts
}

// go doc prints one column-0 func/type/var/const line per declaration, and
// each must become exactly one unit.
func TestParseGoDocFindsEveryDeclaration(t *testing.T) {
	for name, text := range corpusTexts(t) {
		loc := sectionLine.FindStringIndex(text)
		if loc == nil {
			t.Fatalf("%s: no section header", name)
		}
		want := len(declLine.FindAllString(text[loc[0]:], -1))
		if _, _, units := parseGoDoc(text); len(units) != want {
			t.Errorf("%s: %d units, want %d declarations", name, len(units), want)
		}
	}
}

// Splitting only regroups text: nothing dropped, nothing duplicated.
func TestParseGoDocLosesNothing(t *testing.T) {
	for name, text := range corpusTexts(t) {
		_, body, _ := strings.Cut(text, "\n")
		want := strings.Fields(sectionLine.ReplaceAllString(body, ""))
		_, overview, units := parseGoDoc(text)
		got := strings.Fields(overview + "\n" + strings.Join(units, "\n"))
		if !slices.Equal(got, want) {
			t.Errorf("%s: %d words after splitting, want %d", name, len(got), len(want))
		}
	}
}

// The embedder gets at most size bytes per chunk, header included.
func TestChunkGoDocRespectsSizeOnCorpus(t *testing.T) {
	for name, text := range corpusTexts(t) {
		for i, c := range chunkGoDoc(text, 1000, 100) {
			if len(c) > 1000 {
				t.Errorf("%s chunk %d is %d bytes, want <= 1000", name, i, len(c))
			}
		}
	}
}
