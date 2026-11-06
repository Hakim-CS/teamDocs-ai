package ingest

import "strings"

// go doc -all lists declarations under these column-0 headers. Doc prose is
// indented, so after the first header a func/type/var/const at column 0
// always starts a declaration.
var sectionHeaders = map[string]bool{"CONSTANTS": true, "VARIABLES": true, "FUNCTIONS": true, "TYPES": true}

// isGoDoc reports whether text is go doc output, which starts with
// `package name // import "path"`. Plain Go source starts the same way but
// has no import comment, so it keeps the generic chunker.
func isGoDoc(text string) bool {
	first, _, _ := strings.Cut(text, "\n")
	return strings.HasPrefix(first, "package ") && strings.Contains(first, `// import "`)
}

// parseGoDoc splits go doc output into the package line, the overview
// prose, and one unit per declaration (its code plus its doc comment).
func parseGoDoc(text string) (header, overview string, units []string) {
	first, body, _ := strings.Cut(strings.ReplaceAll(text, "\r\n", "\n"), "\n")

	var prose, cur []string
	inDecls := false
	flush := func() {
		if u := strings.TrimSpace(strings.Join(cur, "\n")); u != "" {
			units = append(units, u)
		}
		cur = nil
	}
	for _, line := range strings.Split(body, "\n") {
		switch {
		case sectionHeaders[strings.TrimRight(line, " \t")]:
			flush()
			inDecls = true
		case !inDecls:
			prose = append(prose, line)
		default:
			if startsDecl(line) {
				flush()
			}
			cur = append(cur, line)
		}
	}
	flush()
	return strings.TrimSpace(first), strings.TrimSpace(strings.Join(prose, "\n")), units
}

func startsDecl(line string) bool {
	for _, kw := range [...]string{"func ", "type ", "var ", "const "} {
		if strings.HasPrefix(line, kw) {
			return true
		}
	}
	return false
}

// chunkGoDoc makes one chunk per declaration. A byte window cuts across
// neighbouring signatures, so dense listings all embed alike and the
// retriever can't tell them apart. Every chunk starts with the package line,
// so a method still says which package it belongs to. A declaration longer
// than size is split, and each continuation repeats its first line.
// No chunk is longer than size.
func chunkGoDoc(text string, size, overlap int) []string {
	header, overview, units := parseGoDoc(text)
	budget := size - len(header) - 1 // what is left after "header\n"
	if budget < size/2 {
		return chunk(text, size, overlap)
	}

	var out []string
	add := func(piece string) { out = append(out, header+"\n"+piece) }
	for _, p := range chunk(overview, budget, overlap) {
		add(p)
	}
	for _, u := range units {
		if len(u) <= budget {
			add(u)
			continue
		}
		sig, _, _ := strings.Cut(u, "\n")
		for i, p := range chunk(u, max(budget-len(sig)-1, budget/2), overlap) {
			if i > 0 {
				p = sig + "\n" + p
			}
			add(p)
		}
	}
	return out
}
