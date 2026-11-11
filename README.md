# godocs-rag

A retrieval-augmented assistant for the Go standard library, written in Go on PostgreSQL + pgvector, **with a retrieval-quality eval that gates CI.**

Generic chatbots confidently invent Go APIs. This project answers from the actual package documentation, cites the file each answer came from, and refuses when the docs don't cover the question. Every retrieval change is measured against a hand-labelled question set before it can merge.

## Current results

Measured by `go run ./cmd/eval` over 202 in-scope and 30 out-of-scope questions (`eval/golden.jsonl`), 25 stdlib packages / 1,923 chunks, `nomic-embed-text`, k=5:

| Metric | All questions | Original 117 in-scope | Identifier (50) | Opaque identifier (35) |
| --- | ---: | ---: | ---: | ---: |
| Recall@1 | 0.584 | 0.402 | 0.900 | 0.743 |
| Recall@5 | 0.822 | 0.709 | 1.000 | 0.943 |
| MRR@5 | 0.680 | 0.524 | 0.950 | 0.815 |

| Metric | Value |
| --- | ---: |
| Out-of-scope abstention | 0.500 |
| Misses: right file / wrong file | 31 / 5 |
| Separation AUC (best score, in- vs out-of-scope) | 0.984 |
| Index matches exact search (top-5 scores) | 232 / 232 |
| Query latency p50 / p95 (embed + search, local Ollama) | 42 ms / 51 ms |

**The jump from 0.709 to 0.822 is not an improvement.** It comes from adding 85 identifier-style questions that vector search handles well; the original 117 give exactly the old numbers (the second column). The baseline was re-written for the larger set, so older numbers in this README (0.709, 147 questions) refer to the original set.

These are honest numbers for dense vector search over declaration-aware chunks, not a finished system. The eval shows *where* it still fails:

- **Mostly ranking.** 31 of the 36 misses retrieve the right file but rank the wrong passage higher. The other 5 retrieve a different file altogether (it was 1 before the chunking change, see below).
- **Identifier questions are nearly solved, but not at rank 1.** Dense search finds all 50 named-API questions in the top 5 and 33 of 35 opaque ones (`sql.ErrNoRows`, `slog.LevelWarn`), because every chunk starts with the declaration and the question repeats its name. The room left is ranking: Recall@1 is 0.900 and 0.743. That is the only place a keyword leg could still help, and it can show up in MRR and Recall@1, not Recall@5.
- **Near-domain questions.** Only 4 of the 18 questions about libraries and tools these packages don't document (errgroup, YAML, bcrypt, pprof) are refused; they score as close to the docs as real questions do.

That is what the next retrieval changes target (see [Roadmap](#roadmap)), and the gate makes each one prove itself.

### Change: declaration-aware chunking (shipped)

`go doc` output is split into one chunk per declaration (a function, method, type, or const/var block with its doc comment), and every chunk starts with the `package x // import "x"` line. Byte windows cut across neighbouring signatures, so dense listings all embed alike. Each variant was measured with a full re-index and no relevance floor, so only ranking differs, and compared question by question against the old byte windows:

| Chunking | Chunks | Recall@1 | Recall@5 | MRR | Separation AUC | vs byte windows: recall@5 flips (McNemar p) | MRR Δ, 95% bootstrap CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Byte windows (before) | 832 | 0.299 | 0.573 | 0.410 | 0.977 | | |
| Windows + package line | 863 | 0.368 | 0.684 | 0.468 | 0.966 | +18 −5 (0.011) | +0.058 [+0.002, +0.111] |
| Declarations, no package line | 1,907 | 0.325 | 0.573 | 0.412 | 0.963 | +20 −20 (1.00) | +0.003 [−0.084, +0.090] |
| **Declarations + package line (shipped)** | 1,923 | 0.410 | 0.718 | 0.532 | 0.972 | +25 −8 (0.005) | +0.123 [+0.036, +0.205] |

- **The package line does the work.** Splitting alone changes nothing on average (20 questions gained, 20 lost): a short chunk without its package is easy to confuse with a similar function from another package.
- **Splitting on top of the package line is not proven.** It adds +0.056 MRR over windows with the line, but not significantly (recall@5 flips +13 −10, p = 0.68; CI [−0.011, +0.126]). It stays because it scored higher on every metric and gives hybrid search and citations one declaration per chunk. A one-line "package line on every window" is the simpler alternative.
- **Recall@1 alone is weaker evidence:** +26 −13 against byte windows (p = 0.053). The gain is in recall@5 and MRR.
- **It costs something.** 8 questions got worse (25 got better), and wrong-file misses went from 1 to 4. Short chunks whose names echo the question (`MarshalBinary` for "binary", `Exit` for "program exit") now outrank the package overview or `Flush`. Chunks are also smaller (median 297 bytes, was 905), so each excerpt gives the model less context. The eval scores retrieval, not answers.
- The relevance floor stayed 0.55: on the dev half of the original index, recall was flat up to 0.55 and dropped at 0.575 (on the corrected index it drops at 0.55, see Relevance floor). The same run twice is identical (0 flips).
- **Re-measured after the index fix below.** The byte-window and shipped rows were run again on the corrected index with the current eval (the published version was +25 −9, p = 0.009, MRR +0.114). The two middle rows, the +13 −10 comparison above and the prefix experiment were made before the fix, each arm on its own index, and have not been re-run, so read their small differences as untested.

The two ablation rows are throwaway builds with one line of the chunker changed, not code in this repo. Reproduce the headline rows with `go run ./cmd/eval -min-score 0 -out a.json` on each side of the change, then `go run ./cmd/eval -compare a.json b.json`.

### Experiment: embedding task prefixes (no significant gain, not shipped)

`nomic-embed-text` was trained with `search_query: ` / `search_document: ` prefixes, and its model card says inputs must include them. Before testing them, the served model was checked against Nomic's reference implementation: Ollama's vectors match `nomic-embed-text-v1.5` (sentence-transformers) at cosine 1.00000 on 20 texts with and without prefixes, and 0.81–0.88 against v1. Every variant was then measured with a full re-index, with no relevance floor so they compete on ranking alone, and compared question by question against no prefix:

| Variant | Recall@1 | Recall@5 | MRR | Separation AUC | vs none: R@1 flips (McNemar p) | MRR Δ, 95% bootstrap CI |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| No prefix (run twice: identical) | 0.299 | 0.573 | 0.410 | 0.977 | | |
| Query only | 0.308 | 0.581 | 0.412 | 0.983 | +3 −2 (1.00) | +0.002 [−0.022, +0.026] |
| Document only | 0.333 | 0.598 | 0.424 | 0.957 | +8 −4 (0.39) | +0.014 [−0.025, +0.055] |
| **Both, as documented** (run twice: identical) | 0.342 | 0.564 | 0.421 | 0.978 | +8 −3 (0.23) | +0.012 [−0.027, +0.051] |
| Swapped (negative control) | 0.231 | 0.573 | 0.356 | 0.929 | +1 −9 (**0.02**) | **−0.053 [−0.093, −0.016]** |

The swapped control is significantly worse, so the eval can detect a prefix mistake. The documented prefixes lean better at rank 1 but no difference is significant, so the default stays unprefixed. The prefixes remain switchable (`EMBEDDING_QUERY_PREFIX`, `EMBEDDING_DOCUMENT_PREFIX`). Reproduce with `go run ./cmd/eval -min-score 0 -out a.json` per variant and `go run ./cmd/eval -compare a.json b.json`.

**Re-test after declaration-aware chunking** (same method, 1,923 chunks):

| Variant | Recall@1 | Recall@5 | MRR | Separation AUC | vs none: R@1 flips (McNemar p) | MRR Δ, 95% bootstrap CI |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| No prefix (run twice: identical) | 0.402 | 0.709 | 0.524 | 0.977 | | |
| Both, as documented | 0.410 | 0.684 | 0.508 | 0.977 | +5 −4 (1.00) | −0.016 [−0.052, +0.020] |
| Swapped (negative control) | 0.419 | 0.667 | 0.516 | 0.930 | +7 −5 (0.77) | −0.008 [−0.050, +0.035] |

Still no gain, so the default stays unprefixed. But the control is weaker evidence now: with short, package-labelled chunks the swapped prefixes are not significantly worse on recall (only the separation AUC drops, 0.977 to 0.930), so this test can no longer show that a wrong prefix hurts ranking.

### Change: the relevance floor gates the question (shipped, no metric change)

The floor used to drop hits one by one. It now looks only at the best hit: if that is below the floor the question counts as out of scope and nothing is returned, otherwise all top-k hits go to the model. Hybrid search will return keyword-only matches that score low on cosine but are still wanted, and a per-hit floor would delete them. Each row ran on the same index and was compared question by question with the per-hit rule at the same floor:

| Floor | Rule | Recall@5 | MRR | Abstention | Recall flips (McNemar p) | Abstention flips (p) |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 0.55 | per hit | 0.709 | 0.524 | 0.500 | | |
| 0.55 | per query | 0.709 | 0.524 | 0.500 | +0 −0 (1.00) | 0 refused, 0 answered (1.00) |
| 0.575 | per hit | 0.675 | 0.505 | 0.733 | | |
| 0.575 | per query | 0.692 | 0.511 | 0.733 | +2 −0 (0.50) | 0 refused, 0 answered (1.00) |

- **Nothing moves at the shipped floor.** At 0.55 both rules agree on all 147 questions, so this change leaves the metrics as they were. Where the dev rule lands is covered under Relevance floor below.
- **No recall gain is claimed.** The new rule can only keep more hits, never fewer. At 0.575 it recovers 2 of the 4 hits the per-hit floor lost, but two questions cannot be significant (p = 0.50).
- **It costs prompt size.** At 0.55, 23 weak excerpts (below the floor) now reach the prompt across 116 answered in-scope questions: 5.0 excerpts each, was 4.8. The eval scores retrieval, not answers, so the effect on answers is unmeasured.
- **The limit is overlap, not the rule.** In-scope best scores run from 0.547 (median 0.664); out-of-scope ones reach 0.622 (median 0.551). Between 0.547 and 0.622 no floor can separate them. A floor of 0.575 would trade 2 in-scope hits (p = 0.50) for 7 more refused out-of-scope questions with none going the other way (p = 0.016). That is a policy choice, so it is not shipped here.

Reproduce with `go run ./cmd/eval -out a.json` on a build before and after the change, then `go run ./cmd/eval -compare a.json b.json` (it now also pairs out-of-scope questions by abstention).

**Correction.** This section first reported abstention 0.533 at 0.55 and 0.833 at 0.575, and 9 more refusals (p = 0.004) for the 0.575 trade. Those were measured on an index that returned a worse nearest neighbour for 4 of 147 questions (the bcrypt question scored 0.501 instead of 0.592). The rows and bullets above are re-measured on a correct index, and the conclusions are unchanged. See the next section.

### Change: a wider HNSW search, and an index check in the report (shipped)

The same retrieval code gave recall@5 0.709 in one session and 0.718 in another. The cause was the index, not the model. Postgres chooses between a sequential scan (exact) and the HNSW index (approximate) by estimated cost, and a sequential scan looks dearer as the table fills with dead rows from re-ingests. At pgvector's default `ef_search` of 40, the index then returned a different top-5 than exact search for 7–10% of the questions. With the index forced on, across seven index states (fresh builds in shuffled order, churn, `REINDEX`), the number of questions out of 147 whose top-5 matched exact search was 132–136 at ef 40, 142–146 at 100, 146–147 at 200 and 147 in every state at 400, 800 and 1000. The store now runs every connection with `hnsw.ef_search` 800: twice the smallest value that matched everywhere, a rule written down before the measurement. It costs about 1.2 ms more per query on the index path (0.6–0.8 ms → 1.9–2.4 ms).

One database, deliberately churned (40 rounds of deleting and re-inserting 30% of the chunks), three builds, no floor:

| Build | Recall@5 | MRR | AUC | Recall@5 flips vs exact search (p) |
| --- | ---: | ---: | ---: | --- |
| Exact scan (ground truth) | 0.718 | 0.532 | 0.972 | |
| Before: `ef_search` 40 | 0.624 | 0.459 | 0.951 | +0 −11 (0.001) |
| **After: `ef_search` 800** | 0.709 | 0.528 | 0.972 | +0 −1 (1.00); 0 in a re-run |

At the shipped floor the old build reported abstention 0.667 on that database; the truth is 0.500. On a fresh index nothing changes: ef 40 → 800 gives 0 flips either way. The report now has a row, `Index matches exact search (top-5 scores)`, that compares the index with a sequential scan for every question and prints a warning under the table when they differ (ef 40 on the churned database: 131 of 147; the shipped build: 147 of 147).

What moved in the published numbers: abstention 0.533 → 0.500 (15 of 30 refused, 4 of the 18 near-domain questions), AUC 0.977 → 0.972, and `eval/baseline.json` was re-written (only abstention changed). The committed baseline failed its own gate on every fresh index. This was checked on one 1,923-chunk corpus; on a much larger one the planner uses the index, so re-run the check there.

### Experiment: hybrid full-text + vector search (no significant gain, not shipped)

Vector search can miss exact identifiers, so Postgres full-text search (a generated `tsvector` column with a GIN index) was fused with it by reciprocal rank fusion (RRF, k = 60, 40 candidates per list). The ship rule was written down before the first run: the variant with the best dev-half MRR ships only if, on all in-scope questions with the floor off, recall@5 flips are significant (McNemar p < 0.05) or the MRR interval excludes zero, and separation AUC does not drop. Every arm was compared question by question with vector-only search on the same accurate index:

| Arm | Recall@1 | Recall@5 | MRR | Recall@5 flips vs vector-only (p) | MRR Δ, 95% bootstrap CI |
| --- | ---: | ---: | ---: | --- | --- |
| Vector only | 0.410 | 0.718 | 0.532 | | |
| Full-text OR of the question's words (textbook form) | 0.291 | 0.607 | 0.405 | +6 −19 (0.015) | −0.127 [−0.194, −0.060] |
| **Full-text AND (`websearch_to_tsquery`)** | 0.393 | 0.761 | 0.539 | +6 −1 (0.125) | +0.007 [−0.038, +0.053] |
| Full-text OR, `simple` dictionary | 0.214 | 0.581 | 0.332 | +10 −26 (0.011) | −0.200 [−0.272, −0.128] |
| Full-text only, no vectors | 0.128 | 0.376 | 0.209 | +5 −45 (< 0.001) | −0.323 [−0.404, −0.241] |
| Control: AND with the previous question's words | 0.359 | 0.701 | 0.485 | +1 −3 (0.63) | −0.047 [−0.078, −0.022] |

- **The textbook form hurts.** OR-ing every word of a natural-language question is significantly worse on recall and MRR. A likely reason (not isolated): RRF gives both lists the same vote, so a noisy keyword list puts wrong chunks into the five slots.
- **AND is the only arm above vector-only, and it is not proven.** 6 questions gained, 1 lost, but p = 0.125 and the MRR interval includes zero; recall@1 is slightly lower. The gain sits in the test half (+5 −0, p = 0.0625); the dev half it was picked on shows +1 −1. Not shipped.
- **The control behaves.** The same AND query with another question's words gives no gain (+1 −3), so a gain needs relevant words. The OR form with another question's words drops recall@5 to 0.462.
- **Task 4's gate matters once keyword hits exist.** With the AND variant, a per-hit floor deletes hits the per-query rule keeps: +2 recall@5 at 0.55 (p = 0.50) and +4 at 0.575 (p = 0.125). Directional only.
- **The eval cannot settle it.** 117 in-scope questions cannot show +6 −1 at p < 0.05, and the golden set had almost no identifier-style questions, which are what hybrid targets. Those slices have since been added (see The eval). Vector-only already finds nearly all of them in the top 5, so a second attempt has to show its gain in ranking (Recall@1 and MRR). The variants were throwaway `go build -overlay` builds; the implementation is not in this repo.

## How it works

```text
 ingest:  corpus/*.txt ─▶ chunk (one per go doc declaration; other text ≈1000 B windows) ─▶ embed (batched) ─▶ ReplaceSource (one tx) ─▶ pgvector
 query:   question ─▶ rewrite follow-ups ─▶ embed ─▶ HNSW cosine top-k ─▶ refuse if the best hit < min score ─▶ LLM (streamed)
```

- **Store:** one Postgres table with an HNSW cosine index (searched with `hnsw.ef_search` 800, see above) and JSONB metadata. Re-ingesting a file replaces its chunks atomically: a failed embed or insert leaves the previous version intact.
- **Relevance floor:** the floor gates the question, not each hit. When even the best chunk scores below `RETRIEVAL_MIN_SCORE` (0.55), nothing is returned and the model is told to say the docs don't cover the question; otherwise all top-k chunks go to the model. 0.55 was the highest floor that lost no in-scope hit on the *dev* half (`-split dev -sweep`) on the original index; the other half is only used for reporting. On the corrected index that rule picks 0.525, because one dev question scores 0.5467. That would refuse 5 fewer out-of-scope questions (abstention 0.500 → 0.333) to keep one in-scope hit, so 0.55 stays until that is decided. On the test half 0.55 gives recall@5 0.656 and refuses 6 of 14 out-of-scope questions; on all questions it refuses 15 of 30 (11 of the 12 unrelated ones, 4 of the 18 near-domain ones). The gap is a threshold problem, not a ranking one: the best score ranks in-scope above out-of-scope questions 97.2% of the time (AUC), but a floor high enough to refuse more also drops real answers.
- **Providers:** any OpenAI-compatible endpoint (Ollama, OpenAI, LM Studio, vLLM). Chat and embeddings can point at different hosts.
- **Interfaces:** terminal REPL, web chat with SSE streaming at `/chat`, document and image upload, `/healthz` and `/readyz`.

### The eval

`cmd/eval` indexes `corpus/` into a throwaway database and runs every question in `eval/golden.jsonl` through the same `rag.Retriever` the app uses.

- A retrieved chunk is **relevant** when it comes from the labelled source file *and* contains the labelled phrase. Labelling by phrase instead of chunk ID keeps the set valid when chunking changes, and every label is checked against the corpus at startup, so a typo can't silently become a guaranteed miss.
- Out-of-scope questions have no source; they pass when retrieval returns nothing. Most of them are near-domain: Go questions about libraries and tools these 25 packages don't document, which are much harder to refuse than unrelated questions.
- The report splits misses into *right file* (a ranking problem) and *wrong file* (a coverage problem), so each change can target the one that dominates.
- **Identifier slices.** 85 questions carry a `kind` and are reported on their own row; `-compare` adds a paired McNemar and MRR interval per kind. `"identifier"` (50) names an API whose doc says what it does (`time.ParseDuration`, `sync.Map CompareAndSwap`, `strings.CutPrefix`); `"opaque"` (35) names an error, constant or terse function (`sql.ErrNoRows`, `http.StatusTeapot`, `slog.LevelWarn`). They are written the way a developer types a name, not copied from the doc wording. Names that exist in several packages (`strings.EqualFold`, `bytes.EqualFold`) spell out the package, so each label has one right file. The first slice was built to test hybrid search, and it was saturated (Recall@5 1.000), so the opaque one was added; both are in the normal run, not separate gates.
- CI fails if recall@1, recall@k, MRR or abstention falls below `eval/baseline.json` by more than the larger of 0.02 and two questions. Two questions is 0.067 of abstention (30 questions) but under 0.02 of recall (117+), so only abstention gets the wider margin. Improvements are accepted by re-running with `-write-baseline` and committing the new file.

`corpus/` is generated from the Go toolchain's own docs (`go run ./cmd/corpus`, which calls `go doc -all`) and committed, so the eval is reproducible offline.

## Quick start

Requirements: Go 1.26+, Docker, and [Ollama](https://ollama.com) (or any OpenAI-compatible provider).

```sh
cp .env.example .env            # set a local Postgres password in both places
ollama pull gemma3
ollama pull nomic-embed-text
docker compose up -d
cp corpus/*.txt documents/      # the watcher indexes and moves them to documents/processed/
go run ./cmd/rag
```

Then use the terminal prompt or open <http://127.0.0.1:8080/chat>.

### Running the eval

```sh
docker compose --profile eval up -d eval-db
go run ./cmd/eval                  # report + gate against eval/baseline.json
go run ./cmd/eval -skip-ingest     # reuse the indexed corpus while iterating
go run ./cmd/eval -sweep           # recall vs abstention for each min-score value
go run ./cmd/eval -write-baseline  # accept the current numbers
go run ./cmd/eval -split dev -sweep            # tune the floor on half the questions
go run ./cmd/eval -split test                  # report on the other half
go run ./cmd/eval -out a.json                  # also save per-question results
go run ./cmd/eval -compare a.json b.json       # flips, McNemar p, MRR bootstrap CI, AUC
```

The eval refuses to run on an index embedded with a different model or document prefix than the current settings (each chunk records its `embed_scheme`).

### Tests

```sh
go test -race ./...
# include the Postgres integration test:
TEST_DATABASE_URL=postgres://rag:eval@localhost:5433/rag_eval?sslmode=disable go test ./...
```

## Configuration

| Variable | Purpose |
| --- | --- |
| `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL` | Chat endpoint. Ollama: `http://localhost:11434/v1`, key `ollama`. |
| `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY`, `EMBEDDING_MODEL` | Embedding endpoint; defaults to the chat endpoint. |
| `EMBEDDING_DIM` | Must match the embedding model (768 for `nomic-embed-text`). Fixed into the schema on first run. |
| `EMBEDDING_QUERY_PREFIX`, `EMBEDDING_DOCUMENT_PREFIX` | Text prepended before embedding (default none; see the prefix experiment). |
| `RETRIEVAL_MIN_SCORE` | Similarity floor on the best retrieved chunk: below it the question counts as out of scope (default 0.55, 0 disables). |
| `DATABASE_URL`, `POSTGRES_PASSWORD` | App database; the compose file reads the password. |
| `EVAL_DATABASE_URL` | Eval database (truncated on every run). |
| `SYSTEM_PROMPT_FILE` | System prompt, default `prompts/system-custom.md`. |
| `INGEST_DIR`, `PROCESSED_DIR`, `IMAGES_DIR` | Watched folder, archive folder, uploaded images. |
| `HTTP_ADDR` | Web server address (default `127.0.0.1:8080`). |
| `VISION_MODEL` | Optional model for `/api/caption`. |

Changing the embedding model or dimension requires a fresh table: `docker compose down -v` (this deletes the local database volume) and re-ingest. Changing the document prefix only needs a re-ingest; the app logs a warning at startup when stored chunks were embedded differently.

## HTTP API

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/chat` | Browser chat UI. |
| `POST` | `/api/chat/stream` | `{"messages":[...]}` → SSE `delta` / `done` / `error` events. Only `user`/`assistant` roles are accepted. |
| `POST` | `/api/upload` | Ingest a `.txt` / `.md` file. |
| `POST` | `/api/upload/image` | Ingest an image by its description. |
| `POST` | `/api/caption` | Caption an image (when `VISION_MODEL` is set). |
| `GET` | `/images/{file}` | Uploaded images (no directory listing). |

## Project layout

| Path | Responsibility |
| --- | --- |
| `cmd/rag` | The app: REPL + web server + ingest watcher. |
| `cmd/eval` | Retrieval eval and CI gate. |
| `cmd/corpus` | Regenerates `corpus/` from `go doc`. |
| `app/` | Wiring and lifecycle. |
| `ingest/` | Chunking, embedding, directory watcher. |
| `rag/` | Query rewriting, retrieval, prompt context. |
| `vector/`, `vector/pgvector/` | Store interface and Postgres implementation. |
| `llm/` | OpenAI-compatible chat, embedding and vision clients. |
| `web/` | HTTP handlers, SSE, upload, injection filter, chat UI. |
| `eval/` | Golden question set and committed baseline. |

## Known limitations

- **The client owns the conversation history.** The server rejects client-sent `system` messages, but a client can still forge `assistant` turns. Server-side sessions would close this.
- **Uploads are unauthenticated**, and uploaded documents are retrieved into the prompt, so a malicious document can attempt indirect prompt injection. The regex filter in `web/injection.go` is a tripwire, not a security boundary. Bind to `127.0.0.1` (the default).
- Abstention is measured on 30 out-of-scope questions. In-scope and near-domain questions score too close together to separate cleanly: on an accurate index, moving the floor from 0.55 to 0.575 refuses 7 more of them (15 → 22 of 30) and loses 2 in-scope hits (recall@5 0.709 → 0.692).
- **The abstention gate has a wider margin, not a sharper one.** One out-of-scope question is worth 0.033 of abstention, and 7 of the 30 score within 0.01 of the 0.55 floor (0.543 to 0.558), so a small difference in embeddings (another Ollama build, a CPU instead of a GPU) can move several at once. The gate now allows two questions (0.067) instead of a flat 0.02, which absorbs that noise, but it also means a real loss of one or two refusals passes. A gate on the separation AUC, which does not depend on the floor, would be sharper; it is not done.
- Only `go doc` text is split on declarations. Other documents (uploads, notes) still use byte windows, and the overlap start isn't snapped to a word boundary.
- Declaration chunks are small (median 297 bytes, was 905), so each excerpt gives the model less surrounding context. The eval scores retrieval, not answers, and an answer-level eval is still missing.
- Conversation history is in memory only.

## Roadmap

Each step is judged by the eval, not by intuition:

1. **Decide the floor** (see Relevance floor). The gate now tolerates two questions of abstention noise; a gate on the separation AUC would be sharper.
2. **Hybrid retrieval, second attempt:** the identifier slices now exist (50 + 35 questions) and vector-only is already strong on them (Recall@5 1.000 and 0.943), so the test can only be about ranking: Recall@1 0.900 and 0.743, MRR 0.950 and 0.815. Next: retry the full-text AND variant with a weight on the keyword list under a ship rule fixed in advance, judged on those slices and on the original 117, so a gain on identifiers cannot hide a loss elsewhere (see the experiment above).
3. **Experiments:** other embedding models, and an LLM rerank of the top 20. Each ships only if `-compare` shows a significant gain.
4. **Verified citations:** structured sources in the SSE stream, checked against the chunks actually supplied.
5. **Server-side sessions, upload auth, rate limiting.**

## License

MIT. See [LICENSE](LICENSE). `corpus/` is derived from Go documentation © The Go Authors, BSD-style licence.
