# AGENTS.md

Go RAG assistant over Go stdlib docs. Postgres + pgvector, any OpenAI-compatible LLM endpoint (Ollama locally). Module name: `godocs-rag`. Started from a Udemy course; the course scaffolding has been removed. Keep it that way: no "lesson N" comments, no tutorial essays in doc comments.

## Commands

```sh
go build ./... && go vet ./... && go test -race ./...
docker compose up -d                              # app DB on :5432 (creds in .env)
docker compose --profile eval up -d eval-db       # eval/test DB on :5433, rag/eval, tmpfs
TEST_DATABASE_URL=postgres://rag:eval@localhost:5433/rag_eval?sslmode=disable go test ./vector/pgvector/
EVAL_DATABASE_URL=postgres://rag:eval@localhost:5433/rag_eval?sslmode=disable go run ./cmd/eval [-skip-ingest|-sweep|-write-baseline|-split dev|test|-out f.json]
go run ./cmd/eval -compare a.json b.json         # paired: flips, McNemar, MRR bootstrap CI, AUC
go run ./cmd/corpus                               # regenerate corpus/ from `go doc -all`
go run ./cmd/rag                                  # app: REPL + web (127.0.0.1:8080/chat) + watcher
```

The full eval ingest takes about 3 minutes on local Ollama. Use `-skip-ingest` while iterating on query-side code.

## Layout and flow

- `cmd/rag` → `app.Run`: wires `llm` clients, `vector/pgvector` store, `ingest.Watch`, `web.Server`, `chat.RunREPL`.
- `ingest.index` is the single chunk → embed → `store.ReplaceSource` path for text and images. `go doc` text is split per declaration (`chunkGoDoc`, every chunk led by the package line); anything else uses the byte-window `chunk`.
- `rag.Retriever.Search` handles rewrite → `EmbedQuery` → `store.Query` → `MinScore` gate: if even the best hit is below it nothing is returned, otherwise every hit is. Documents go through `EmbedDocuments`; each chunk records `embed_scheme` (model + document prefix), and the eval refuses a mixed index. `rag.WithContext` prepends excerpts to the last user turn (shared by REPL and web).
- `vector.Store` has only `ReplaceSource`, `Query` and `Close`. Don't add methods nothing calls. `pgvector.Store.Reset` is eval-only.
- `cmd/eval` uses the same `rag.Retriever` the app uses. **The eval must test what ships.** Don't give it a separate retrieval path.

## Eval rules (the core of the project)

- `eval/golden.jsonl`: `{"q", "source", "contains"}`. A hit is a chunk from `source` that contains `contains` (case-insensitive). No source means out of scope, which passes when nothing is retrieved. Labels are validated against `corpus/` at startup.
- Write questions the way a developer would ask them. Don't copy doc wording, or the set becomes trivially easy.
- `eval/baseline.json` is the CI gate (tolerance 0.02). Re-baseline only for a deliberate change (new model, k, min_score, cases, or an improvement) and state the before/after numbers in the commit message.
- `config.DefaultMinScore` (0.55) was chosen with `-split dev -sweep` (highest floor losing no dev in-scope hit) and reported on `-split test`. If retrieval changes, re-sweep and update both the constant and the baseline.
- Claim an improvement only with `-compare` evidence (McNemar / bootstrap CI), never from one aggregate number.
- `corpus/` and `eval/` are pinned to LF (`.gitattributes`). CRLF would change chunk boundaries and therefore the numbers.

## Conventions

- Match the existing style: small packages, `fmt.Errorf("...: %w")`, context first, concise doc comments explaining *why*.
- Tests: one focused test per behaviour. DB tests skip unless `TEST_DATABASE_URL` is set; CI provides it.
- The worktree is CRLF on Windows (autocrlf), so a local `gofmt -l` lists every file. Check real formatting with `gofmt -d <file>`. CI checks out LF.
- Never commit `.env`. The user's local `.env` uses `DATABASE_PASSWORD`, while compose reads `POSTGRES_PASSWORD`.

## Security notes

- `/api/chat/stream` accepts only `user`/`assistant` roles (`web.validateHistory`). Clients can still forge assistant turns; server-side sessions are on the roadmap.
- Uploads are unauthenticated and retrieved text reaches the prompt, so indirect injection is possible. `web/injection.go` is a tripwire, not a boundary.

## Roadmap (next, in order)

Hybrid FTS + vector with RRF → embedding-model bake-off and LLM-rerank experiments (each ships only with `-compare` evidence) → verified structured citations → server-side sessions, upload auth and rate limiting. Each one must move the eval numbers.
