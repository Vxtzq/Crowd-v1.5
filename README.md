# Crowd-v1.5

Pretraining by **DUST** — perturbing activations instead of backpropagating — with
a coordinator that accepts DUST and backprop submissions on **separate routes**.

This is the ground work for the v1.5 pivot: v1 trained a 658M+350M model with
backprop and shipped dense weight deltas. v1.5 trains a 1B dense model (no
engrams) and lets any device contribute, because DUST needs no backward pass, no
optimizer state and no gradient buffer. Backprop stays in the tree as a route
that is simply unused when DUST is active.

## Why the two halves are separable

A DUST client still uploads a **weight-shaped object**. The estimator is

```
c_s = (L(+a)_s - L(-a)_s) / 2                    antithetic centring
r_t = sum_{s>=t} gamma^(s-t) c_s                 temporal credit
G_W = (1/sigma) sum_t (r_t a_t) x_t^T            outer product with the input
```

and that last line needs the layer inputs, which only the client has. So the
coordinator cannot recompute a client's gradient from seeds; it can only check
that the submission is well formed, bound to the weights it claims, and
plausibly produced by the evidence it reports. That is what this repo enforces:

- `baseSha256` binds a gradient to the weights it was measured from. A gradient
  measured at one point in parameter space is not a gradient at another.
- `tokens`, `population`, `sigma`, `gamma` are the estimator's own record. A DUST
  submission with zero tokens or a non-positive sigma is refused before it can
  influence a step.
- one submission per client per round, one round at a time, no route mixing.

## Layout

```
crowd_v15/
  model.py        GPT: RMSNorm, fused-QKV attention, SwiGLU MLP, tied embeddings
  dust.py         estimator, optimizer, cosine probe against backprop
  tasks.py        identity / induction / copy / random data sources
  codec.py        NGD1 and NGD1R gradient containers (the v1 wire format, reused)
  protocol.py     header + body framing and every guard around it
  store.py        atomic weights, status.json, audit log
  coordinator.py  who counts, how they combine, when the model moves
  server/app.py   POST /submit/dust, POST /submit/backprop, model + status reads
  client.py       a DUST client: verify weights, train, upload, re-sync
tests/            89 tests, ~15 s on CPU
```

## Run it

```bash
pip install -e .[dev]
python -m pytest tests -q          # 89 passed

# coordinator
export CROWDGPT_ADMIN_SECRET=change-me
export CROWDGPT_V15_ROOT=./v15_state
python -m uvicorn --factory crowd_v15.server.app:create_app --port 8000

# install the weights it will serve (one-time; body is a safetensors file)
curl -X POST http://127.0.0.1:8000/admin/bootstrap \
     -H "X-Admin-Secret: change-me" --data-binary @weights.safetensors

# a client (CPU is fine)
python -m crowd_v15.client --server http://127.0.0.1:8000 \
       --preset tiny --task identity --rounds 2 --population 32
```

`--dry-run` trains and reports without uploading, which is how you check a port
before spending bandwidth.

## Routes

| route | who | accepted when |
|---|---|---|
| `POST /submit/dust` | a v1.5 client | route open, round current, `baseSha256` matches, tokens/population/sigma sane |
| `POST /submit/backprop` | a v3.5 client | route open, round current, `baseSha256` matches |

Both take the same container; the route in the URL and the route in the header
must agree, so a noisy estimate cannot be re-labelled as an exact gradient.

Read-only endpoints (`/status`, `/model/manifest`, `/model/weights`) are open:
a client needs the model before it can train, and the weights are public anyway.
Administration (`/admin/bootstrap`, `/admin/finalize`, `/admin/routes`) reads the
secret from `X-Admin-Secret` or `Authorization: Bearer`, never the query string.
With no secret configured, every admin route answers 503 — fail closed.

## Wire format

```
<u32 header length><header JSON><container>
```

The header is JSON (`schema`, `route`, `clientId`, `roundId`, `baseSha256`,
`codec`, `names`, `sizes`, `params`, plus the estimator fields). The container is
the v1 NGD family, unchanged:

| codec | frame | container | bits/value |
|---|---|---|---|
| `dense` | raw bf16 | — | 16 |
| `ngd1` | `NGD1` | `NGD2` | 1 + one fp16 scale per 1024 values |
| `ngd1r` | `NGD1R` | `NGD3` | 2 + two fp16 scales per 1024 values |

Measured on a 5,000-value random gradient: dense 10,255 B, `ngd1` 951 B (10.8×),
`ngd1r` 1,603 B (6.4×); sign agreement >98%, mean-absolute ratio `ngd1` 0.59 /
`ngd1r` 0.33 of the input scale. On a real tiny-model DUST gradient `ngd1r`
uploads ~8× smaller than dense.

Reusing the v1 container is deliberate: the quantisation error and the abuse
surface are already understood, and a v1.5 coordinator can read a v1 client's
delta with no new code path.

## What a step does

`Coordinator.finalize()` runs when the quorum (`min_submissions`) is reached:

1. average the pending gradients, weighting a DUST estimate by the tokens it
   reports (a backprop gradient gets unit weight — it is exact, not sampled);
2. optionally clip the global norm;
3. apply momentum, then `param -= lr * update`;
4. write the weights atomically, bump `round`, record history and the audit log.

A coordinator restart **drops** accepted-but-unapplied submissions rather than
applying a step nobody can audit: the round number has not moved, so clients
simply resubmit.

## Measured (CPU, tiny preset)

- `tests`: 89 passed in ~15 s (`protocol` 22, `coordinator` 18, `app` 17,
  `e2e` 4, `dust` estimator/training 28).
- DUST gradient vs backprop gradient on the same batch (cosine, seed 0):
  **0.89** at population 64 on 1-layer d32; **0.55** on a 2-layer d64 model with
  per-layer draws (`mode="layer"`); **0.02** on the same 2-layer model when every
  layer is jittered in one pass — interference control is the difference between
  a usable estimate and noise.
- One step at lr 1e-3 from the same weights: loss 0.991684 -> backprop
  0.982955, DUST (population 64) 0.983358, so the sign and scale are right to
  ~4e-4.
- End to end: two clients -> one round -> weights move -> new digest served ->
  the next round verifies against it.

## Honest limits

- The estimator is validated on tiny CPU models. Nothing here proves the
  paper's claim that larger models are *more* population-efficient at 1B; that
  needs the GPU box and the falsifiable gates (does a 1B-shaped step fit; does
  gradient alignment hold as size and tokens grow; how many draws to match
  backprop at fixed tokens — measured as time-to-target-loss, not FLOPs).
- The DUST path here draws each population member from a seed derived from
  `(seed, step, shape)` on a CPU generator, so a coordinator can in principle
  reproduce a member's noise. It does not yet re-score a submission; that is the
  audit primitive left to wire up (the v1 seed-derivation and signed-receipt work
  is the template).
- Weight serving is one safetensors file. A real 1B model needs ranged or sharded
  downloads; the manifest already reports `bytes`, `names`, `sizes` and `shapes`
  so a shard layer can be added without changing the client's checks.
- The task sources are identity/induction/copy/random. Identity is a learning
  test, not a pretraining corpus: real runs need real tokens (`--task copy
  --data chunks/chunk_0000.bin --data-dtype uint32`).
- No UI, no leaderboard, no reputation. The three-user verification pool of v1 is
  not reimplemented here.
