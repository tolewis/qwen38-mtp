# Intel Arc (Arc Pro B70) - contributor sweeps and studies

Contributor-authored deep dives for Qwen3.8-27B MTP on Intel Arc. Each section carries its author. Add yours with a PR to this file. The row and footnote go in the main [community table](../README.md#community-numbers).

### Intel Arc Pro B70 32GB, 1× and 2× (SYCL, Q4_0): the paired A/B behind the two rows
*by [@tolewis](https://github.com/tolewis)*

The table's first Intel rows. One headless Ubuntu 26.04 server (Linux 7.0, in-kernel `xe` driver, Intel Xeon 6 with 36 threads) holds two Arc Pro B70 32GB cards (BMG-G31, 608 GB/s). Each card has its own PCIe 5.0 x16 root port, with no bridge between the cards. The power limit is 275 W a card.

- llama.cpp `master` `9183a552e` (build 12374), SYCL backend: oneAPI DPC++ 2026.2 and oneMKL 2026.2, `-DGGML_SYCL=ON -DGGML_SYCL_F16=ON`. All `GGML_SYCL_*` runtime settings at their defaults. `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`.
- The rows add one DPC++ runtime setting to the server env of both arms: `SYCL_ENABLE_DEFAULT_CONTEXTS=0`. The table below shows both settings.
- unsloth `Qwen3.8-27B-Q4_0.gguf` (repo rev `4af51b3`, 16,056,478,688 B, sha256 `ede16c7b…4e671d`). The MTP head is inside the file (`nextn_predict_layers = 1`). No separate draft file.
- Both arms: `-c 131072 -ngl 999 -fa 1 --cache-type-k q4_0 --cache-type-v q4_0 --parallel 1`. The pair adds `-dev SYCL0,SYCL1 -sm tensor`. The flag arm adds `--spec-type draft-mtp --spec-draft-n-max N`.
- Method: unchanged `probe.py` at `a4c3028`. Three arms a row (spec off, n-max 3, n-max 5). Each arm has 3 server starts with one full `probe.py` pass a start, in a Latin-square order: (off, 3, 5), (3, 5, off), (5, off, 3). The value is the median of the 3 pass `OVERALL` medians.
- Two finals windows in the same order: first at the defaults, then with `SYCL_ENABLE_DEFAULT_CONTEXTS=0` on every arm. The pick rules were written before each window. In a window, the faster flag arm wins (n-max 3 on a tie under 1%). Between the windows, the faster flag arm wins (the defaults on a tie under 1%), if acceptance stays inside 0.02.
- Acceptance: server log, warm-up request dropped, accepted/generated over all requests of the arm, with the per-request range.
- Weights on the card (rule 7): `drm-resident-system` of the server process (xe `fdinfo`) is 0 after load and after the probe, in every set.

| row | SYCL contexts | arm | Overall | 3 passes | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance | Card W | VRAM after |
|---|---|---|---|---|---|---|---|---|---|---|
| 1× B70 | `=0` (row) | spec off | **27.8** | 27.8, 27.8, 27.7 | 27.7 | 27.8 | 27.7 | - | 269 | 17,996 MiB |
| 1× B70 | `=0` (row) | n-max 3 | **61.1** | 61.8, 61.1, 57.1 | 69.5 | 46.0 | 61.1 | 5388/7553 = 0.713 (0.40-0.92) | 267 | 19,669 MiB |
| 1× B70 | `=0` | n-max 5 | 55.0 | 56.4, 55.0, 51.9 | 75.5 | 37.7 | 55.0 | 6136/10950 = 0.560 (0.23-0.87) | 267 | 19,968 MiB |
| 1× B70 | default | spec off | 27.8 | 27.8, 27.8, 27.8 | 27.8 | 27.8 | 27.7 | - | 269 | 17,997 MiB |
| 1× B70 | default | n-max 3 | 59.7 | 60.1, 57.2, 59.7 | 68.7 | 43.9 | 59.7 | 5575/7897 = 0.706 (0.34-0.91) | 267 | 19,670 MiB |
| 1× B70 | default | n-max 5 | 57.5 | 57.5, 53.0, 58.2 | 75.5 | 39.6 | 57.5 | 5995/10510 = 0.570 (0.25-0.87) | 267 | 19,969 MiB |
| 2× B70 TP | `=0` (row) | spec off | **34.9** | 34.9, 35.0, 34.9 | 34.9 | 35.0 | 34.9 | - | 202 + 201 | 9,482 + 9,502 MiB |
| 2× B70 TP | `=0` (row) | n-max 3 | **71.6** | 65.4, 76.0, 71.6 | 85.2 | 53.1 | 71.6 | 5424/7591 = 0.715 (0.39-0.94) | 209 + 207 | 10,452 + 10,472 MiB |
| 2× B70 TP | `=0` | n-max 5 | 69.2 | 74.7, 61.4, 69.2 | 94.0 | 47.6 | 69.2 | 5562/10450 = 0.532 (0.27-0.88) | 216 + 215 | 10,602 + 10,622 MiB |
| 2× B70 TP | default | spec off | 32.2 | 32.2, 32.2, 32.3 | 32.3 | 32.3 | 32.1 | - | 195 + 194 | 9,482 + 9,503 MiB |
| 2× B70 TP | default | n-max 3 | 69.1 | 66.4, 69.1, 71.5 | 79.9 | 48.4 | 69.1 | 5713/8060 = 0.709 (0.36-0.93) | 204 + 202 | 10,452 + 10,473 MiB |
| 2× B70 TP | default | n-max 5 | 65.5 | 71.4, 65.5, 65.0 | 93.6 | 47.2 | 65.5 | 6083/10953 = 0.555 (0.28-0.85) | 213 + 211 | 10,602 + 10,622 MiB |

**The rows: 27.8 to 61.1 on one card (+120%, 2.20x) and 34.9 to 71.6 on the pair (+105%, 2.05x), both at n-max 3.** n-max 5 reads under n-max 3 on both configs in both windows. Code gains from the deeper draft (P1 75.5 against 69.5 on one card, 94.0 against 85.2 on the pair). Prose and bash lose. The spec-off arm is flat across starts (inside 0.1). A flag pass moves a lot between server starts: up to 10% on one card (n-max 5, 53.0 to 58.2) and up to 22% on the pair (n-max 5, 61.4 to 74.7). So take the median of several starts.

**`SYCL_ENABLE_DEFAULT_CONTEXTS=0` makes the pair 8% faster with the flag off (34.9 against 32.2, in every pass).** It is a DPC++ runtime setting, not a llama.cpp one. At 0, each card gets its own SYCL context. Then `-sm tensor` uses `ggml_sycl_comm_exchange` (four pinned host buffers and host waits). At the default (1), the cards share one context, and `-sm tensor` uses `ggml_sycl_comm_exchange_fast` (two buffers and queue barriers, llama.cpp PR #30956). So the default, which by the source selects the `_fast` exchange, reads 8% lower on decode. No log line shows which exchange ran. We did not profile the cause, and the setting can change more than the exchange. PR #30956 reports its gain on prompt eval. These rows measure decode only. With the flag, the pair gains 3.6% (71.6 against 69.1), inside its pass spread. On one card the setting does not change the spec-off arm (27.8 both). Its n-max 3 gain (61.1 against 59.7) is inside the pass spread too. The written rule picked the `=0` window for both rows, so both rows use it. Acceptance did not drop (0.713 and 0.715, against 0.706 and 0.709 at the default). `GGML_SYCL_ENABLE_GRAPH=1` gives no gain on the pair (32.2 spec off, 70.7 at n-max 3 in one pass).

**The second card adds 26% with the flag off and 17% with it on (34.9 against 27.8, 71.6 against 61.1).** One card draws 267-269 W of its 275 W limit. At `-sm tensor` each card draws 194-216 W. We did not profile the pair. The power numbers fit a pair that waits on its all-reduce: every partial sum goes through pinned host memory. On the same pair another engine reads 56.7 with no drafter (see the TensorFold section). So the cards have more speed than the current SYCL split uses.

### Backend: SYCL against Vulkan on one B70
*by [@tolewis](https://github.com/tolewis)*

Single-run screens (n=1) unless the row says n=3. Same flags as above, one card. SYCL rows at the default SYCL contexts.

| backend | host | quant | spec off | n-max 3 | n-max 5 |
|---|---|---|---|---|---|
| SYCL (oneAPI 2026.2) | server | UD-Q4_K_M | 23.7 | 47.7 | 50.0 (n=3) |
| SYCL (oneAPI 2026.2) | server | Q4_0 | 27.8 (n=3) | 59.7 (n=3) | 57.5 (n=3) |
| Vulkan (Mesa 26.0.8, Ubuntu) | server | UD-Q4_K_M | 15.9 | 23.1 | - |
| Vulkan (Mesa 26.2.5, Arch) | desktop | UD-Q4_K_M | 22.1 | 41.1 | 37.8 |
| Vulkan (Mesa 26.2.5, Arch) | desktop | Q4_0 | 25.0 | 54.1, 51.5 (two starts) | 49.4 |

**On the same server, SYCL leads Vulkan on Mesa 26.0.8 (the Mesa that Ubuntu 26.04 ships) on both arms.** Vulkan reads 15.9 spec off, 33% under SYCL, and 23.1 at n-max 3, 52% under SYCL. Mesa 26.2.5 closes most of that gap. We ran Mesa 26.2.5 only on a second machine: a desktop workstation (Arch-based, Linux 7.2, Xeon Platinum 8168, one B70 on PCIe 3.0 x16 that also drives the desktop). So SYCL against Mesa 26.2.5 is a screen across two hosts, not a paired result. Weights stayed on the card on the desktop too (`drm-resident-system` 0). This ranking holds for these hosts, drivers and builds only. Other Mesa, oneAPI or llama.cpp versions can rank the two backends another way. If you run a B70 on Vulkan, use a recent Mesa.

`-sm row` does not load on SYCL: `pre-allocated tensor (blk.0.attn_norm.weight) in a buffer (SYCL_Split) that cannot run the operation`.

### Quant: Q4_0 against UD-Q4_K_M on SYCL
*by [@tolewis](https://github.com/tolewis)*

| config | quant | spec off | n-max 5 | Acceptance at n-max 5 |
|---|---|---|---|---|
| 1× B70 | UD-Q4_K_M (16.56 GB) | 23.7 | 50.0 (n=3) | 0.585 |
| 1× B70 | Q4_0 (16.06 GB) | 27.8 (n=3) | 57.5 (n=3) | 0.570 |
| 2× B70, `-sm tensor` | UD-Q4_K_M | 30.3 | 62.7 (n=3) | 0.558 |
| 2× B70, `-sm tensor` | Q4_0 | 32.2 (n=3) | 65.5 (n=3) | 0.555 |

The Q4_0 values are the default-contexts finals above. The UD-Q4_K_M values come from earlier windows on the same box. The UD-Q4_K_M file is the upload at repo rev `4af51b3` (16,564,259,744 B, sha256 `1bce504d…7dc529`), a newer upload than the file of the 2× RTX 5090 row (16,464,440,224 B). **On SYCL, Q4_0 is 17% faster than UD-Q4_K_M with the flag off and 15% faster at n-max 5 on one card.** On the pair, Q4_0 is 6% faster with the flag off. At n-max 5 the pair gap (65.5 against 62.7) is inside the pair's spread: an earlier Q4_0 screen of 3 passes read 70.5. The file is only 3% smaller, so the size does not explain the gain. We did not profile the cause. Acceptance is the same for the two quants. That is why the rows use Q4_0. If you run a B70 on SYCL and quality allows it, try Q4_0 first.

### One B70: n-max, p-min, KV type and the SYCL graph (UD-Q4_K_M)
*by [@tolewis](https://github.com/tolewis)*

| arm | runs | Overall | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance | VRAM after |
|---|---|---|---|---|---|---|---|
| spec off | 1 | 23.7 | 23.7 | 23.8 | 23.7 | - | 18396 MiB |
| n-max 2 | 1 | 42.2 | 47.8 | 34.1 | 42.2 | 0.785 (0.52-0.97) | 19833 MiB |
| n-max 3 | 1 | 47.7 | 57.2 | 35.6 | 47.7 | 0.739 (0.46-0.97) | 19982 MiB |
| n-max 4 | 3 | 49.1 | 60.2 | 32.7 | 49.1 | 0.641 (0.32-0.93) | 20132 MiB |
| n-max 5 | 3 | 50.0 | 61.9 | 30.6 | 50.0 | 0.585 (0.27-0.85) | 20281 MiB |
| n-max 6 | 1 | 47.3 | 67.2 | 28.2 | 47.3 | 0.521 (0.25-0.82) | 20431 MiB |
| n-max 4, `--spec-draft-p-min 0.60` | 1 | 49.8 | 59.9 | 34.8 | 49.8 | 0.790 (0.50-0.91) | 20132 MiB |
| n-max 4, `--spec-draft-p-min 0.75` | 1 | 47.1 | 57.5 | 33.5 | 47.1 | 0.860 (0.62-0.94) | 20132 MiB |
| n-max 4, q8_0 K/V | 1 | 51.0 | 61.2 | 34.4 | 51.0 | 0.669 (0.38-0.90) | 22180 MiB |
| n-max 4, f16 K/V | 1 | 48.6 | 57.1 | 30.5 | 48.6 | 0.637 (0.26-0.91) | 25531 MiB |
| spec off, `GGML_SYCL_ENABLE_GRAPH=1` | 1 | 23.9 | 23.9 | 24.1 | 23.9 | - | 18422 MiB |
| n-max 5, `GGML_SYCL_ENABLE_GRAPH=1` | 3 | 47.2 | 60.8 | 30.8 | 47.2 | 0.575 (0.25-0.85) | 20301 MiB |

**n-max 4 and 5 tie (49.1 and 50.0 over 3 runs, each inside the other's run spread).** Code gains from a deeper draft (P1 rises to 67.2 at n-max 6), and prose loses (P2 falls from 35.6 at n-max 3 to 28.2 at n-max 6). So the best depth depends on the workload, as on the other cards. The p-min gate raises acceptance and does not raise speed here, which fits rule 2: one B70 is not starved for this model. q8_0 and f16 K/V cost 2 GB and 5.4 GB more VRAM and give no clear gain at n=1. The SYCL graph (`GGML_SYCL_ENABLE_GRAPH=1`, built in) gives no gain. One card runs at 264-272 W against its 275 W limit.

### 2× B70: split mode and n-max (UD-Q4_K_M)
*by [@tolewis](https://github.com/tolewis)*

Split-mode screen first (rule 4), then the spec sweep on the faster split. Card power is per card.

| split | arm | runs | Overall | P1 code (py) | P2 prose (mmap) | P3 code (bash) | Acceptance | Card W |
|---|---|---|---|---|---|---|---|---|
| layer | spec off | 1 | 23.7 | 23.7 | 23.8 | 23.7 | - | 168 + 177 |
| tensor | spec off | 1 | 30.3 | 30.4 | 30.3 | 30.1 | - | 214 + 213 |
| row | spec off | - | does not load | | | | | |
| layer | n-max 3 | 1 | 46.1 | 54.8 | 32.7 | 46.1 | 0.724 (0.40-0.95) | 148 + 175 |
| tensor | n-max 2 | 1 | 56.0 | 61.3 | 44.8 | 56.0 | 0.813 (0.54-0.96) | 195 + 192 |
| tensor | n-max 3 | 3 | 62.2 | 71.1 | 45.0 | 62.2 | 0.711 (0.35-0.94) | 203 + 202 |
| tensor | n-max 4 | 1 | 62.7 | 83.2 | 43.4 | 62.7 | 0.640 (0.33-0.93) | 205 + 204 |
| tensor | n-max 5 | 3 | 62.7 | 83.7 | 40.9 | 62.7 | 0.558 (0.22-0.87) | 213 + 211 |
| tensor | n-max 6 | 1 | 59.5 | 87.6 | 36.9 | 59.5 | 0.479 (0.21-0.92) | 214 + 213 |
| tensor | n-max 5, `--spec-draft-p-min 0.60` | 1 | 58.9 | 83.5 | 37.7 | 58.9 | 0.760 (0.52-0.93) | 192 + 192 |
| tensor | n-max 5, q8_0 K/V | 1 | 64.0 | 82.7 | 39.9 | 64.0 | 0.570 (0.26-0.85) | 210 + 209 |
| tensor | n-max 5, `GGML_SYCL_ENABLE_GRAPH=1` | 1 | 69.3 | 83.9 | 43.1 | 69.3 | 0.610 (0.30-0.84) | 209 + 208 |

**`-sm tensor` is the split for two B70s: 28% over `-sm layer` with the flag off.** At `-sm layer` the pair reads the same as one card (23.7). Each card holds about 10.5 GB at `-sm tensor`, and each card draws about 200-215 W, under its limit. The pair is noisy: three n-max 5 passes read 58.3, 67.1 and 62.7, so an n=1 screen on the pair can move about 15%. n-max 3 to 5 tie on the pair. The `GGML_SYCL_ENABLE_GRAPH=1` pass (69.3) is one pass. On Q4_0 the graph gave no gain (first section), so we read 69.3 as pass spread.

### TensorFold on the same cards (another engine, not a table row)
*by [@tolewis](https://github.com/tolewis)*

The same cards and the same `probe.py` prompts, on another engine. [TensorFold](https://github.com/tolewis/TensorFold) is an inference engine for Intel GPUs on Level Zero. It runs the pair at tensor parallel 2. Commits: [`73efe7441`](https://github.com/tolewis/TensorFold/commit/73efe7441) (pair) and [`c664084e3`](https://github.com/tolewis/TensorFold/commit/c664084e3) (one card, on the desktop workstation). What differs from the llama.cpp rows:

- Weights: `mlx-community/Qwen3.8-27B-4bit` (MLX affine 4-bit, group size 64, rev `b935172`), not a GGUF.
- K/V cache: int8, not q4_0. Context: 131,072 on the pair, 217,088 on the one-card desktop.
- Drafter: a DFlash2 block-diffusion tree drafter (a separate small model, 15 drafted tree nodes a round), not the MTP head.
- Both arms at `--parallel 1` equivalent (one request at a time). Only the drafter changes between the arms. Thinking off. 3 server starts an arm, interleaved, one full pass a start.

**Count fault and fix.** TensorFold streams one SSE event a drafter round. A round holds about 5 tokens, so unchanged `probe.py` (it counts events) reads the drafter arm about 5x low. TensorFold has no option to stream one event a token. So the table shows three readings:

1. Unchanged `probe.py` at `a4c3028`.
2. `probe.py` with one change: the request adds `"stream_options": {"include_usage": true}`, and the count is `usage.completion_tokens` of the reply in place of the event count. The clock (first event to last event), the prompts, the runs and the warm-up are unchanged. The tokens of the first event are taken out (by their share of the characters), because the clock starts at that event.
3. The server's own count: (tokens - 1) / decode seconds, from its log, median of the 9 timed requests.

| config | drafter | 1. unchanged `probe.py` (events/s) | 2. `probe.py`, usage tokens | 3. server log | P1 code (py) | P2 prose (mmap) | P3 code (bash) | tokens a round | card W |
|---|---|---|---|---|---|---|---|---|---|
| 1× B70 (desktop) | off | 32.3 | 32.3 | 32.3 | 32.2 | 32.5 | 32.2 | 1.0 | 202-207 |
| 1× B70 (desktop) | on | 23.2 | 123.6 | 123.5 | 163.1 | 82.2 | 123.6 | 5.4 | 211-213 |
| 2× B70 TP (server) | off | 56.7 | 56.7 | 56.7 | 56.6 | 57.0 | 56.6 | 1.0 | 202-204 a card |
| 2× B70 TP (server) | on | 42.3 | 213.6 | 213.7 | 302.6 | 148.2 | 213.6 | 5.1 | 211-213 a card |

P1, P2 and P3 show reading 2. The two token counts (2 and 3) agree within 0.1% in every pass. With the flag off, all three readings agree, because each event holds one token. TensorFold's own acceptance (accepted over drafted tree nodes, about 0.29) does not compare with the MTP acceptance of the table, so the table shows tokens a round. One drafter start can read lower than the others (one-card passes 123.9, 110.4 and 123.6), because a new start changes the reply text.

**What this says about the B70, not about the engines:** the pair has more speed than the llama.cpp rows use. Spec off, TensorFold reads 56.7 on the pair against 34.9 for the llama.cpp row at `-sm tensor` (other weights, so this is not a paired result).

### Caveats
*by [@tolewis](https://github.com/tolewis)*

- `OVERALL median` is the middle request of 9. In 23 of the 24 finals flag passes (one read 71.5 against a P3 of 72.8), it equals the P3 (bash) median. So a flag value is in practice a P3 value. The per-prompt columns show the rest: prose (P2) gains least.
- Screens marked n=1 are single passes. A flag pass moves up to 10% between server starts on one card and up to 22% on the pair.
- Thinking is off in `probe.py` (`enable_thinking: false`) and in every arm.
- The cards ran at a 275 W power limit. One card on SYCL draws 264-272 W during decode.
- VRAM is the server process's `drm-resident-vram0` from xe `fdinfo`, after the probe.
