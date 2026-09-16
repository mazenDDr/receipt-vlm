# Two bugs that looked like findings

*Fine-tuning and quantizing a 3B vision-language model to read receipts, and the two times the measurement
lied convincingly.*

The plan was ordinary. Take Qwen2.5-VL-3B, fine-tune it on CORD v2 so it turns a receipt photo into JSON,
then quantize it four ways and find where accuracy starts to hurt. The fine-tuning part went as expected:
field F1 on the held-out test split went from **0.440 to 0.836**, a paired improvement of
**+0.396 [+0.345, +0.442]**.

The interesting part was everything after that, because twice the pipeline produced a result that looked
like a discovery and was actually a bug in my own tooling. Both times the tell was the same: the result was
*too dramatic*.

## The first one: floods of exclamation marks

GGUF conversion for llama.cpp seemed to work. The bf16 GGUF — a lossless copy of the model, which should
score exactly what the original scores — came back at field F1 **0.540**, with **60% of receipts truncated**
at the generation cap. The truncated outputs were not garbled text. They were the character `!`, repeated
1024 times.

Four explanations were plausible, and I tested all four before believing any of them.

**Prompt cache reuse.** Every receipt shares the same text prompt and differs only in its image, so a server
reusing a cached prefix across requests could plausibly attach the wrong image. Disabling prompt caching at
both the request and the server changed nothing — the identical six receipts failed identically.

**Context exhaustion.** Roughly 1,400 tokens per receipt against a 4,096-token context could overflow after
a few requests. Raising the context to 8,192 and pinning the server to a single slot changed nothing; the
logs showed every request releasing with `truncated = 0` and plenty of room.

**Media type.** My code guessed `image/png` from the file extension. Every CORD image turned out to be a
genuine PNG with a matching extension.

**Position.** Requests 1–4 passed and 5 onward failed, which looks conclusive until you test it. Running the
failing receipts *first* produced a 100% failure rate. Not positional.

What survived: the failures followed specific receipts, the images were ingested correctly (prompt token
counts matched the other runtimes exactly, receipt for receipt), the server reported normal throughput, and
the output was token 0 repeated — which is what llama.cpp emits when the logits are NaN.

The vision projector was built at **f16**. f16 saturates at 65504. Everywhere else in this project the
vision tower runs in bf16, which has far more range at the same width. On certain images the projector
overflowed, the embeddings became NaN, and the model dutifully emitted token 0 until it hit the cap.
Rebuilt at f32, the same row scores **0.899** with zero truncation.

There is a deployment consequence hiding in that fix. The projector cannot be quantized — it has to stay
f32, where it is **2.49 GiB**. That is most of the smallest GGUF row's total of **3.67 GiB**, and it means
*no* GGUF bit width, not even 2-bit, undercuts the AWQ checkpoint's **3.31 GiB**. Lower precision did not
produce a smaller deployment.

## The second one: a cliff that wasn't there

With the projector fixed, the bit-width sweep looked clean and then fell off a cliff:

| | Field F1 (dev) | Valid JSON |
|---|---|---|
| bf16 | 0.901 | 1.000 |
| Q8_0 | 0.901 | 1.000 |
| Q6_K | 0.899 | 1.000 |
| Q5_K_M | 0.905 | 1.000 |
| Q4_K_M | 0.897 | 0.990 |
| Q3_K_M | **0.100** | 0.172 |
| Q2_K | **0.000** | 0.000 |

A sharp, clean threshold between 4-bit and 3-bit. I wrote it up that way — in a commit message and a pull
request description — and then went to check *why*.

Q2_K was producing a median of **three tokens** per receipt. Q3_K_M was producing coherent, well-formed
receipt JSON with real item names and plausible prices, and yet 82 of 99 outputs failed to parse. Those two
failure modes have nothing in common, which is not what a single underlying cause looks like.

llama.cpp's K-quants at Q3 and below normally expect an **importance matrix**: a calibration pass that
measures which weights matter before deciding where to spend the few bits available. My build script had
never produced one. Looking at the build script explained why — it compiled four llama.cpp targets, and
`llama-imatrix` was not among them. The tool that *consumes* a matrix was there; the tool that *makes* one
was never built.

With a matrix computed from the training receipts:

| | Without imatrix | With imatrix |
|---|---|---|
| Q3_K_M | 0.100 | **0.858** |
| Q2_K | 0.000 | **0.876** |

Q2_K — two-bit weights — came back to within noise of bf16 on dev, at **74.0 output tok/s against bf16's
40.5**. The cliff was mine, not the format's.

I had to retract the earlier claim publicly: the commit, the PR description and a comment on the PR. The
uncalibrated rows are still in the results tables, because the difference between the two is the actual
finding.

**One caveat, and it is the reason dev and test are separate.** That Q2_K tie held on dev and did *not*
survive the test split: −0.025 (not real) on dev became **−0.054 [−0.092, −0.020]** (real) on test. Dev
flattered it. The recommendation is Q4_K_M.

## What quantization actually costs

Down to 4 bits, nothing measurable. On test, AWQ W4A16 differs from its bf16 source by −0.005
[−0.026, +0.013] and GGUF Q4_K_M from its bf16 reference by −0.014 [−0.038, +0.009]. Neither is real, and
both are roughly half the size and much faster.

Below 4 bits, everything degrades together rather than one field type first. The going-in expectation —
that 4-bit would blur *digits* before free text — was wrong in an interesting way. The one place digits
stand out is at 2–3 bits, where numeric F1 drops measurably while overall F1 is still within noise.

The biggest lever was not precision at all:

| Same fine-tuned weights | Latency p50 |
|---|---|
| transformers | 13.22 s |
| llama.cpp, bf16 GGUF | 2.57 s |
| vLLM, merged bf16 | 2.07 s |
| vLLM, AWQ W4A16 | **1.05 s** |

The gap between serving stacks is larger than the entire spread across every bit width tested. Choosing the
runtime mattered more than choosing the precision.

## What the model actually gets wrong

Reading the 20 worst test receipts by hand was the most useful hour of the project. Every one of the 100
test receipts produced valid JSON — there are no format failures. What there is instead:

- **41%** exactly correct
- **16%** a digit difference somewhere
- **15%** differ *only* in separators or whitespace — `Rp.56.000` against `Rp. 56.000`, or `20,909` against
  `20.909`. Indonesian receipts use `.` and `,` interchangeably and the labels follow the printed glyph.
- **9%** a nesting error

The worst receipt in the split has **every value correct and every key wrong**: `itemsubtotal` written as
`discountprice`, the item code folded into the product name, and a `sub_total` invented out of a quantity
line. The second worst reads all eight items correctly and then nests six of them underneath item three.
One receipt scores text F1 of exactly 0.000 purely by swapping two keys, with every number right.

So this model does not have an OCR problem. It has a schema problem: it reads receipts well and misfiles
what it reads. That points somewhere specific — constrained or schema-aware decoding — rather than at more
training data, which is where I would have spent the effort without looking.

## The method, briefly

Every choice — prompt, image cap, LoRA rank, learning rate, checkpoint, bit width — was made on the dev
split. Test was scored once, for final variants only. Differences between variants are paired bootstrap
comparisons over the same receipts, and a difference counts as real only when its 95% interval excludes
zero. Most differences, including several I expected to matter, were not real.

Two habits did all the work. Compare within one runtime, so an engine difference cannot leak into a
quantization number. And when a result looks like a dramatic finding, suspect the pipeline first — twice
that instinct was right, and both times the bug was upstream of the model, in code I had written myself.
