# RAG vs. LoRA Fine-Tuning for Course-Specific Question Answering

**A controlled comparison of retrieval and parameter-efficient fine-tuning, on one corpus, one base model, one test set.**

**Tamar Telele** · Applied Language Models · Dr. Barak Or · Google &amp; Reichman University Tech School

---

## TL;DR

Three systems were built on **the same frozen `flan-t5-base` checkpoint** — RAG, LoRA, and their combination — so that every measured difference is attributable to *method*, not to model capacity.

| System | ROUGE-L | BLEU | Hallucination | Latency |
|---|---|---|---|---|
| Baseline (zero-shot) | 0.168 | 0.015 | — | 0.67 s |
| RAG | 0.389 | 0.204 | 11.9% | 0.75 s |
| LoRA | 0.273 | 0.057 | 14.3% | 1.10 s |
| **Hybrid (RAG + LoRA)** | **0.444** | **0.238** | **11.9%** | 1.05 s |

**Hybrid wins every quality metric** — 2.6× ROUGE-L and 15.9× BLEU over the zero-shot floor — and matches RAG's hallucination rate while doing it.

Three headline findings:

1. **Retrieval and fine-tuning are complementary, not competing.** Hybrid *exceeds* both components rather than matching the better of the two, which is only possible if each contributes different signal.
2. **The reader is the ceiling, not the retriever.** Three independent measurements agree: a 0.065 hit/miss gap, 63–83% generation error, and near-zero retrieval error.
3. **Hallucination and low lexical overlap are independent failure modes.** NLI-measured hallucination rates are flat across ROUGE-L quartiles, so a single quality number cannot substitute for both.

---

## Links

| Resource | Link |
|---|---|
| Notebook 1 — Data Preparation | https://colab.research.google.com/drive/1gK3xTrbMvp9UjOBxu6PC_xHMvOd1lFH7?usp=sharing |
| Notebook 2 — RAG · LoRA · Evaluation · Demo | https://colab.research.google.com/drive/17f-2m4YDLkQyxtndQ7LtkN6HvigNlOwJ?usp=sharing |
| Report| [`RAG_vs_LoRA_Report_EN_Tamar_Telele.pdf`](./RAG_vs_LoRA_Report_EN_Tamar_Telele.pdf) |
| Slides| `Presentation_EN_*.pptx` · `Presentation_HE_*.pptx` |

---

## Why this problem

Adapting a general language model to a narrow domain admits two architecturally distinct answers:

- **RAG** leaves the weights untouched and injects evidence into the prompt at inference. Knowledge lives *outside* the model and updates by re-indexing.
- **LoRA** writes knowledge *into* the weights through low-rank adapters. Nothing is retrieved at inference, but knowledge freezes at training time.

They have opposite update costs, opposite latency profiles, and — as this project shows — opposite failure modes.

Published comparisons rarely isolate the variable, because they typically differ in base model, corpus and evaluation protocol simultaneously. This project removes those confounds: **one corpus, one base checkpoint, one test set opened exactly once.**

A third condition, **Hybrid** (adapters *plus* retrieval), turns the question into a falsifiable prediction: if the two mechanisms capture the same signal, Hybrid should match the better of them; if different signal, it should exceed both.

---

## Dataset

Built end-to-end from **9 course lecture PDFs** (≈500 slides). No external corpus.

### Pipeline

```
9 PDFs → PyMuPDF (page-level, provenance kept)
       → 9-step deterministic cleaning
       → Gemini QA generation (strict grounding, JSON output, typed mix)
       → Stage 1: heuristic filter
       → Stage 2: LLM-as-a-Judge
       → Stage 3: human audit + ChatGPT deletion/correction lists
       → 1,130 QA pairs → stratified 70/15/15 split
```

### Three-stage quality protocol

| Stage | Mechanism | Criterion |
|---|---|---|
| 1 — Heuristic | Deterministic rules | Length bounds; degenerate answers (`yes`, `n/a`); leakage phrases (*"according to the slide"*); exact duplicates; near-duplicates at Jaccard ≥ 0.85 |
| 2 — LLM Judge | Gemini @ temp 0.0 | Four criteria scored 1–5 — groundedness, correctness, clarity, self-containment. **Accept iff mean ≥ 4.0 AND groundedness ≥ 4, checked separately** |
| 3 — Human + GPT | Seeded 10% audit | Random reproducible sample shown with machine scores; ChatGPT produced separate deletion and correction lists, applied programmatically |

**Judge acceptance rate: 89.1% (1,140 / 1,279).** After Stage 3: **1,130 pairs.**

Two design decisions carry real weight here:

- **The judge setup is asymmetric on purpose.** The generator grading itself sounds circular, so the judge sees the *source context* and is asked a verification question ("is this answer supported by this text?") — a strictly easier and better-calibrated task than generation.
- **Groundedness is checked separately, not folded into the mean.** A pair could average 4.2 on clarity and self-containment while scoring 3 on groundedness — fluent, well-formed, and not actually supported by the slides. Such a pair would corrupt the RAG index *and* poison LoRA's training signal simultaneously. The conjunctive rule makes that unreachable.

### Composition

- **54.4% factual · 34.8% conceptual · 10.8% comparative** — a deliberate mix. A dataset of only *"what is X?"* cannot distinguish RAG from LoRA, because both handle definitions well. Divergence appears on conceptual and comparative items.
- Questions average 11 words, answers 17 — short and grounded, with no multi-paragraph answers that reward verbosity over accuracy.
- Split **792 / 170 / 168**, stratified by **source PDF, not topic**: the LLM's free-text topic labels produced 500+ distinct values, so stratifying on them would be random assignment wearing a stratification label. Deduplication runs *before* splitting to prevent leakage.

---

## Part A — RAG pipeline

### Chunking: measured, not assumed

Three granularities compared on validation MRR, with `RecursiveCharacterTextSplitter` and its separator hierarchy **held fixed** — a granularity ablation, not an algorithm comparison.

| Strategy | Size | Overlap | Chunks produced | Realised avg |
|---|---|---|---|---|
| sentence | 50 tok | 10 | 1,407 | 39 tok |
| paragraph | 300 tok | 50 | 208 | 283 tok |
| slide | 500 tok | 75 | 125 | 472 tok |

Two implementation details carry the rest of the project:

- The splitter is built **from the flan-t5 tokenizer**, so 50/300/500 are true token counts rather than a character heuristic that would drift silently per document.
- `add_start_index=True` maps every chunk back to its **source PDF and page range** — the provenance that makes both retrieval evaluation and error analysis possible with **no manual labelling at all**.

### Embedding and index

`all-MiniLM-L6-v2` (384-d, 22M params), **L2-normalised**, stored in **FAISS `IndexFlatIP`**.

Inner product on normalised vectors *is* cosine similarity — mathematically identical to `IndexFlatL2`, but IP returns the bounded `[-1, 1]` score that the abstention guardrail consumes directly. `Flat` means exact exhaustive search: milliseconds at this scale, and zero approximation error in the experiment.

**t-SNE shows the space is semantic, not lexical.** Computer Vision Parts 1 and 2 share a region with GANs adjacent — visual generative models share architecture and loss vocabulary with CV. Deep RL separates cleanly, reflecting how distinct *reward, policy, Bellman* are from the rest of the curriculum. General ML topics overlap centrally, reflecting shared foundations like gradient descent.

### Three retrievers

| Method | Strength | Weakness |
|---|---|---|
| **Dense** | Meaning — retrieves a LoRA chunk for a question about parameter count with near-zero vocabulary overlap | Technical tokens (`ROUGE-L`, `flan-t5-base`) are averaged into one vector and lose discriminative power |
| **BM25** | Exact acronyms, model names, metric identifiers | Vocabulary mismatch between query and document |
| **Hybrid** | `α·s̃_dense + (1−α)·s̃_BM25` | — |

`s̃` is min-max normalisation **per query, not per corpus**. That detail is load-bearing: BM25 scores are unbounded and corpus-dependent, so without per-query normalisation BM25 would silently dominate the weighted sum and α would stop meaning what it appears to mean.

One diagnostic question makes the case for fusion concrete. For *"What are two examples of supervised learning provided in the course material?"* — Dense returned 2 relevant chunks of 3, BM25 returned 1, Hybrid returned 2. Dense won because the question is conceptual; BM25 failed because *learning* is shared with unrelated chunks. The point isn't that Dense is better — it's that **no single method wins consistently**.

### Grid search — and the sentence-level surprise

Relevance ground truth comes free from provenance: a chunk is relevant if its source file matches and its page range overlaps within ±1 page.

**MRR was chosen over Hit@3 as the selection metric.** Each pair has exactly one ground-truth source. A retriever surfacing it at rank 3 still scores Hit@3 = 1 — but the reader then receives two irrelevant chunks *before* the correct one, and that context pollution degrades generation. MRR penalises it directly (rank 3 → 0.33), aligning retrieval quality with end-to-end answer quality.

| Configuration | MRR | Hit@1 | Hit@3 |
|---|---|---|---|
| **sentence + Hybrid** | **0.706** | 0.565 | 0.847 |
| sentence + Dense | 0.687 | 0.547 | 0.812 |
| sentence + BM25 | 0.657 | 0.529 | 0.771 |
| paragraph + Hybrid | 0.670 | 0.506 | 0.835 |
| paragraph + Dense | 0.617 | 0.459 | 0.741 |
| slide + Hybrid | 0.620 | 0.494 | 0.724 |
| slide + Dense | 0.524 | 0.394 | 0.606 |

Two clean results: **Hybrid wins on all three chunk strategies**, and **sentence-level wins on all three retrieval methods**.

The second contradicts the t-SNE intuition, and the explanation is that the two measurements answer different questions. t-SNE asks whether a chunk is *topically identifiable* — a 300-token paragraph obviously is. Retrieval asks whether a chunk *answers a specific question*, and there a 50-token window holding one fact with no surrounding noise is a far better match. **Granularity must be measured against the actual objective, not inferred from cluster quality.**

**α sweep:** MRR peaks at **α = 0.7** → final config **sentence + Hybrid, α = 0.7, MRR 0.717, Hit@1 0.576, Hit@3 0.841**. The retriever leans on dense embeddings with BM25 as a lightweight corrector — the right shape for a corpus whose lecture language is internally consistent.

### Abstention guardrail

A study assistant that confidently invents an answer to an off-syllabus question is worse than useless.

The threshold was **calibrated, not guessed**: top-1 similarity was computed for in-scope validation questions and for 12 deliberately out-of-scope probes (cooking, football, astronomy), then thresholds were swept and the one maximising **balanced accuracy** kept — the correct objective when both error types are real and the classes are artificially balanced.

**Threshold 0.330 → 100% balanced accuracy.** In-scope p05 (0.462) sits far above out-of-scope p95 (0.294) — a 0.168 gap — and accuracy peaks across a wide band (≈0.30–0.35) rather than one fragile point.

### Reader and prompt

The reader is the **untouched** `flan-t5-base` — the same checkpoint Part B fine-tunes, which is what keeps the Part C comparison clean.

- Beam search (`num_beams=4`, `do_sample=False`) for reproducibility
- `min_new_tokens=15` eliminates degenerate one-token outputs that occur when EOS is the locally optimal first token
- Three prompt variants compared on validation by BLEU; **v1** (role framing + task-first ordering) won
- Direct chunk context beat Small-to-Big expansion: at 39 tokens per chunk the top-3 context already fits the 512-token budget, so expansion adds noise without headroom

---

## Part B — LoRA fine-tuning

`target_modules=["q","v"]` only. Hu et al. (2021) showed adapting Q and V alone matches full attention fine-tuning, while adding K and O costs parameters for negligible gain. Fixing this also keeps the coordinate search clean — one factor moves at a time.

### Coordinate search: 15 runs instead of 54

A full grid over 3 tokenizations × 3 ranks × 3 alphas × 2 dropouts is 54 runs. Coordinate search costs 11 — but it is blind to interactions, and there is exactly one place an interaction is genuinely expected: the adapter contributes `(α/r)·BA`, so **α and r are not independent**.

Four grid-completion runs were therefore added so **every cell of the r×α heatmap is measured rather than interpolated**. Total: **15 runs, a 3.6× reduction**, with the one interaction that matters measured explicitly.

| Stage | Varies | Selected on | Runs | Winner |
|---|---|---|---|---|
| A | word / subword / char | val ROUGE-L | 3 | **word** (0.2530) |
| B | r ∈ {4, 8, 16} | val ROUGE-L | 3 | **r = 8** |
| C | α ∈ {8, 16, 32} + grid | val ROUGE-L | 3+4 | **α = 32** |
| D | dropout ∈ {0.05, 0.1} | val **loss** | 2 | **0.05** |

Three findings worth stating:

- **Character-level tokenization was a controlled baseline, not a candidate.** It expands input length 6× while halving ROUGE-L to 0.129. The value is in the magnitude, not the direction — the 50% degradation quantifies the cost of losing term identity, which is the *upper bound* on what tokenization alone can affect. Word and subword were statistically indistinguishable (0.2530 vs 0.2521, inside validation noise).
- **r = 16 underperformed r = 8** despite twice the parameters, with higher eval loss (2.60 vs 2.54) — the signature of memorisation rather than generalisation on a 792-example set. The bias–variance trade-off appearing directly in a hyperparameter sweep.
- **Dropout was selected on validation loss, not ROUGE-L**, deliberately: loss is the more sensitive over-fitting signal, and by Stage D the ROUGE gap between dropout values is inside the noise. Using a noisy metric for a fine-grained decision is how spurious hyperparameters get chosen.

**Final: word, r = 8, α = 32, dropout = 0.05 → 884,736 trainable params of 248M (0.36%).**

α/r = 4.0, double the conventional heuristic — on a small dataset a higher scaling factor accelerates learning before overfitting sets in, and the learning curve confirms it.

Retrained on the **full** validation set (the search used a 100-example subset for speed): **ROUGE-L 0.2612, the highest of all 16 runs.** Adapter **6 MB** against a **990 MB** base — the practical argument for LoRA in production, where one frozen base serves many tasks.

---

## Part C — Evaluation

**The first and only use of the test set.** Every prior decision — chunk size, retrieval method, α, guardrail threshold, prompt, tokenization, r, α, dropout — was made on validation.

| System | Weights | Context at inference |
|---|---|---|
| Baseline | frozen base | none — zero-shot floor |
| RAG | frozen base | top-3 retrieved chunks |
| LoRA | fine-tuned adapters | none — knowledge in weights |
| Hybrid | fine-tuned adapters | top-3 retrieved chunks |

The **Baseline was added deliberately**: without it, a ROUGE-L of 0.44 is a number without a scale.

**BLEU is the primary metric** (course recommendation), **ROUGE-L the robustness check.** BLEU's n-gram precision penalises omitting or substituting precise technical terms — the failure that matters most here. The two produce **identical rankings**, so the conclusion does not depend on metric choice.

### Results

| System | ROUGE-L | ROUGE-1 | BLEU | Latency |
|---|---|---|---|---|
| Baseline | 0.1679 | 0.2071 | 0.0150 | 0.67 s |
| RAG | 0.3894 | 0.4328 | 0.2036 | 0.75 s |
| LoRA | 0.2728 | 0.3247 | 0.0569 | 1.10 s |
| **Hybrid** | **0.4436** | **0.4805** | **0.2381** | 1.05 s |

The ordering **RAG > LoRA** is informative in itself: with 792 training examples, adapters cannot absorb the factual content of nine lectures, but retrieval can surface any of it on demand. LoRA's contribution is not knowledge — it is *fluency and answer shape*, which is exactly what Hybrid exploits.

### By question type

| Type | RAG | LoRA | Hybrid | Δ vs RAG |
|---|---|---|---|---|
| Factual | 0.3975 | 0.2886 | 0.4500 | +0.053 |
| Conceptual | 0.4049 | 0.2551 | 0.4395 | +0.035 |
| **Comparative** | 0.3187 | 0.2655 | **0.4328** | **+0.114** |

Comparative questions are where the architecture earns its keep. RAG is *weakest* there — comparison requires synthesising several facts, and a frozen reader handed three chunks tends to echo one. Hybrid's +0.114 is more than double its gain on any other type: retrieval supplies the facts to compare, the adapter supplies the learned pattern of how a comparison is phrased. **This is the clearest evidence for complementarity in the study.**

### Retrieval-conditioned analysis

Hit rate **87.5%** (147/168). RAG's ROUGE-L split by retrieval outcome:

- Hits: **0.3981**
- Misses: **0.3330**
- **Gap: 0.065**

If retrieval quality were the binding constraint, that gap would be large — finding the right chunk would transform the answer. It does not. **The bottleneck is the reader, not the index.**

---

## Error analysis

The 30 worst answers per system, classified using signals already collected — so the taxonomy is **reproducible rather than impressionistic**.

| Category | Hybrid | LoRA | RAG |
|---|---|---|---|
| Generation Error | 23 | 25 | 19 |
| Off-Topic | 4 | 5 | 10 |
| Retrieval Error | 3 | 0 | 1 |
| Knowledge Gap | 0 | 0 | 0 |
| Truncation Error | 0 | 0 | 0 |

- **Generation Error dominates at 63–83%** and Retrieval Error is nearly absent — independent confirmation of the 0.065 gap from an entirely different measurement.
- **Zero Truncation Errors** validates the sentence-level choice: at 39 tokens per chunk, three chunks never approach the 512-token limit.
- **RAG's Off-Topic count (10) is double Hybrid's (4)** — the adapter's contribution made visible. Given a marginally relevant chunk, the frozen reader echoes it; the fine-tuned reader stays anchored to the answer shape.

> **One category was renamed during development.** The original *Hallucination* category (grounding ratio < 0.3) measured lexical divergence, not fabrication — flagging correct paraphrases and missing confident inventions that reused vocabulary. It was renamed **Off-Topic**, redefined as ROUGE-L < 0.05, and genuine hallucination moved to the NLI method below. A metric that assigns LoRA its worst possible score for an *architectural* property (no context ⇒ grounding ratio 0 by construction) measures nothing.

---

## Hallucination detection via NLI

Hallucination is **fabrication**: asserting a specific fact that is not true. No lexical statistic can detect it, because fabrication is semantic.

We framed it as **natural language inference** — reference as *premise*, each claim in the prediction as *hypothesis*, scored by a **DeBERTa-v3 NLI cross-encoder** (`cross-encoder/nli-deberta-v3-small`):

- Each prediction is split into claims (sentence-level, discarding fragments under five words that cannot carry a fact)
- Flagged only if **at least one claim is contradicted at confidence > 0.70**
- **Entailment and neutral are never penalised** — a correct paraphrase and a vague-but-harmless answer both pass

Only assertions that *conflict* with the truth are counted. That is what separates hallucination from being merely wrong.

| System | Hallucination rate | Flagged | ROUGE-L |
|---|---|---|---|
| RAG | 11.9% | 20 / 168 | 0.389 |
| **LoRA** | **14.3%** | 24 / 168 | 0.273 |
| Hybrid | 11.9% | 20 / 168 | 0.444 |

**LoRA hallucinates most**, exactly as the architecture predicts. With no retrieved evidence it answers from parametric memory and invents confidently:

- *"TensorFlow is a neural network developed by the University of California, Berkeley"* — contradiction 1.000
- *"PyTorch is a deep learning framework developed by Carnegie Mellon University"* — contradiction 1.000

RAG and Hybrid are **identical at 11.9%**, which localises the effect precisely: **retrieval is what suppresses fabrication**, and the adapter neither helps nor hurts on this axis. Hybrid therefore achieves RAG's factual safety *and* the best quality scores simultaneously.

### The decisive result

Hallucination rate by ROUGE-L quartile:

| Q1 (worst) | Q2 | Q3 | Q4 (best) |
|---|---|---|---|
| 13.2% | 14.4% | 15.3% | 7.9% |

**Essentially flat.** Hallucination and low lexical overlap are **independent failure modes** — a model can score poorly because it paraphrases differently (a ROUGE-L problem with no factual content), or score well while asserting something false. A single quality number cannot substitute for both.

### High confidence, wrong answer

Live demo probing surfaced a failure the aggregates conceal. Asked *"What is the function of convolutional layers in a CNN?"*, the retriever returned chunks at similarity **0.997, 0.964, 0.896** — all from the correct PDF, all about CNNs. The answer was still wrong: the top chunks discussed CNN *visualisation* and a Keras layer definition, not the feature-extraction role asked about.

**A near-perfect similarity score is not evidence of an answer-bearing chunk.** Cosine similarity measures topical proximity; it cannot distinguish a chunk that *is about* convolutional layers from one that *answers a question about* them. This is invisible to any metric scoring relevance by provenance — including the Hit@k and MRR used here — and is the strongest argument for cross-encoder reranking.

---

## Interactive demo

A Gradio interface exposing exactly what the evaluation measured, so the live demo and the reported numbers tell the same story:

- Free-text question input
- Method selector: **RAG / LoRA / Hybrid**
- Generated answer
- **Top-3 retrieved chunks with similarity scores** — the evidence behind the answer
- The guardrail's decision whenever the system abstains

Some demo questions worth trying:

| Question | What it shows |
|---|---|
| *What is TensorFlow and who developed it?* | LoRA hallucinates (UC Berkeley); Hybrid answers correctly |
| *What is the recipe for chocolate cake?* | Guardrail fires on RAG and Hybrid; LoRA invents a recipe |
| *What is the function of convolutional layers in a CNN?* | High retrieval scores, wrong answer — the reranking argument |
| *What impact do deeper networks have on inference time?* | Clean Hybrid win |

---

## Repository structure

```
.
├── notebooks/
│   ├── notebook_1_data_preparation.ipynb     # PDFs → 1,130 audited QA pairs
│   └── notebook_2_rag_lora_evaluation.ipynb  # RAG · LoRA · evaluation · demo
├── reports/
│   ├── RAG_vs_LoRA_Report_EN_Tamar_Telele.pdf
│   └── RAG_vs_LoRA_Report_HE_Tamar_Telele.pdf
├── slides/
│   ├── Presentation_EN_Tamar_Telele.pptx / .html
│   ├── Presentation_HE_Tamar_Telele.pptx / .html
│   └── Speaker_Notes_HE_Tamar_Telele.pdf
├── figures/                                   # all plots used in the report
├── requirements.txt
└── README.md
```

Runtime artefacts written to Google Drive under `course_study_assistant/`:

| Path | Contents |
|---|---|
| `data/` | `qa_train/val/test.json`, `clean_corpus.json`, `manifest.json`, `DATASET_CARD.md` |
| `models/lora_model/` | adapter weights (6 MB) |
| `results/` | `rag_results.json`, `final_results.json` |
| `figures/` | every figure, regenerated on each run |
| `reports/` | `error_analysis.csv`, `nli_hallucination.csv`, `qualitative_examples.csv` |

---

## Reproducing

```bash
pip install -r requirements.txt
```

1. Put the 9 course PDFs in `MyDrive/AI_cours_pdfs`.
2. Add `GOOGLE_API_KEY` to Colab Secrets.
3. Run **Notebook 1** top to bottom → writes `data/` and `manifest.json`.
4. Run **Notebook 2** top to bottom → RAG pipeline, LoRA search, evaluation, demo.

Every seed is fixed in a single frozen `Config` dataclass at the top of each notebook. Changing an experiment means changing one line there, never hunting through cells.

**Runtime on a Colab T4:** Notebook 1 ≈ 45 min (dominated by Gemini API calls); Notebook 2 ≈ 2.5 h (15 LoRA runs ≈ 90 min, test evaluation ≈ 12 min, NLI ≈ 30 s).

---

## Limitations

- **Reference-answer noise.** Answers were LLM-generated and audited on 10%. Residual noise caps ROUGE-L and BLEU from above: a correct paraphrase is penalised against a reference reflecting one model's phrasing. The *ranking* is trustworthy; the absolute ceiling is not.
- **Generation capacity binds.** `flan-t5-base` has 250M parameters. The 0.065 hit/miss gap and the 63–83% generation-error share both say retrieval improvements have little headroom on this reader.
- **Granularity, not algorithm.** The chunking ablation varies size and overlap with the splitter fixed; semantic and page-boundary chunking were not tested.
- **Coordinate search is not a full grid.** Only the r×α interaction was measured explicitly; an interaction involving tokenization or dropout would be invisible to this procedure.
- **NLI inherits its model's calibration.** A fabricated fact *absent from* rather than *conflicting with* the reference registers as neutral, so reported rates are a **lower bound** on true fabrication.
- **Single-seed training.** Each configuration was trained once; gaps under ~0.01 ROUGE-L are not significant.

---

## Future work

Ordered by expected value, given that the reader — not the retriever — is the measured ceiling:

1. **Hybrid fine-tuning.** Train the adapter *with retrieved context inside the prompt*, so it learns to read evidence rather than recall facts. This attacks the measured bottleneck directly.
2. **Cross-encoder reranking.** Rerank the top-k by query–chunk interaction rather than embedding distance, addressing the 0.997-similarity failure that no bi-encoder can detect.
3. **Question-type routing.** Different systems lead on different types; a lightweight classifier routing factual → RAG, conceptual → LoRA, comparative → Hybrid could beat any fixed architecture.
4. **Semantic evaluation.** BERTScore or a calibrated LLM judge as the primary metric, removing the reference-phrasing ceiling that caps ROUGE-L.
5. **Larger reader.** A quantised 7–8B model on the identical pipeline would isolate how much of the ceiling is model capacity.

---

## Beyond the course baseline

| | Course notebook | This project |
|---|---|---|
| **RAG corpus** | 7 hand-written sentences | 1,130 audited QA pairs from 9 real course PDFs |
| **Retrieval** | Dense only | Dense + BM25 + Hybrid, α tuned on validation |
| **Retrieval eval** | none | Hit@1, Hit@3, MRR across a 3×3 grid |
| **Index** | `IndexFlatL2` | `IndexFlatIP` + L2 normalisation → calibrated guardrail |
| **Reader** | distilgpt2 | flan-t5-base + beam search + prompt selection |
| **Embedding viz** | PCA | t-SNE across all three granularities |
| **LoRA search** | one fixed config | 4-stage coordinate search, r×α interaction measured |
| **Task** | generic sentiment (`tweet_eval`) | course-specific free-text QA |
| **Baseline** | none | zero-shot floor, making every gain interpretable |
| **Hallucination** | none | NLI-based, validated as independent of ROUGE-L |
| **Demo** | none | live three-way Gradio comparison |

The methodological core is that **all systems share one frozen base checkpoint** — without which none of these numbers would mean anything.

---

## Acknowledgements

Course materials and project framing: **Dr. Barak Or**, Applied Language Models, Google &amp; Reichman University Tech School.

Base model: `google/flan-t5-base`. Encoder: `sentence-transformers/all-MiniLM-L6-v2`. NLI: `cross-encoder/nli-deberta-v3-small`. Dataset generation and judging: Gemini API.

LoRA follows Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*.
