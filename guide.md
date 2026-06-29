# PyTorch → Neural Nets → AGI-Adjacent Systems → Kaggle
### A 5-10 hr/week roadmap, free-GPU only

---

## How this is structured

Five phases, each ending in a project that becomes *infrastructure* for the next phase — not a throwaway exercise. Phase 0 builds a training harness you reuse forever. Phase 2's transformer becomes the backbone of Phase 3's recommender. Phase 3's recsys becomes the environment for Phase 4's bandit/RL agent. By the capstone, you're not doing five separate things — you're assembling one system.

Rough timeline at 5-10 hrs/week:

| Phase | Weeks | Focus |
|---|---|---|
| 0 | 1-2 | PyTorch fluency + your own training harness |
| 1 | 3-5 | CNNs, vision, transfer learning, first Kaggle entry |
| 2 | 6-10 | Transformers/LLMs from scratch + fine-tuning |
| 3 | 11-15 | Recsys + Kafka (the "X algorithm" clone) |
| 4 | 16-19 | RL, bandits, agents |
| 5 | 20-23 | Multimodal (CLIP, small VLMs) |
| Capstone | 24+ | Unify into one system, ongoing |

Don't treat phases as strictly sequential prisons — once you're in Phase 1, keep a Kaggle tab open and submit something most weeks, even sloppy. Reps matter more than perfection.

---

## Compute strategy (important, given free-tier only)

You effectively have **two separate free GPU quotas**: Colab (T4, dynamic limits, ~12hr session cap) and Kaggle Notebooks (P100/T4x2, 30 hrs/week, 9hr session cap). Use Kaggle's quota for anything that needs a long uninterrupted run (full fine-tunes, longer RL training), and Colab for iteration/debugging. That's effectively 40+ GPU-hours/week for free if you're disciplined about it.

**Phase 3 (Kafka/recsys systems) needs zero GPU** — it's a systems/data-engineering problem, not a training problem. Don't waste Colab time on it. Run Kafka via Docker Compose locally, or on GitHub Codespaces (60 free hrs/month), or a free-tier Confluent Cloud / Redpanda cluster. Keep that work in a separate repo from your model-training notebooks.

---

## Phase 0 — PyTorch fluency + your training harness (Weeks 1-2)

**Concepts:** tensors/autograd internals, `nn.Module` composition, custom `Dataset`/`DataLoader`, manual training loops (don't lean on Trainer abstractions yet — you need to feel the loop), mixed precision (`torch.amp`), gradient clipping, checkpointing, `torch.compile`.

**Project — "TrainKit":** Build a small, reusable training harness from scratch: a `Trainer` class handling the train/val loop, AMP, checkpointing, early stopping, and logging to Weights & Biases (free tier). Validate it end-to-end on FashionMNIST with a CNN you write by hand (no `torchvision.models`).

Why this matters: every later phase reuses this harness instead of you rewriting boilerplate each time. Treat it like a personal library — give it a GitHub repo.

---

## Phase 1 — CNNs, vision, transfer learning (Weeks 3-5)

**Concepts:** convolutions, batchnorm, residual connections, augmentation, transfer learning, the `timm` library, learning rate schedules (cosine, warmup).

**Project:** Fine-tune a `timm` backbone (e.g. `efficientnet` or `convnext_tiny`) on a real Kaggle vision competition using your Phase 0 harness. Good current target: **BirdCLEF+ 2026** (acoustic species ID — technically audio, but you'll process it as spectrogram images, which is a great bridge skill). Alternatively any active Playground Series episode if you want a faster, lighter-weight cycle to practice submission mechanics before committing to a longer comp.

**Kaggle note:** Kaggle is currently running **Playground Series Season 6** — a new lightweight tabular/vision competition roughly every month. These are perfect for fast reps: read top public notebooks, submit something within your first session, iterate. Don't skip these even once you're deep into later phases — they're your "scales practice."

---

## Phase 2 — Transformers & LLMs from scratch (Weeks 6-10)

This is the core "AGI-adjacent" phase, and it directly feeds Phase 3.

**Concepts:** self-attention math (by hand, not just conceptually), positional encoding, the transformer block, causal masking, KV-cache, tokenization (BPE), LoRA/PEFT for fine-tuning, quantization basics.

**Project A — nanoGPT-style from scratch:** Build a character-level GPT with no Hugging Face — raw `nn.Module`, write the attention yourself. Train on a small corpus (Shakespeare is the classic, but use something more interesting to you) on a Colab T4. Goal isn't SOTA, it's that you *understand every line*.

**Project B — fine-tune a small open model:** Use Hugging Face + PEFT to LoRA fine-tune a small open model (Qwen2.5-0.5B/1.5B or Llama-3.2-1B — small enough to fit on a free T4) on a downstream task. This is the actual industry-relevant skill.

**Kaggle stretch goal (don't start here, but aim at it):** **ARC Prize 2026** (ARC-AGI-2 and ARC-AGI-3 are both running on Kaggle right now) is *literally* an AGI benchmark — novel reasoning, program synthesis, test-time adaptation. It's hard and most solutions blend neural nets with search/program synthesis, not pure end-to-end deep learning. Once you've got Phase 2 + Phase 4 (RL/search) under your belt, this is the single most direct "closer to AGI" competitive outlet available on Kaggle today.

---

## Phase 3 — Recommendation systems + Kafka (the X-algorithm clone) (Weeks 11-15)

**Concepts:** two-tower retrieval models (user tower + item tower → embeddings → ANN search), candidate generation vs. ranking (light ranker → heavy ranker pattern), sequence-aware recommenders (SASRec/BERT4Rec — your Phase 2 transformer knowledge pays off directly here), real-time feature pipelines, Kafka producer/consumer basics, feature stores.

X's open-sourced "the-algorithm" repo is a good reference architecture: candidate retrieval (multiple sources) → light ranker (cheap, high-recall) → heavy ranker (expensive, precise) → real-time engagement events streamed through Kafka to update features/models. You're not trying to reproduce their actual signals (proprietary, and a lot has likely changed since 2023) — you're borrowing the *shape* of the pipeline.

**Project — mini recommendation engine:**
1. Train a two-tower retrieval model in PyTorch on MovieLens or the H&M Kaggle recsys dataset.
2. Upgrade the item-sequence side to a small transformer (reuse Phase 2 code) so recommendations are sequence-aware, not just static embeddings.
3. Build a Kafka pipeline (Docker Compose, runs on your laptop/Codespaces — no GPU): a producer simulating user click/engagement events, a consumer that updates a feature store, and a scoring loop that periodically re-ranks using your trained model.
4. Bonus: add a simple online metric dashboard (CTR proxy, etc).

This becomes the spine of your capstone.

---

## Phase 4 — RL, bandits, and agents (Weeks 16-19)

**Concepts:** MDPs, policy gradients, PPO, value functions, multi-armed bandits (especially contextual bandits — directly relevant to recsys exploration/exploitation).

**Project A:** PPO via Stable-Baselines3 on Gymnasium classic control (CartPole) then LunarLander — get the mechanics down without writing PPO from scratch first.

**Project B (ties back to Phase 3):** Implement a contextual bandit (e.g. LinUCB or a small neural bandit) as the *exploration layer* on top of your Phase 3 recommender — instead of always serving the top-ranked item, balance exploration of uncertain items vs. exploiting known-good ones. This is exactly the kind of explore/exploit problem real recsys ranking layers solve.

**Project C (agentic loop, optional but worthwhile):** Build a simple ReAct-style agent using your Phase 2 fine-tuned small model with function/tool calling — even a toy agent that can call 2-3 tools teaches you the actual agent loop (plan → act → observe → repeat) that underlies most "AGI-adjacent" product work today.

---

## Phase 5 — Multimodal (Weeks 20-23)

**Concepts:** contrastive learning, CLIP's architecture, vision-language alignment.

**Project:** Linear-probe or LoRA fine-tune open CLIP on an image+text task (could be a Kaggle multimodal competition if one's active, or a custom image-search/captioning project). Optionally explore a small open VLM (quantized, runs on a free T4) for a visual reasoning task.

---

## Capstone (Week 24+, ongoing)

Merge everything into one coherent system: the Phase 3 recsys + Kafka pipeline, with a Phase 2 transformer as the sequence encoder, a Phase 4 bandit as the exploration layer, and (optionally) a Phase 5 multimodal signal (e.g. image embeddings of items feeding the item tower). Deploy it as a small end-to-end demo — this is your portfolio centerpiece, and it's a genuinely good thing to show in interviews: it demonstrates you can do modeling *and* systems, which most ML-only candidates can't.

---

## Kaggle competitiveness, as an ongoing layer (not a phase)

- **Quick reps:** Playground Series (new episode roughly monthly) — submit within the first session of any new one, even before you've "finished" a phase. Speed of iteration matters more than getting it right first try.
- **Depth:** pick one Featured/Code competition aligned with whatever phase you're in (vision comp during Phase 1, ARC Prize once Phase 2+4 are solid) and go deep — read winning solution write-ups *after* every competition ends, even ones you didn't enter. This is one of the highest-leverage habits in competitive ML; the post-mortems teach you tricks no course will.
- **GPU stacking:** use Kaggle's notebook GPU quota for the actual competition (data's already there, no upload friction) and Colab for unrelated experimentation, so you're not burning competition-quota on side projects.
- **Don't ensemble too early.** Get one clean model working end-to-end before reaching for ensembling/stacking — it's tempting to jump straight to leaderboard tricks, but the fundamentals compound more.

---

## Suggested weekly split (5-10 hrs)

- ~60% current phase project work
- ~25% Kaggle reps (whatever's active)
- ~15% reading: one paper or one winning-solution write-up, not more — depth over breadth

---

## A few honest notes

- "All four directions, 5-10 hrs/week" is genuinely ambitious — expect this roadmap to take 6-7 months to get through once, not as mastery but as solid hands-on grounding in each. That's normal and fine.
- The ARC Prize is a good long-term aiming point precisely *because* it's hard — it's one of the few competitions where "closer to AGI" isn't just a vibe, it's the literal evaluation criterion (novel reasoning on unseen tasks). Don't start there; arrive there.
- If at any point a phase feels too slow, it's fine to compress — e.g. you can fold Phase 5 into Phase 3 if multimodal item embeddings interest you more than RL does. The dependency order (0→1→2→3) matters more than the exact pacing.
