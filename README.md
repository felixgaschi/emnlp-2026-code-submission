# Code for Anonymous EMNLP 2026 Submission

## Provenance

This repository extends an existing codebase for multilingual realignment of
language models, that is not our contribution. The base code implements the
training infrastructure and realignment objectives described in prior work;

There are only two commits on this repository:

- one commit with the base code, which we inherited and did not author. We squashed all the history in a single commit
- one commit with our contributions for this paper, which is the focus of this repository and is our contribution.

We describe exactly what is ours vs. inherited below.

---

## What we inherited (base code)

The following were already present and are **not** our contribution:

| Path | Description |
|------|-------------|
| `multilingual_eval/datasets/` (most files) | Dataset loading for XNLI, WikiANN, XTREME-R |
| `multilingual_eval/models/realignment_loss.py` | Base realignment loss |
| `multilingual_eval/models/with_realignment_factory.py` | Model factory |
| `multilingual_eval/training/training_loops.py` | Base training loop |
| `multilingual_eval/training/epoch_loop.py` | Base epoch loop |
| `multilingual_eval/training/evaluation_loops.py` | Evaluation loop |
| `multilingual_eval/training/states.py` | Training state management |
| `scripts/2025_aacl/` | Experiment scripts from prior work |
| `download_resources/` | Data download utilities |

---

## Our contributions (this paper)

### New modules

| Path | Description |
|------|-------------|
| `multilingual_eval/information_retrieval/` | **New.** Cross-lingual information retrieval support: MS MARCO training (`train_msmacro.py`), CLIRMatrix evaluation (`eval_clirmatrix.py`), and shared data utilities (`data_fn.py`) |
| `multilingual_eval/training/decoupled_weighted_selection.py` | **New.** Decoupled meta-weight learning for language selection during realignment |
| `multilingual_eval/datasets/weighted_data_utils.py` | **New.** Weighted data sampling utilities used by the meta-learning approach |
| `multilingual_eval/models/simplified.py` | **New.** Simplified model variants for ablation |
| `multilingual_eval/models/stwrapper.py` | **New.** Sentence-transformer wrapper for CLIR evaluation |
| `scripts/ours/` | **New.** All experiment scripts for this paper's results |

### Extensions to existing modules

| Path | What we added |
|------|---------------|
| `multilingual_eval/training/training_loops.py` | Weighted sampling integration, decoupled weight support |
| `multilingual_eval/training/epoch_loop.py` | Checkpoint resumption for meta-learning iterations |
| `multilingual_eval/models/realignment_loss.py` | Closed-form solution variant |
| `multilingual_eval/datasets/dispatch_datasets.py` | CLIR dataset dispatch |
| `multilingual_eval/datasets/data_utils.py` | Minor fixes |

---

## Reproducing our experiments

See [`scripts/ours/`](scripts/ours/) for
the experiment scripts. The main entry point is:

```bash
python scripts/ours/controlled_realignment.py [args]
```

Dependencies are managed with `uv` (see `pyproject.toml`).

### Pre-defined scripts

There are pre-defined scripts to re-run our experiments but they require a lot
of time (more than 12 hours) and storage to retrieve (and segment in the case of Chinese) the parallel
data used for realignment.

To download the full dataset you can run:

```bash
bash scripts/ours/download_resources.sh ./data
```

And then you can use the script scripts/ours/final_run_script.sh as you see fit:

```bash
bash scripts/ours/final_run_script.sh mix_opus100_nllb \
    <STRATEGY> \
    <SELECTION_STRAT> \
    <MODEL> \
    <TASK> \
    <SEED> \
    16000 \
    128 \
    <ADDITIONAL_ARGS>
```

STRATEGY can be one of "baseline" or "before_noaligner" (the codebase we worked with have other aligners but they are not relevant to our paper, so we ignore them in the script).

SELECTION_STRAT can be one of the following:

- "xt_afri": for sampling among all languages (uniform or not)
- "most_uriel_en_40": 40 languages with the most Uriel distance to English
- "baseline": for fine-tuning only

MODEL can be any Huggingface model, but we tested "xlm-roberta-base" and "google/gemma-2-9b"

TASK can be one of the following: xnli, wikiann or xtreme_r.udpos

SEED can be an integer or a string containing several integers

ADDITIONAL_ARGS are additional arguments to pass to the training script, for our two baselines we use:

- gradient-based: `"--enable_weighted_sampling --weighted_sampling_method meta_learning --inner_batch_before_outer 10 --meta_learning_rate 1e-3"`
- UCB-based: `"--enable_weighted_sampling --weighted_sampling_method ucb --ucb_exploration_coef 0.1"`

It's important they are wrapped into a string to be passed as a single argument.

### Toy reproduction

There are toy scripts that reproduce results with a smaller subset of languages and data, which can be run relatively quickly for testing purposes.

However they still do require more than 12GB of storage.

Instead of download_resources.sh and final_run_script.sh, you can run the following scripts:

```bash
bash scripts/ours/download_resources_toy.sh ./data
bash scripts/ours/final_run_script_toy.sh \
    <STRATEGY> \
    <SELECTION_STRAT> \
    <MODEL> \
    <TASK> \
    <SEED> \
    16000 \
    128 \
    <ADDITIONAL_ARGS>
```