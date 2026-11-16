# Final Dev Pipeline — SemEval-2027 Semantic Change (SV)

Google Colab pipeline for **SemEval-2027: Semantic change across multiple time periods** (Swedish, `SV`, Subtask 1a & 1b). It extracts contextual embeddings, clusters word usages into overlapping senses, and evaluates against gold labels.

## Pipeline Steps

1. **Data Loading**: Mounts Google Drive and reads `usages.jsonl` and `SV/subtask1.jsonl` from `dev.zip` (validating unique `(word, sentence_id)` keys and character spans).
2. **Embedding Extraction** (4 branches):
   - `bert_target`: Mean-pooled target token states from `KB/bert-base-swedish-cased`.
   - `xl_target`: Mean-pooled states from `pierluigic/xl-lexeme` with `<t>...</t>` target markers (`XL_MAX_LENGTH = 128`).
   - `bert_predicted_x`: Top-1 alphabetic `[MASK]` prediction for `"<target> och [MASK]"` embedded via Swedish BERT.
   - `xl_predicted_x`: XL-LEXEME embeddings for `predicted_X` (currently overwritten by `xl_model.encode(usages_df.text.tolist())` from a `# TESTING SMALLER XL-LEXEME` block).
3. **Sense Clustering & Assignment**:
   - **GMM & DPMM Sweep**: L2-normalized + PCA (`<= 20` dims) embeddings per word across `GMM` (`K = 1..50`) and `BayesianGaussianMixture` (`priors = [0.1, 1.0, 10.0]`), with multi-sense assignment via posterior thresholds (`[0.10, 0.20, 0.25, 0.30, 0.40]`).
   - **Agglomerative Clustering**: Average-linkage clustering on `xl_target` cosine distances using a per-word `0.50` distance quantile cutoff and `0.05` overlap margin.
4. **Evaluation**: Computes word-level Overlapping B-cubed (`bcubed`) and full Subtask 1a (`B3_F1`) / 1b (`subtask1b_rho`) scores.

## Setup

```bash
pip install transformers sentence-transformers scikit-learn scipy numpy pandas tqdm matplotlib bcubed
