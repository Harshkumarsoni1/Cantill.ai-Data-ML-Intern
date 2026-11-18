# Method and measured performance

The pipeline reads each source through the mappings in its company config. It normalizes amounts to net reporting currency, evaluates status and account rules, then reconciles the secondary export against primary receipts. Candidate selection by purchase order or amount keeps the matching search small. Amount, text similarity, quantity and supplier agreement distinguish candidates. Each accepted match consumes one receipt. The ERP row carries the spend.

Supplier names are normalized for punctuation, accents, whitespace, and legal suffixes. Training labels supply majority mappings from normalized supplier and material identifiers to company keys. A unique description-to-material mapping helps identify uncoded items. Fallback keys are deterministic hashes. These rules are conservative for aliases and can split equivalent items with different descriptions.

With labels, a linear support vector classifier predicts a complete path from word and character TF-IDF features over descriptions, account or material context, supplier and material identifiers. Only paths appearing in the runtime taxonomy can be predicted. A bounded function of the top-two margin supplies a confidence indicator. It is not a calibrated probability.

Without labels, the model represents each taxonomy path by its names, giving extra weight to its leaf. Character n-gram similarity scores each purchased item against each path. For suppliers with at least three distinct informative descriptions, the supplier's average scores contribute 28% of the score. The highest-scoring valid path wins. Short codes, translations, and descriptions that share little wording with their category remain difficult.

**Hidden-label evaluation.** A fresh run disabled label loading, classified A and scored all 37,029 ERP records against their labels after prediction. Exact L1 accuracy was 49.27% (18,246 rows), L2 23.13% (8,564), and L3 16.65% (6,167). These are row-weighted results. Company C performance is unknown. No supervised training-set accuracy is represented as a blind estimate.

**Runtime.** On the build machine A processed 65,549 rows in 26.8 seconds, B processed 41,028 in 3.9 seconds, and the hidden-label A run took 8.1 seconds. The submitted CSVs passed the row and taxonomy checks in verify.py.

**AI disclosure.** OpenAI Codex generated the implementation and documentation and ran the build and checks. The submitted program runs locally without an LLM or external service. API cost is zero per 1,000 rows.
