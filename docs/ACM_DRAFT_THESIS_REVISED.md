# Interpretable Action-Item Detection in Multilingual Meeting Pipelines Using Incremental Linear SVM

Venus Ruselle B. Daanoy  
College of Informatics and Computing Studies, New Era University  
venusruselle.daanoy@neu.edu.ph

John Keith S. Mercado  
College of Informatics and Computing Studies, New Era University  
johnkeith.mercado@neu.edu.ph

## ABSTRACT
Hybrid and remote collaboration has increased the need for reliable meeting intelligence systems that can extract actionable commitments from long transcripts, especially in multilingual and code-switched contexts such as Taglish. This paper presents an interpretable, modular four-phase pipeline for meeting processing: (1) speech transcription with dual-engine fallback (Groq and local Whisper), (2) semantic segmentation using sentence-embedding cosine boundaries and token-budget constraints, (3) action-item detection using a hashing-based feature space and incremental linear SVM (SGD hinge), and (4) abstractive summarization with Groq/local BART fallback. The classifier supports deployment-oriented threshold modes (balanced and high-recall), enabling explicit false-positive versus false-negative control. On the primary evaluation set, the precision-recall curve achieved AUPRC = 0.9162. Balanced mode produced precision = 0.8080 and recall = 0.9095, while high-recall mode achieved precision = 0.8089 and recall = 0.9146, reducing missed actions (FN: 18 to 17) without increasing false alarms (FP: 43 to 43). Data hygiene analysis reported 184,976 raw rows reduced to 64,144 unique rows after deduplication with 12 conflict groups, strengthening reproducibility and label quality claims. Results show that an interpretable linear model, when embedded in a robust pipeline and calibrated via threshold operating points, remains practical and competitive for multilingual meeting action extraction.

**CCS Concepts:** Computing methodologies -> Supervised learning by classification; Natural language processing; Information extraction; Document summarization.  
**Keywords:** Action-item detection, Taglish NLP, Linear SVM, Meeting intelligence, Precision-recall tradeoff, Interpretable ML

## 1 INTRODUCTION
Meeting transcripts contain operationally critical commitments but are often too long and noisy for manual review. In multilingual settings, especially Taglish, sentence structure, politeness forms, and indirect directives make action-item detection difficult for generic NLP systems. While deep neural architectures can perform strongly, they are often harder to interpret and more expensive to retrain for iterative domain updates.

This work proposes a practical alternative: an interpretable linear SVM core integrated into a resilient end-to-end pipeline with explicit fallback behaviors and deployment threshold control. The study emphasizes action-item extraction as the primary objective and summarization as a supporting output layer for user readability.

## 2 PROBLEM STATEMENT
The rapid growth of hybrid and remote meetings has produced long, unstructured transcripts where operational commitments are difficult to track manually. In multilingual settings, especially Taglish conversations, action-oriented statements are often expressed through code-switching, indirect phrasing, and politeness markers, which makes automatic extraction difficult for generic NLP pipelines. Existing methods also present practical constraints: some prioritize broad summarization over explicit task extraction, some reduce interpretability for end users, and some do not provide clear operating-threshold control for balancing missed commitments against false alarms. As a result, organizations risk overlooked follow-ups, inconsistent task accountability, and delayed execution.

This study therefore addresses the following research questions:

1. How effectively can a modular multilingual pipeline detect action items in long meeting transcripts under real deployment constraints?
2. How do threshold operating modes (balanced versus high-recall) change precision-recall behavior and operational error costs?
3. Can an interpretable linear SVM remain competitive when combined with robust preprocessing, segmentation, and calibration?
4. How reliable is the end-to-end workflow when transcription, semantic segmentation, classification, and summarization are integrated in one system?

### 2.1 Main Objective
To develop and evaluate an interpretable, modular meeting-intelligence system that accurately extracts action items from multilingual meeting data and supports practical deployment through threshold-aware decision control.

### 2.2 Specific Objectives
1. To implement a robust transcription phase with dual-engine processing and fallback behavior for audio and video meeting inputs.
2. To design a semantic segmentation phase that partitions transcripts into topic-coherent, token-bounded units for cleaner downstream analysis.
3. To build an incremental linear SVM-based action-item classifier using hashing-based n-gram features and confidence-aware decision outputs.
4. To configure and compare balanced and high-recall operating modes to identify deployment thresholds based on false-positive and false-negative tradeoffs.
5. To generate concise meeting summaries while preserving an auditable list of extracted action items.
6. To evaluate the system using precision, recall, F1, confusion matrices, and precision-recall analysis across multiple test sets.
7. To assess data quality and reproducibility through deduplication, conflict checks, and transparent reporting of evaluation artifacts.
8. To identify current limitations and recommend improvements for generalization, live-capture robustness, and action-owner attribution.

## 3 SCOPE AND LIMITATIONS
This study covers the full meeting-intelligence pipeline implemented in the current system: audio and video transcription using Groq with local Whisper fallback, semantic transcript segmentation using embedding-based cosine boundaries with token constraints, sentence-level action-item classification using hashing-based features and incremental linear SVM, and abstractive summary generation using Groq or local BART fallback. Evaluation is performed across available AMI-focused, Taglish-focused, and stricter unique-split test sets. The primary modeled objective is action-item extraction, while summary generation is treated as a downstream support output for readability.

Within this scope, several limitations remain. End-to-end output quality depends on upstream transcription quality, particularly in noisy or overlapped speech. Speaker attribution for explicit owner assignment is limited and is not the primary modeled target in the current version. Threshold operating points may shift across domains and therefore require periodic recalibration. Generalization to dialect-heavy or previously unseen domain styles may need additional retraining data. In live operation, real-time capture constraints, especially system-audio capture stability, can also affect reliability.

## 4 CONTRIBUTIONS
1. A production-oriented four-phase architecture with failover transcription and failover summarization.
2. Semantic segmentation with topic-coherent, token-bounded units to improve downstream classification context.
3. Incremental linear SVM action classifier with configurable operating modes for threshold-aware deployment.
4. Reproducible evaluation artifacts: PR curve, threshold tradeoff, confusion matrices, confidence distribution, data hygiene audit, and ablation outputs.

## 5 METHODOLOGY
### 5.1 Phase 1: Transcription
Audio/video inputs are transcribed using a dual-engine strategy:

1. Primary path: Groq transcription with retry logic.
2. Fallback path: local Whisper model when cloud retries are exhausted.
3. Large files and videos are processed through chunked transcription mode.

This phase minimizes complete-pipeline failure under network or quota constraints.

### 5.2 Phase 2: Semantic Segmentation
The transcript is split into sentence-like units and segmented using cosine similarity of sentence embeddings. Topic boundaries are introduced when semantic continuity drops, while a token budget enforces bounded segment size. Each segment stores metadata (token count, sentence count, topical gist) to support downstream context handling and explainability.

### 5.3 Phase 3: Action-Item Classification
Each candidate sentence is transformed with hashing-based n-gram features and classified by an incremental linear SVM (SGD hinge). A signed margin is converted to confidence-like scores for calibration.

Decision function:

f(x) = w^T x + b

Operating threshold:

y_hat = 1 if f(x) >= tau, else 0

Where tau is selected by deployment mode:

1. Balanced mode: default threshold prioritizing overall tradeoff.
2. High-recall mode: lower threshold to reduce missed commitments.

The system also includes sentence-boundary safeguards and optional Llama-based verification for uncertain contexts.

### 5.4 Phase 4: Abstractive Summarization
After extracting actionable items, the pipeline generates a concise meeting summary using Groq or local BART. If the selected engine fails, runtime fallback is applied. Action items remain separately auditable from the narrative summary.

## 6 EQUATIONS AND MATHEMATICAL MODEL
This section formalizes the core classifier and evaluation functions used in the system.

### 6.1 Linear Decision Function
For an input sentence feature vector x in hashed feature space:

f(x) = w^T x + b

Where w is the learned weight vector and b is the bias term.

### 6.2 Thresholded Prediction Rule
Given operating threshold tau:

y_hat = 1 if f(x) >= tau, else 0

1. tau = 0.0 corresponds to balanced mode in the reported main run.
2. tau < 0 shifts toward high-recall mode by labeling more samples as action items.

### 6.3 Margin-to-Confidence Mapping
The implementation maps absolute margin to a confidence-like score using sigmoid:

confidence = sigma(|f(x)|) = 1 / (1 + exp(-|f(x)|))

This supports confidence distribution analysis and uncertainty-aware review.

### 6.4 Incremental Linear SVM Objective (SGD Hinge)
The classifier is trained incrementally with hinge loss and L2 regularization:

min over w,b:  (lambda / 2) * ||w||^2 + (1/N) * sum max(0, 1 - y_i * (w^T x_i + b))

Where y_i in {-1, +1} are transformed class labels.

### 6.5 Evaluation Metric Equations
Given confusion-matrix counts TP, FP, FN:

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 = 2 * Precision * Recall / (Precision + Recall)

Average Precision / AUPRC summarizes precision-recall behavior over all thresholds.

## 7 DATA PREPARATION AND QUALITY CONTROLS
The training and evaluation workflow uses multilingual meeting datasets, including Taglish-heavy sources and AMI-style meeting corpora. Data hygiene is explicitly measured before model reporting:

1. Rows before deduplication: 184,976
2. Rows after deduplication: 64,144
3. Duplicates removed: 120,832
4. Label conflict groups: 12

These controls reduce leakage risk and improve label consistency transparency.

## 8 EXPERIMENTAL SETUP
### 8.1 Primary Metrics
Model quality is evaluated using:

1. Precision
2. Recall
3. F1 (action class)
4. Macro F1
5. Precision-Recall curve and AUPRC
6. Confusion matrices per operating mode

### 8.2 Why PR-Centric Evaluation
Because action-item detection is cost-sensitive and potentially class-imbalanced, PR analysis is prioritized over accuracy-only reporting.

## 9 RESULTS AND DISCUSSION
### 9.1 Precision-Recall Performance
The primary evaluation run achieved AUPRC = 0.9162, indicating strong ranking quality for actionable versus non-actionable sentences across thresholds.

### 9.2 Threshold Operating Points
Observed operating points are:

1. Balanced mode (tau = 0.0): precision = 0.8080, recall = 0.9095, F1_action = 0.8558, FP = 43, FN = 18.
2. High-recall mode (tau = -0.0015): precision = 0.8089, recall = 0.9146, F1_action = 0.8585, FP = 43, FN = 17.

Interpretation: high-recall mode reduced missed actions by one instance without adding false alarms in this set.

### 9.3 Confusion-Matrix Interpretation
Balanced mode: TP = 181, TN = 158, FP = 43, FN = 18.  
High-recall mode: TP = 182, TN = 158, FP = 43, FN = 17.

Operationally, if the cost of missing commitments exceeds the cost of reviewing extra candidates, high-recall mode is preferred.

### 9.4 Cross-Domain Behavior
Evaluation folders indicate distinct difficulty regimes:

1. Full Taglish set reports near-ceiling metrics (distribution-aligned behavior).
2. Strictly unique Taglish split shows expected performance drop (harder generalization).
3. AMI-focused run demonstrates competitive behavior with threshold-dependent tradeoffs.

This pattern supports reporting both in-distribution and stricter out-of-distribution style subsets.

## 10 THREATS TO VALIDITY
1. Transcription quality variance can propagate errors to segmentation and classification.
2. Dataset overlap and synthetic patterns may inflate easier-split performance.
3. Threshold tuning can overfit a single validation distribution if not monitored across held-out subsets.
4. Multilingual coverage remains strongest for English-Tagalog mixtures and may require adaptation for other dialect-heavy settings.

## 11 CONCLUSION
This study demonstrates that interpretable, incremental linear SVM classification remains effective for multilingual meeting action-item extraction when embedded in a robust, threshold-calibrated four-phase pipeline. Rather than relying on a monolithic black-box architecture, the system separates transcription, segmentation, classification, and summarization into independently testable components with fallback behavior. Empirical results (AUPRC 0.9162 in the main run) and operating-point analysis show that deployment priorities can be explicitly tuned, enabling practical, transparent tradeoff management between missed commitments and review burden.

## 12 RECOMMENDATIONS
1. Add confidence intervals via bootstrap for PR and threshold metrics.
2. Expand hard-split evaluation with stricter unseen-domain protocols.
3. Improve live system-audio capture reliability and publish stress-test logs.
4. Add speaker attribution for action ownership extraction.
5. Introduce structured experiment tracking for model/dataset versioning.

## Appendix A: Replace-Now Fixes for the Existing Draft
Use this checklist when transferring this revision into the original manuscript:

1. Replace "six-phase methodology" with the current four-phase pipeline.
2. Remove statements claiming no STT engine; the current system includes transcription as Phase 1.
3. Remove "extractive-only" claim; the current implementation includes abstractive summarization.
4. Replace accuracy-centric claims with PR-centric analysis and threshold operating points.
5. Align scope wording with current runtime: file mode, live mode, and training mode.
6. Keep action-item extraction as primary objective, summarization as support objective.
