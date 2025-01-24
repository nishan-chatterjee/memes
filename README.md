# Mining Emerging Multi-word Expressions (MEMES) from Social Media

## Overview

This project aims to develop an automated framework for discovering emerging multi-word expressions (MWEs) from large social media corpora. The framework assigns confidence scores to potential MWEs, which can then be used in a human-in-the-loop system for training models to detect emerging MWEs effectively.

---

## Project Objectives

1. **Bridge the Gap:** Address the lack of frameworks for detecting emerging MWEs by leveraging social media data.
2. **Develop Automatic Tools:** Use statistical measures and language models to identify potential MWEs.
3. **Human-in-the-Loop System:** Create a pipeline that enables human supervision to train more accurate MWE detection models.

---

## Methodology

### 1. Reference and Social Media Corpora

- **Reference Corpus:** Acts as the baseline for expected expressions.
- **Social Media Corpus:** Target for identifying novel expressions and potential MWEs.

### 2. N-gram Extraction

- Extract n-grams (n = [2-5]) for experimentation and bootstrapping.

### 3. Candidate MWE Extraction

- Primary Approach:

  - **Pointwise Mutual Information (PMI):** Measures co-occurrence of word pairs beyond random chance.
- Alternative Approaches:

  - **Specific Total Correlation (STC):** Captures total interaction among words in multi-word phrases.
  - **Specific Information Interaction (SII):** Quantifies shared information between words in a phrase.
  - **Student’s T-Test:** Tests significance of word co-occurrences.
  - **Dice Coefficient:** Measures similarity between word pairs.
  - **Chi-Square Test (χ²):** Assesses statistical significance of co-occurrence.
  - **LogDice (Rychlý, 2008):** Adjusts Dice Coefficient to capture low-frequency MWEs in large corpora.
  - **Longest-Commonest Match (LCM):** Identifies patterns of commonly recurring word sequences.

### 4. Filtering Existing MWEs

- Remove MWEs already identified using:
  - **PARSEME1.3 Corpus**
  - **MWE-CWI Corpus**
  - **Streusle Corpus**
  - **EPIE Corpus**
  - Named Entity Recognition (NER) models

### 5. Using LLMs as Annotators

- Leverage large language models (LLMs) to:
  - Annotate potential MWEs.
  - Evaluate whether an n-gram, in context, qualifies as a multi-word expression.

---

## Project Structure

```
project-root/
│
├── data/
│   ├── corpus/         # Intermediate files
│   ├── external/       # Downloaded MWEs and cleaned versions
│
├── notebooks/
│   ├── DataCollection.ipynb       # Scripts for corpus downloading and n-gram extraction
│   ├── MWECandidateExtraction.ipynb # Scripts for candidate MWE extraction
│   ├── FilterExistingMWEs.ipynb   # Filtering existing MWEs using pre-downloaded corpora
│   ├── LLMAsAJudge.ipynb          # Prompting LLMs to annotate MWEs
│
├── papers/
│   ├── MEMES-Unidive-Submission.pdf # Accepted submission for UniDive 3rd General Meeting (HUN-REN)
│
└── README.md
```

---

## TODOs

1. **Improve Text Cleaning Pipeline:** Enhance the pre-processing steps for noisy social media text.
2. **Bootstrap an MWE Tagger:** Train a small model or fine-tune an LLM to accelerate MWE tagging.
3. **Evaluate Approaches:** Compare and combine alternative MWE detection techniques for better accuracy.
4. **Optimize Framework:** Streamline the human-in-the-loop system for efficient MWE identification.

---

## Tools and Techniques

- **Statistical Measures:** PMI, STC, SII, Dice Coefficient, LogDice, etc.
- **LLMs:** Used for initial annotation and judgment of MWEs.
- **Corpora:** PARSEME1.3, MWE-CWI, Streusle, EPIE.
- **Python:** Primary programming language for scripts and notebooks.

---

## Citation

If you use this framework in your work, please cite the following:

**MEMES: Mining Emerging Multi-word Expressions from Social Media**
Accepted submission for the UniDive 3rd General Meeting, HUN-REN Hungarian Research Centre for Linguistics, Hungary, Budapest, 29-30 January 2025.

---

## Contact

For questions or collaboration inquiries, please contact:

- **[Nishan Chatterjee, Antoine Doucet, Senja Pollak]**
  [[nchatter@etudiant.univ-lr.fr](mailto:nchatter@etudiant.univ-lr.fr)]
  [Institute Jožef Stefan, Slovenia, and the University of La Rochelle, France]

---

Happy Mining!
