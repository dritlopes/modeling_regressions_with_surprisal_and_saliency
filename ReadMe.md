# Analysing Regressions in Reading with Surprisal and Saliency

This repository contains the code for the following paper: 

Adrielli Tina Lopes Rego, Alline Nogueira and Martijn Meeter (2024). _Language Models capture where readers look back to in text_. [Manuscript in preparation] 

**Abstract**

"What makes us go back in a text, where do we re-fixate, and why? These questions remain challenging due to the great variety of regressions, i.e. backward movements of the eyes through text. One prevailing hypothesis is that regressions reflect comprehension difficulty, that is, readers selectively go back in text to reanalyze the textual input. Here we investigate whether surprisal and saliency derived from large language models can respectively predict when backward saccades occur and to where. Surprisal is a measure of how (un)expected a word is given its context and may be interpreted to reflect the difficulty of integrating the fixated word into the mental representation built from the previously read context. Saliency is a measure of how relevant each word is in the previously read context to the prediction of the upcoming word. We hypothesized that words that are unexpected given the context (i.e. surprisal) may trigger regressions to earlier parts of the text, and where the eyes land may be the most relevant parts of the text to the prediction of the surprising word (i.e. saliency). We found a positive effect of surprisal on the chance of being the source of the regression at the upcoming fixation, as well as a positive effect of saliency on the chance of being the target of the regression. Our results support the link between regression and repair, particularly associated with error costs from predictive processing. Successfully predicting regressions in reading may advance our understanding about the interaction between oculomotor behavior and reading comprehension."

## 1. Folder structure

The folder "src" contains all the scripts needed to re-run the experiments and the analyses reported in the paper.

- **main.py**: creates datasets with surprisal and saliency values, and datasets for analysing regression triggering and landing.
- **pre_process_corpus.py**: processes file with the corpus texts to generate dataset with each word as a row, and processes fixation report to add variables for analysis.
- **compute_surprisal.py**: takes the file with each word as a row and computes the surprisal value for each word.
- **compute_saliency.py**: takes the file with each word as row and computes the saliency for each word relative to each other text word.
- **post_process_regression.py**: takes the generated saliency values and the pre-processed fixation report and generates the dataset for the regression landing analysis.
- **stats_analysis.R**: R script with all the statistical analysis reported in the paper.
- **visualise_results.ipynb**: generates the graphs seen in the paper.

## 2. How to run the experiments

In order to re-run the experiments, you first need to create a folder called "data" in the root directory of the project (same level as the folder "src") and add the MECO corpus files "join_fix_trimmed.rda", which contains the fixation report, "supp texts.csv", which contains the trial passages, and "wordlist_meco.csv", which contains the frequency values per word in the corpus. These files are available in the folder "release 1.0/version 1.2" in the OSF repository (https://osf.io/3527a/) linked to the MECO paper (Siegelman et al, 2022).

After you have data to be processed, open `main.py`, make sure the filepaths and other experiment settings (e.g. language) are correct, and then simply run it. 

Finally, with all the datasets needed for analysis being generated, run `stats_analysis.R` to reproduce the results reported in the paper.

If you would like to reproduce the graphs seen in the paper, run the notebook `visualise_results.ipynb`.

## References
Siegelman, N., Schroeder, S., Acartürk, C., Ahn, H. D., Alexeeva, S., Amenta, S., ... & Kuperman, V. (2022). Expanding horizons of cross-linguistic research on reading: The Multilingual Eye-movement Corpus (MECO). Behavior research methods, 54(6), 2843-2863.