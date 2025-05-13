# Analysing Regressions in Reading with Surprisal and Saliency

This repository contains the code for the following paper: 

Adrielli Tina Lopes Rego, Alline Nogueira, Joshua Snell and Martijn Meeter (2025). _A Prediction-based Approach to Regressions in Reading_. [Manuscript in preparation] 

**Abstract**

"What makes us go back in a text, where do we re-fixate, and why? These questions remain challenging due to the great variety of regressions, i.e. backward movements of the eyes through text. One prevailing hypothesis is that regressions reflect comprehension processes, that is, readers selectively go back in text to reanalyze the textual input. Here we investigate whether surprisal and saliency derived from large language models can predict when backward saccades occur and to where. Surprisal is a measure of how (un)expected a word is given its context and may be interpreted to reflect the difficulty of integrating the fixated word into the mental representation built from the previously read context. Saliency is a measure of how relevant each word is in the previously read context to the prediction of the upcoming word. We hypothesized that words that are unexpected given the context (i.e. surprisal) may trigger regressions to earlier parts of the text, and where the eyes land may be the most relevant parts of the text to the prediction of the surprising word (i.e. saliency). Across two English corpora of eye movements and two monolingual large language models, we found that less surprising words are more likely to trigger a regression, while more surprising words, as well as more salient words relative to the regression source, are more likely to be the target of a regression. Our results suggest that upon increasing confidence, readers tend to go back to the more difficult words, which are also more relevant for the correct prediction of the word where the regression initiated. All in all, prediction-based, language model estimates of surprisal and saliency capture language comprehension processes to some extent, particularly when the need for further comprehension arises and where it resides during reading. The relevance of this study is two-folded: testing how well language models reflect human language behavior may reveal their potential and limitations as models of the language brain, and successfully predicting regressions in reading may advance our understanding about the interaction between oculomotor behavior and reading comprehension."

## 1. Folder structure

The folder "src" contains all the scripts needed to re-run the experiments and the analyses reported in the paper.

- **main.py**: creates datasets with surprisal and saliency values, and datasets for analysing regression triggering and landing.
- **pre_process_corpus.py**: processes file with the corpus texts to generate dataset with each word as a row, and processes fixation report to add variables for analysis.
- **compute_surprisal.py**: takes the file with each word as a row and computes the surprisal value for each word.
- **compute_saliency.py**: takes the file with each word as row and computes the saliency for each word relative to each other text word.
- **post_process_regression.py**: takes the generated saliency values and the pre-processed fixation report and generates the dataset for the regression landing analysis.
- **stats_analysis.R**: R script with all the statistical analysis reported in the paper.

## 2. How to re-run experiments

In order to re-run the experiments, make sure the corpus files are added to the project directory the relative filepaths in the code point to the respective locations. 

For MECO, add files "join_fix_trimmed.rda", which contains the fixation report, "supp texts.csv", which contains the trial passages, and "wordlist_meco.csv", which contains the frequency values per word in the corpus. These files are available in the folder "release 1.0/version 1.2" in the OSF repository (https://osf.io/3527a/) linked to the MECO paper (Siegelman et al, 2022).

For Provo, add files "Provo_Corpus-Additional_Eyetracking_Data-Fixation_Report.csv", which contains the fixation report, "Provo_Corpus-Predictability_Norms.csv", which contains the trial passages, and SUBTLEX_UK.txt, which contains the frequency values per word in the corpus. The Provo files are available in the OSF repository (https://osf.io/sjefs/) linked to the Provo paper. The SUBTLEX-UK frequency resource can be found here. 

After you have data to be processed, open `main.py`, make sure the filepaths and other experiment settings (e.g. language model name) are correct, and then simply run it. 

Finally, with all the datasets needed for analysis being generated, run `stats_analysis.R` to reproduce the results reported in the paper.

## References
Luke, S. G., & Christianson, K. (2018). The Provo Corpus: A large eye-tracking corpus with predictability norms. Behavior research methods, 50, 826-833.
Siegelman, N., Schroeder, S., Acartürk, C., Ahn, H. D., Alexeeva, S., Amenta, S., ... & Kuperman, V. (2022). Expanding horizons of cross-linguistic research on reading: The Multilingual Eye-movement Corpus (MECO). Behavior research methods, 54(6), 2843-2863.