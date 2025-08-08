# Analysing Regressions in Reading with Surprisal and Saliency

This repository contains the code for a paper in prep.

**Abstract**

"During reading, what makes us regress – i.e., go back in a text instead of going forward? One prevailing hypothesis is that regressions reflect comprehension processes, that is, readers selectively regress to reanalyze the text. Here we investigate whether surprisal and saliency derived from large language models can predict the initiation and destination of regressions. Surprisal is a measure of how (un)expected a word is given its context and may be interpreted to reflect the difficulty of integrating the fixated word into the mental representation built from the previously read context. Saliency is a measure of how relevant each word is in the previously read context to the prediction of the upcoming word. We hypothesized that surprising words are the source of regressions, and that the eyes land on the most salient word, i.e., the word that best predicts the regression source. Across two English corpora of eye movements and two monolingual large language models, we confirmed our hypothesis with respect to the destination of regressions, with salient predictors being more likely to be the target of a regression. However, our hypothesis for the initiation of regressions was not confirmed, as less surprising words were more likely to trigger a regression. In addition, more surprising words were more likely to be the regression target. Our results suggest that readers tend to regress from easy words to difficult, information-rich words. These findings may provide more sophisticated accounts of comprehension-driven regressions in computational models of reading."

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

For MECO, add files "join_fix_trimmed.rda", which contains the fixation report, "supp texts.csv", which contains the trial passages, and "wordlist_meco.csv", which contains the frequency values per word in the corpus. These files are available in the folder "release 1.0/version 1.2" in the [OSF repository](https://osf.io/3527a/) linked to the MECO paper (Siegelman et al., 2022).

For Provo, add files "Provo_Corpus-Additional_Eyetracking_Data-Fixation_Report.csv", which contains the fixation report, "Provo_Corpus-Predictability_Norms.csv", which contains the trial passages, and SUBTLEX_UK.txt, which contains the frequency values per word in the corpus. The Provo files are available in the [OSF repository](https://osf.io/sjefs/) linked to the Provo paper (Luke & Christianson, 2018). The SUBTLEX-UK frequency resource (Van Heuven et al., 2014) can be found [here](https://osf.io/zq49t/). 

After you have data to be processed, open `main.py`, make sure the filepaths and other experiment settings (e.g. language model name) are correct, and then simply run it. 

Finally, with all the datasets needed for analysis being generated, run `stats_analysis.R` to reproduce the results reported in the paper.

If you'd like to have access to the already pre-processed data with surprisal and saliency values for analysis, please contact **a.t.lopesrego@vu.nl**.

## References
Luke, S. G., & Christianson, K. (2018). The Provo Corpus: A large eye-tracking corpus with predictability norms. Behavior research methods, 50, 826-833.
Siegelman, N., Schroeder, S., Acartürk, C., Ahn, H. D., Alexeeva, S., Amenta, S., ... & Kuperman, V. (2022). Expanding horizons of cross-linguistic research on reading: The Multilingual Eye-movement Corpus (MECO). Behavior research methods, 54(6), 2843-2863.
Van Heuven, W. J., Mandera, P., Keuleers, E., & Brysbaert, M. (2014). SUBTLEX-UK: A new and improved word frequency database for British English. Quarterly journal of experimental psychology, 67(6), 1176-1190.
