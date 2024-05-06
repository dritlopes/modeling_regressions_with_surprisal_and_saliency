from pre_process_corpus import pre_process_corpus
from compute_surprisal import calculate_surprisal_values
import pandas as pd

texts_filepath = "../data/MECO/supp texts.csv"
# corpus_filepath = '../data/MECO/trials_eng_df.csv'
fixation_filepath = '../data/MECO/joint_fix_trimmed.rda'
# fixation_filepath = '../data/MECO/fixation_report.csv'

fixation_df, words_df = pre_process_corpus(texts_filepath, fixation_filepath)
# fixation_df = pd.read_csv(fixation_filepath)
# words_df = pd.read_csv('../data/MECO/words_df.csv')
surprisal_df = calculate_surprisal_values(words_df, 'MECO', 'gpt2')
meco_surprisal_df = pd.merge(fixation_df, surprisal_df[['trialid', 'ianum', 'surprisal']], how='left', on=['trialid', 'ianum'])
meco_surprisal_df.to_csv('../data/MECO/surprisal_fixation_df.csv')
# df_regression = calculate_mean_ia_regression(meco_surprisal_df)
# meco_surprisal_df = surprisal_df.merge(df_regression[['trialid', 'ianum', 'reg.out']], how='left', on=['trialid', 'ianum'], suffixes=('', '_mean'))