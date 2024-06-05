from pre_process_corpus import pre_process_corpus
from compute_surprisal import calculate_surprisal_values
import pandas as pd
from run_analysis import run_stats

texts_filepath = "../data/MECO/supp texts.csv" # each row is a trial text
corpus_filepath = '../data/MECO/corpus_en_df.csv' # '../data/MECO/joint_data_trimmed.rda' or '../data/MECO/joint_data_trimmed.csv'
fixation_filepath = '../data/MECO/fixation_en_df.csv' # '../data/MECO/joint_fix_trimmed.csv' '../data/MECO/joint_data_trimmed.rda'
language = 'en'

# Generate datasets to add surprisal and saliency
# Fixation_df is the fixation-based dataset;
# Words_df only contains the words in each trial;
# And corpus_df is the word-based data;
# fixation_df, words_df, corpus_df = pre_process_corpus(texts_filepath,
#                                                       fixation_filepath=fixation_filepath,
#                                                       corpus_filepath=corpus_filepath,
#                                                       language=language)
# fixation_df = pd.read_csv(fixation_filepath)
# words_df = pd.read_csv('../data/MECO/words_en_df.csv')

# Generate surprisal values for each word in each trial text
# surprisal_df = calculate_surprisal_values(words_df, 'MECO', 'gpt2')
# surprisal_df = pd.read_csv('../data/MECO/surprisal_df2.csv', sep='\t')

# Merge the surprisal values with the eye-tracking data
# meco_surprisal_df = pd.merge(fixation_df, surprisal_df[['trialid', 'ianum', 'surprisal']], how='left', on=['trialid', 'ianum'])
# meco_surprisal_df.to_csv('../data/MECO/surprisal_fixation_df.csv')
meco_surprisal_df = pd.read_csv('../data/MECO/surprisal_fixation_df.csv')

# Generate saliency values for each word in each trial text



