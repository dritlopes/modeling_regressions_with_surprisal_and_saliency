from pre_process_corpus import pre_process_corpus
from compute_surprisal import calculate_surprisal_values
from compute_saliency import calculate_saliency_values
from src.post_process_regression import create_incoming_regression_df, create_outgoing_regression_df

# eye-tracking corpus name
corpus = 'Provo'
# each row is a trial text
texts_filepath = '../data/Provo/raw/Provo_Corpus-Predictability_Norms.csv' # '../data/MECO/raw/supp texts.csv'
# original fixation report
corpus_filepath = '../data/Provo/raw/Provo_Corpus-Additional_Eyetracking_Data-Fixation_Report.csv' # '../data/MECO/raw/joint_fix_trimmed.csv'
# each row is a text word
words_filepath = '../data/Provo/processed/words_en_df.csv' # '../data/MECO/processed/words_en_df.csv'
# fixation report with extra variables for analysis
eye_filepath = '../data/Provo/processed/fixation_en_df.csv' # '../data/MECO/processed/fixation_en_df.csv'
# language model name
model = "gpt2" # gpt2
# word frequency resource
frequency_filepath = '../data/Provo/raw/SUBTLEX_UK.txt' # '../data/MECO/raw/wordlist_meco.csv'

# Generate datasets to add surprisal and saliency
print(f'Pre-processing the fixation dataset and creating a word dataset...')
fixation_df, words_df = pre_process_corpus(texts_filepath=texts_filepath,
                                            corpus_filepath=corpus_filepath,
                                            eye_filepath=eye_filepath,
                                            words_filepath=words_filepath,
                                            frequency_filepath=frequency_filepath,
                                            corpus=corpus)

print(f'-------Language Model: {model}-------')
print('Extracting surprisal values per text word...')
surprisal_filepath = f"../data/{corpus}/processed/surprisal_{model}.csv"
surprisal_df = calculate_surprisal_values(words_df, corpus, model)
surprisal_df.to_csv(surprisal_filepath, index=False)

print('Creating data frame for analysis of outgoing regressions...')
surprisal_eye_filepath = f'../data/{corpus}/processed/surprisal_{model}_fixation.csv'
surprisal_fixation_df = create_outgoing_regression_df(surprisal_df, fixation_df)
surprisal_fixation_df.to_csv(surprisal_eye_filepath, index=False)

print('Extracting saliency values relative to each text word...')
saliency_filepath = f'../data/{corpus}/processed/saliency_{model}.csv'
importance_df = calculate_saliency_values(words_df, model)
importance_df.to_csv(saliency_filepath)

print('Creating data frame for analysis of incoming regressions...')
saliency_eye_filepath = f'../data/{corpus}/processed/saliency_{model}_fixation.csv'
saliency_fixation_df = create_incoming_regression_df(surprisal_fixation_df, importance_df)
saliency_fixation_df.to_csv(saliency_eye_filepath, index=False)
