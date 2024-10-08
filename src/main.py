import pandas as pd
from pre_process_corpus import pre_process_corpus
from compute_surprisal import calculate_surprisal_values
from compute_saliency import calculate_saliency_values
from post_process_regression import create_regression_saliency_df

experiment_set_up = {'language': 'en',
                     'texts_filepath': '../data/MECO/supp texts.csv', # each row is a trial text
                     'fixation_report_filepath': '../data/MECO/joint_fix_trimmed.rda', # original fixation report
                     'words_filepath': '../data/MECO/words_en_df.csv',  # each row is a text word
                     'fixation_filepath': '../data/MECO/fixation_en_df.csv', # fixation report with extra variables for analysis
                     'models': ["gpt2-large"], # gpt2, gpt2-large, meta-llama/Llama-2-7b-hf
                     'frequency_filepath': '../data/MECO/wordlist_meco.csv'} # resource with frequency value for each text word

# Generate datasets to add surprisal and saliency
print(f'Pre-processing the fixation dataset and creating a word dataset...')
fixation_df, words_df = pre_process_corpus(texts_filepath=experiment_set_up['texts_filepath'],
                                            fixation_report_filepath=experiment_set_up['fixation_filepath'],
                                            fixation_filepath=experiment_set_up['fixation_filepath'],
                                            words_filepath=experiment_set_up['words_filepath'],
                                            frequency_filepath=experiment_set_up['frequency_filepath'],
                                            language=experiment_set_up['language'])

for model in experiment_set_up['models']:
    print(f'-------Language Model: {model}-------')

    print('Extracting surprisal values per text word...')
    surprisal_df = calculate_surprisal_values(words_df, 'MECO', model)

    print('Merging the surprisal values with the regression data.')
    meco_surprisal_df = pd.merge(fixation_df, surprisal_df[['trialid', 'ianum', 'surprisal']], how='left', on=['trialid', 'ianum'])
    meco_surprisal_df.to_csv(f'../data/MECO/surprisal_{model}_fixation_df.csv')

    print('Extracting saliency values relative to each text word...')
    importance_df = calculate_saliency_values(words_df, model)

    print('Merging the saliency values with the regression data.')
    create_regression_saliency_df(model, fixation_df, importance_df)