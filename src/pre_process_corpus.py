import pandas as pd
import rdata
import os

###############################################################
# Preparing CSV files from original texts
def convert_rdm_to_csv(original_filepath):

    converted = rdata.read_rda(original_filepath)
    converted_key = list(converted.keys())[0]
    df = pd.DataFrame(converted[converted_key])
    dir = os.path.dirname(original_filepath)
    fixation_filepath = f'{dir}/fixation_report.csv'
    df.to_csv(fixation_filepath)

    return fixation_filepath

def create_original_texts_dataframe(file_path):

    # texts_df = pd.read_csv("/Users/anm/code/RegressionMECO/data/MECO/supp texts.csv", sep=',')
    texts_df = pd.read_csv(file_path, sep=',')
    texts_df.drop(['Unnamed: 13', 'Unnamed: 14'], axis=1, inplace=True)
    texts_df.columns = ['lang', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

    eng_filter = (texts_df['lang'] == 'English') #filter for only the texts in English
    eng_texts_df = texts_df.loc[eng_filter]

    trialid_raw_df = eng_texts_df.stack().astype(str).reset_index(level=1)
    trialid_raw_df.rename(columns={'level_1':'trialid', 0:'text'}, inplace=True)

    trialid_raw_df = trialid_raw_df.reset_index(drop=False)

    trialid_raw_df.drop([0], inplace=True)

    trialid_raw_df.drop(['index'], axis=1, inplace=True)

    # trialid_raw_df.to_csv('/Users/anm/code/RegressionMECO/data/MECO/trialid_raw_df.csv', sep='\t')
    return trialid_raw_df

###############################################################
# Cleaning the original texts (trialid_raw_df.csv)
def clean_original_texts(trialid_raw_df: pd) -> pd.DataFrame:
    trialid_cleaning_df = trialid_raw_df.copy()

    # 1-4
    # replace with "space" the "\\n" at the beginning of a word
    trialid_cleaning_df["text"] = trialid_cleaning_df["text"].str.replace(" \\n", " ") #replace only the "space+\\n, these ones are in the beginning of words

    # 2-4
    # replace with "space" the "\\n" between words as "word\\nword"
    trialid_cleaning_df["text"] = trialid_cleaning_df["text"].str.replace("\\n", " ") #replace the \\n between words

    # 3-4
    # when "word-word" add a space after first word, then the words would be separated equally
    trialid_cleaning_df["text"] = trialid_cleaning_df["text"].str.replace("-", "- ")

    # 4-4
    # replace with a empty string all the quotation marks
    trialid_cleaning_df["text"] = trialid_cleaning_df["text"].str.replace('"','')

    # create a cleaned dataframe
    # trialid_cleaning_df.to_csv('/Users/anm/code/RegressionMECO/data/MECO/trialid_df.csv', sep='\t')
    return trialid_cleaning_df

###############################################################
# Transforming original texts file into words file (words_df)
def create_ianum_from_original_texts(texts_df: pd):
    # trialid_df = pd.read_csv("/Users/anm/code/RegressionMECO/data/MECO/trialid_df.csv", sep='\t', index_col=0) #with index_col=0 there will not be an unnamed column

    trialid = []
    text = []
    ia_new = []
    ianum_new = []

    # to interact between a list of rows
    for _, row in texts_df.iterrows():
        # transform the text into a list of words
        text_words = row['text'].split()

        # transform text_words into a dataframe
        for i in range(0, len(text_words)):
            trialid.append(row['trialid'])
            # print(trialid)
            text.append(row['text'])
            # print(text)
            ianum_new.append(i + 1)
            # print(ianum_new)
            ia_new.append(text_words[i])
            # print(ia_new)

    # adding it to dataframe
    words_df = pd.DataFrame({'trialid': trialid,
                            'texts': text,
                            'ianum': ianum_new,
                            'ia': ia_new})

    # words_df.to_csv('/Users/anm/code/RegressionMECO/data/MECO/words_df.csv', sep='\t')
    return words_df

#function that creates a new ID for each word, based on the new ID created in the words_df and combined with the old ID in MECO dataset
def find_right_ianum(words_df, meco_row):

    filter = ((words_df['trialid'] == meco_row['trialid'])  # same text
              & (words_df['ianum'] <= meco_row['ianum'])  # AL: smaller or same ID
              & (words_df['ia'] == meco_row['ia'])  # same word
              )
    df_filtered = words_df[filter]
    r = df_filtered[
        'ianum']  # list with all ianum from the reference table (words_df) that matched the filter > than row
    # print(meco_row[['ianum','ia']])
    # print(df_filtered[['ianum','ia']])
    new_id = meco_row['ianum']  # return original ianum from meco, for the case that it's corrected means same ianum
    if len(r) == 1 and r.tolist()[0] != meco_row['ianum']:
        new_id = r.tolist()[0]
    elif len(r) > 1:
        new_id = r.max()    # return for the case that the ianum in meco is smaller
    elif len(r) == 0:
        filter = ((words_df['trialid'] == meco_row['trialid'])  # same text
                  & (words_df['ianum'] > meco_row['ianum'])  # AL: greater ID (e.g. text 12)
                  & (words_df['ia'] == meco_row['ia'])  # same word
                  )
        df_filtered = words_df[filter]
        r = df_filtered['ianum']
        new_id = r.min()    # return for the case that the ianum in meco is greater
    # print(new_id)
    # print()
    return new_id

def assign_ianum_to_meco(words_df: pd.DataFrame, meco_df: pd.DataFrame):

    reg_df_corrected = meco_df.copy()
    reg_df_corrected['ianum_meco'] = reg_df_corrected['ianum'] # using the same column to include the correct ianum (it's not creating a new empty variable, then no need to rename the columns later)
    reg_df_corrected['ianum'] = meco_df.apply(lambda row: find_right_ianum(words_df, row), axis=1)

    # reg_df_corrected.to_csv('/Users/anm/code/RegressionMECO/data/MECO/ianum_corrected_df.csv', sep='\t')
    return reg_df_corrected

###############################################################
#Correcting the ID of each word in MECO English dataset (trials_eng_df)

def clean_words_meco(meco_df: pd.DataFrame):
    # data_frame = pd.read_csv("/Users/anm/code/RegressionMECO/data/MECO/word_data.csv", sep='\t')

    # MECO English dataset

    # trials_eng_df = pd.read_csv("/Users/anm/code/RegressionMECO/data/MECO/trials_eng_df.csv", sep='\t', index_col=0) #with index_col=0 there will not be a unnamed column)
    trials_eng_df = meco_df.copy()

    # 1-3
    # remove quotation marks
    trials_eng_df["ia"] = trials_eng_df["ia"].str.replace('"','')

    # 2-3
    # remove the word "officialidentification" in trialid 12.0 ianum 16.0
    trials_eng_df = trials_eng_df[trials_eng_df["ia"].str.contains("officialidentification")==False]

    # 3-3
    # remove ia "trucks,cars" in trialid 12.0 ianum 29.0
    trials_eng_df = trials_eng_df[trials_eng_df["ia"].str.contains("trucks,and")==False]

    # dataset with new IANUM, as reference:
    # words_df = pd.read_csv("/Users/anm/code/RegressionMECO/data/MECO/words_df.csv", sep='\t', index_col=0) #with index_col=0 there will not be a unnamed column)

    return trials_eng_df

def check_alignment(meco_corrected_ianum, words_df):

    # AL: checking which words are not matching between meco_df and words_df
    for index, meco_row in meco_corrected_ianum.iterrows():
        filter = ((words_df['trialid'] == meco_row['trialid'])
                  & (words_df['ianum'] == meco_row['ianum']))
        words_filtered = words_df[filter]
        if len(words_filtered) > 0:
            if meco_row['ia'] != words_filtered['ia'].tolist()[0]:
                print('Same id with different words!')
                print(meco_row[['subid', 'trialid', 'ia', 'ianum']])
                print(words_filtered['ia'].tolist()[0], words_filtered['ianum'].tolist()[0])
                print()

# def compute_distance(word_ids, regression_out, regression_in):
#
#     distances = []
#     counter = 0
#     for word, r_out in zip(word_ids, regression_out):
#         if r_out == 1:
#             regression_in_till_i = [i for r, i in zip(regression_in[:counter], word_ids[:counter]) if r == 1]
#             closest_regression_in = max(regression_in_till_i)
#             distances.append(word - closest_regression_in)
#         else:
#             distances.append(None)
#         counter += 1
#
#     return distances

def compute_distance(word_ids, regression_out, regression_out_to):

    distances = []
    for i, word_id in enumerate(word_ids):
        distance = None
        if regression_out[i] == 1:
            distance = word_id - regression_out_to[i]
        distances.append(distance)

    return distances

def compute_all_distances(fixation_data):

    distances = []

    for id, group in fixation_data.groupby(['uniform_id', 'trialid']):
        # distance = compute_distance(group['ianum'].tolist(), group['reg.out'].tolist(), group['reg.in'].tolist())
        # filter = ((fixation_data['trialid'] == id[1])
        #           & (fixation_data['uniform_id'] == id[0]))
        # fixation_filtered = fixation_data[filter]
        distance = compute_distance(group['ianum'].tolist(), group['reg.out'].tolist(),
                                    group['reg.out.to'].tolist())
        distances.extend(distance)

    return distances

# ######################################################
# When merging the two datasets so far, it's possible to check that there are some problems:
#
# ######################################################
# #FIRST PROBLEM: THERE ARE STILL 82 ROWS WITH EMPTY SURPRISAL VALUES
#
# #checking whether there are empty values for surprisal, besides all the first words
# merge_df[(merge_df['surprisal'].isnull()) & (merge_df['ianum_merge'] > 1)][['trialid', 'ianum', 'ia', 'uniform_id', 'IA_NEW']]
#
# ######################################################
# #SECOND PROBLEM: THERE ARE STILL 3454 ROWS WITH DIFFERENT WORDS AND SAME ID
#
# #checking whether there are different words with the same ianum (major problem of the dataset)
# merge_df[(merge_df['ia'] != merge_df['IA_NEW'])][['trialid', 'ianum', 'ia', 'uniform_id', 'IA_NEW', "ianum_merge"]]
# #and there are 3454 rows for the trialid 1.0, 2.0, 3.0 and (not 12.0 anymore BECAUSE I CORRECTED IT ABOVE)
#
# #FOR WHICH TEXTS?
# merge_df[(merge_df['ia'] != merge_df['IA_NEW'])]['trialid'].unique()
# ######################################################


###############################################################
# Run all the functions in the script
def pre_process_corpus(texts_filepath, fixation_filepath):

    original_df = create_original_texts_dataframe(texts_filepath) # with index_col=0 it will not have an unnamed column)
    original_df = clean_original_texts(original_df)
    words_df = create_ianum_from_original_texts(original_df)

    # meco_df = pd.read_csv(corpus_filepath, sep='\t', index_col=0) # with index_col=0 it will not have an unnamed column)
    # meco_df = clean_words_meco(meco_df)
    # meco_df = meco_df.sort_values(by=['subid','trialid','ianum']) # re-order words by ianum
    # meco_df.to_csv('../data/MECO/meco_df_sorted.csv')

    # meco_corrected_ianum = assign_ianum_to_meco(words_df, meco_df)
    # check_alignment(meco_corrected_ianum, words_df) # check alignment
    # meco_corrected_ianum = pd.read_csv('../data/MECO/meco_corrected_ianum.csv')

    if fixation_filepath.endswith('.rda'):
        fixation_filepath = convert_rdm_to_csv(fixation_filepath)
    fixation_df = pd.read_csv(fixation_filepath)
    fixation_df = fixation_df[['uniform_id',
                               'trialid',
                               'fixid',
                               'dur',
                               'ia',
                               'ianum',
                               'ia.reg.out',
                               'ia.reg.out.to']]

    fixation_df = fixation_df.rename(columns={'ia.reg.out': 'reg.out', 'ia.reg.out.to': 'reg.out.to'})
    # find distance of regressions
    distances = compute_all_distances(fixation_df)
    fixation_df['reg.dist'] = distances
    # meco_corrected_ianum.to_csv('../data/MECO/meco_corrected_ianum.csv')

    # add length and frequency
    fixation_df['length'] = [len(word) for word in fixation_df['ia'].tolist()]

    # add reg word + 1 and word + 2

    fixation_df.to_csv('../data/MECO/fixation_en_df.csv')
    words_df.to_csv('../data/MECO/words_df.csv')

    # return meco_corrected_ianum, words_df
    return fixation_df, words_df

