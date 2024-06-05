import pandas as pd
import rdata
import os
import numpy as np
import matplotlib.pyplot as plt

###############################################################
# Preparing CSV files from original texts
def convert_rdm_to_csv(original_filepath):

    converted = rdata.read_rda(original_filepath)
    converted_key = list(converted.keys())[0]
    df = pd.DataFrame(converted[converted_key])
    filepath = original_filepath.replace('rda', 'csv')
    df.to_csv(filepath)

    return filepath

def create_original_texts_dataframe(file_path, language):

    # texts_df = pd.read_csv("/Users/anm/code/RegressionMECO/data/MECO/supp texts.csv", sep=',')
    texts_df = pd.read_csv(file_path, sep=',')
    texts_df.drop(['Unnamed: 13', 'Unnamed: 14'], axis=1, inplace=True)
    texts_df.columns = ['lang', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

    if language == 'en': filter = 'English'
    elif language == 'du': filter = 'Dutch'
    lan_filter = (texts_df['lang'] == filter)
    lan_texts_df = texts_df.loc[lan_filter]

    trialid_raw_df = lan_texts_df.stack().astype(str).reset_index(level=1)
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

    # words_df.to_csv('/Users/anm/code/RegressionMECO/data/MECO/words_en_df.csv', sep='\t')
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

def clean_words_meco(meco_df: pd.DataFrame, language):
    # data_frame = pd.read_csv("/Users/anm/code/RegressionMECO/data/MECO/word_data.csv", sep='\t')

    # MECO English dataset

    # trials_eng_df = pd.read_csv("/Users/anm/code/RegressionMECO/data/MECO/trials_en_df.csv", sep='\t', index_col=0) #with index_col=0 there will not be a unnamed column)
    trials_df = meco_df.copy()

    # filter trials for language
    lan_filter = (trials_df['lang'] == language)
    trials_df = trials_df.loc[lan_filter]

    # remove quotation marks
    trials_df["ia"] = trials_df["ia"].str.replace('"','')

    if language == 'en':
        # remove the word "officialidentification" in trialid 12.0 ianum 16.0
        trials_df = trials_df[trials_df["ia"].str.contains("officialidentification")==False]
        # remove ia "trucks,cars" in trialid 12.0 ianum 29.0
        trials_df = trials_df[trials_df["ia"].str.contains("trucks,and")==False]

    return trials_df

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

def add_variables(variables, df, resources, language):

    if 'distance' in variables:
        # find distance of regressions
        distances = compute_all_distances(df)
        df['reg.dist'] = distances
        df["reg.dist"] = df["reg.dist"].fillna(0)
        df['reg.dist.log'] = np.log(np.array(df["reg.dist"].tolist()))

    if 'length' in variables:
        # add length and frequency
        df['length'] = [len(word) for word in df['ia'].tolist()]
        df["length.log"] = np.log(df["length"])

    if 'frequency' in variables and 'frequency' in resources.keys():
        frequency_df = pd.read_csv(resources['frequency'])
        if language == 'en': language = 'english'
        elif language == 'du': language = 'dutch'
        frequency_df = frequency_df[frequency_df['lang'] == language]
        frequency_col = []
        for word in df['ia'].tolist():
            word = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), word))
            if word.isalpha():
                word = word.lower()
            if word in frequency_df['ia_clean'].tolist():
                frequency_col.append(frequency_df['zipf_freq'].tolist()[frequency_df['ia_clean'].tolist().index(word)])
            else:
                frequency_col.append(None)
        df['frequency'] = frequency_col

    if 'word+1' in variables:
        reg_out_plus_one = []
        for id, group in df.groupby(['uniform_id','trialid']):
            reg = [group['reg.out'].tolist()[i+1] if i + 1 < len(group['reg.out'].tolist()) else None for
                   i, reg_out in enumerate(group['reg.out'].tolist())]
            reg_out_plus_one.extend(reg)
        df['reg.out.plus.one'] = reg_out_plus_one

        if 'distance' in variables:
            reg_out_plus_one_dist = []
            for id, group in df.groupby(['uniform_id', 'trialid']):
                reg = [group['reg.dist.log'].tolist()[i + 1] if i + 1 < len(group['reg.dist.log'].tolist()) else None for
                       i, reg_out in enumerate(group['reg.dist.log'].tolist())]
                reg_out_plus_one_dist.extend(reg)
            df['dist.log.plus.one'] = reg_out_plus_one_dist

    if 'word+2' in variables:
        reg_out_plus_two = []
        for id, group in df.groupby(['uniform_id', 'trialid']):
            reg = [group['reg.out'].tolist()[i + 2] if i + 2 < len(group['reg.out'].tolist()) else None for
                   i, reg_out in enumerate(group['reg.out'].tolist())]
            reg_out_plus_two.extend(reg)
        df['reg.out.plus.two'] = reg_out_plus_two

        if 'distance' in variables:
            reg_out_plus_two_dist = []
            for id, group in df.groupby(['uniform_id', 'trialid']):
                reg = [group['reg.dist.log'].tolist()[i + 2] if i + 2 < len(group['reg.dist.log'].tolist()) else None for
                       i, reg_out in enumerate(group['reg.dist.log'].tolist())]
                reg_out_plus_two_dist.extend(reg)
            df['dist.log.plus.two'] = reg_out_plus_two_dist

    return df

def pre_process_corpus_data(corpus_filepath, language, words_df):

    if corpus_filepath.endswith('.rda'):
        corpus_filepath = convert_rdm_to_csv(corpus_filepath)
    corpus_df = pd.read_csv(corpus_filepath, index_col=0)
    corpus_df = clean_words_meco(corpus_df, language)
    corpus_df = corpus_df.sort_values(by=['subid', 'trialid', 'ianum'])  # re-order words by ianum
    corpus_df.to_csv(f'../data/MECO/meco_{language}_df_sorted.csv')
    corpus_df = assign_ianum_to_meco(words_df, corpus_df)
    check_alignment(corpus_df, words_df)  # check alignment

    return corpus_df

def find_x_changes(fixation_df):

    x_changes = []
    count_line_above = 0
    count_forward = 0
    for id, group in fixation_df.groupby(['uniform_id','trialid']):
        # select regressions
        reg_out_rows = group[group['reg.out'] == 1]
        for i, reg_out_to in enumerate(reg_out_rows['reg.out.to'].tolist()):
            # find row with destination of regression
            reg_in_row = group[(group['ianum'] == reg_out_to)
                               & (group['reg.in'] == 1)
                               & (group['fixid'] > reg_out_rows['fixid'].tolist()[i])]
            if not reg_in_row.empty:
                index = 0
                if len(reg_in_row['line.change'].tolist()) > 1:
                    index = reg_in_row['fixid'].tolist().index(min(reg_in_row['fixid'].tolist()))
                # was the regression to the line before?
                if reg_in_row['line.change'].tolist()[index] == -1:
                    count_line_above += 1
                    # forward x?
                    # find out 90th percentile of x changes for forward saccades
                    reg_out_x = reg_out_rows['xs'].tolist()[i]
                    reg_in_x = reg_in_row['xs'].tolist()[index]
                    diff = reg_in_x - reg_out_x
                    if diff > 0:
                        count_forward += 1
                        x_changes.append(diff)

    return x_changes

def pre_process_fixation_data(fixation_filepath, language):

    if fixation_filepath.endswith('.rda'):
        fixation_filepath = convert_rdm_to_csv(fixation_filepath)

    fixation_df = pd.read_csv(fixation_filepath)
    fixation_df = fixation_df[(fixation_df['lang'] == language)]
    fixation_df = fixation_df.loc[:, ~fixation_df.columns.str.contains('^Unnamed')]
    fixation_df = fixation_df[['uniform_id',
                               'trialid',
                               'fixid',
                               'ia',
                               'ianum',
                               'ia.firstskip',
                               'dur',
                               'xs',
                               'ys',
                               'ym',
                               'line',
                               'line.change',
                               'sac.in',
                               'sac.out',
                               'ia.reg.out',
                               'ia.reg.out.to',
                               'ia.reg.in']]

    fixation_df = fixation_df.rename(columns={'ia.reg.out': 'reg.out',
                                              'ia.reg.out.to': 'reg.out.to',
                                              'ia.reg.in': 'reg.in',
                                              'ia.firstskip': 'skip'})


    # Remove first fixation of a trial that is also triggers a regression (likely noise)
    fixation_df = fixation_df.drop(fixation_df[(fixation_df['fixid'] == 1) & (fixation_df['reg.out'] == 1)].index)

    # Remove regression that is subsequently followed by another regression (likely noise)
    indices_to_drop = []
    for id, group in fixation_df.groupby(['uniform_id','trialid']):
        reg_in = group['reg.in'].tolist()
        for i, reg in enumerate(reg_in):
            if i + 1 < len(reg_in):
                if reg_in[i] == 1 and reg_in[i+1] == 1:
                    indices_to_drop.append(group.iloc[i].name)
    fixation_df.drop(indices_to_drop, inplace=True)

    # Deal with noise in regressions to upper lines
    # we assume regression was unintended when it was to the previous line (<y) and forward (>x),
    # and the number of pixels that the saccade goes forward (ie., change in x) is below the 90th
    # percentile for x changes for forward saccades
    # check distance of reg not in the same line and remove if to line above and close to origin
    indices_to_drop = []
    x_changes = find_x_changes(fixation_df)
    threshold_x_change = np.percentile(x_changes, 90)
    for id, group in fixation_df.groupby(['uniform_id','trialid']):
        # select regressions
        reg_out_rows = group[group['reg.out'] == 1]
        for i, reg_out_to in enumerate(reg_out_rows['reg.out.to'].tolist()):
            # find row with destination of regression
            reg_in_row = group[(group['ianum'] == reg_out_to)
                               & (group['reg.in'] == 1)
                               & (group['fixid'] > reg_out_rows['fixid'].tolist()[i])]
            if not reg_in_row.empty:
                index = 0
                if len(reg_in_row['line.change'].tolist()) > 1:
                    index = reg_in_row['fixid'].tolist().index(min(reg_in_row['fixid'].tolist()))
                # was the regression to the line before?
                if reg_in_row['line.change'].tolist()[index] == -1:
                    # forward x?
                    reg_out_x = reg_out_rows['xs'].tolist()[i]
                    reg_in_x = reg_in_row['xs'].tolist()[index]
                    diff = reg_in_x - reg_out_x
                    if diff > 0:
                        # close x (< 90th percentile of x changes for forward saccades)?
                        if diff < threshold_x_change:
                            # remove regression
                            indices_to_drop.append(reg_out_rows.iloc[i].name)
    # distribution of regressions in terms of line distance
    # reg_out_rows = fixation_df[fixation_df['reg.in'] == 1]
    # reg_out_rows['line.change'].value_counts().to_csv('dist_reg_line.csv')
    fixation_df.drop(indices_to_drop, inplace=True)

    return fixation_df

###############################################################

def pre_process_corpus(texts_filepath, fixation_filepath=None, corpus_filepath=None, language='en'):

    words_df, fixation_df, corpus_df = None, None, None

    # original_df = create_original_texts_dataframe(texts_filepath, language) # with index_col=0 it will not have an unnamed column)
    # original_df = clean_original_texts(original_df)
    # words_df = create_ianum_from_original_texts(original_df)
    # words_df.to_csv(f'../data/MECO/words_{language}_df.csv')
    resources = {'frequency': '../data/MECO/wordlist_meco.csv'}
    # words_df = pd.read_csv(f'../data/MECO/words_{language}_df.csv')

    # if corpus_filepath:
    #     # corpus_df = pre_process_corpus_data(corpus_filepath, language, words_df)
    #     corpus_df = pd.read_csv(corpus_filepath)
    #     # corpus_df = corpus_df.loc[:, ~corpus_df.columns.str.contains('^Unnamed')]
    #     corpus_df = add_variables(['length', 'frequency'], corpus_df, resources, language)
    #     corpus_df.to_csv(f'../data/MECO/corpus_{language}_df.csv', index=False)

    if fixation_filepath:
        # fixation_df = pd.read_csv(f'../data/MECO/fixation_{language}_df.csv')
        fixation_df = pre_process_fixation_data(fixation_filepath, language)
        fixation_df = add_variables(['distance', 'length', 'frequency', 'word+1', 'word+2'], fixation_df, resources, language)
        fixation_df.to_csv(f'../data/MECO/fixation_{language}_df.csv', index=False)

        # num_bins = 10
        # min_val = fixation_df['reg.dist'].min()
        # max_val = fixation_df['reg.dist'].max()
        # bin_size = (max_val - min_val) // num_bins
        # bins = np.arange(min_val, max_val, bin_size)
        # fixation_df['reg.dist'].astype(int).value_counts(bins=bins).to_csv('counts3.csv')

    # return meco_corrected_ianum, words_df
    return fixation_df, words_df, corpus_df

