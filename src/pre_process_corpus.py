import pandas as pd
import rdata
import os
import numpy as np
import matplotlib.pyplot as plt

### Functions to clean and generate file with each word as row, from MECO file with texts (supp texts.csv)
def create_original_texts_dataframe(file_path, language):

    texts_df = pd.read_csv(file_path, sep=',')
    texts_df.drop(['Unnamed: 13', 'Unnamed: 14'], axis=1, inplace=True)
    texts_df.columns = ['lang', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

    filter = ''
    if language == 'en': filter = 'English'
    elif language == 'du': filter = 'Dutch'
    lan_filter = (texts_df['lang'] == filter)
    lan_texts_df = texts_df.loc[lan_filter]

    trialid_raw_df = lan_texts_df.stack().astype(str).reset_index(level=1)
    trialid_raw_df.rename(columns={'level_1':'trialid', 0:'text'}, inplace=True)
    trialid_raw_df = trialid_raw_df.reset_index(drop=False)
    trialid_raw_df.drop([0], inplace=True)
    trialid_raw_df.drop(['index'], axis=1, inplace=True)

    return trialid_raw_df

def clean_original_texts(trialid_raw_df: pd) -> pd.DataFrame:

    trialid_cleaning_df = trialid_raw_df.copy()

    # replace with "space" the "\\n" at the beginning of a word
    trialid_cleaning_df["text"] = trialid_cleaning_df["text"].str.replace(" \\n", " ")
    # replace with "space" the "\\n" between words as "word\\nword"
    trialid_cleaning_df["text"] = trialid_cleaning_df["text"].str.replace("\\n", " ")
    # when "word-word" add a space after first word, then the words would be separated equally
    trialid_cleaning_df["text"] = trialid_cleaning_df["text"].str.replace("-", "- ")
    # replace with a empty string all the quotation marks
    trialid_cleaning_df["text"] = trialid_cleaning_df["text"].str.replace('"','')

    return trialid_cleaning_df

def create_ianum_from_original_texts(texts_df: pd):

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

###############################################################

### Functions to pre-process fixation report

def convert_rdm_to_csv(original_filepath):

    converted = rdata.read_rda(original_filepath)
    converted_key = list(converted.keys())[0]
    df = pd.DataFrame(converted[converted_key])
    filepath = original_filepath.replace('rda', 'csv')
    df.to_csv(filepath)

    return filepath

def remove_subsequent_regressions(fixation_df):

    indices_to_drop = []
    for id, group in fixation_df.groupby(['uniform_id', 'trialid']):
        reg_in = group['reg.in'].tolist()
        for i, reg in enumerate(reg_in):
            if i + 1 < len(reg_in):
                if reg_in[i] == 1 and reg_in[i + 1] == 1:
                    indices_to_drop.append(group.iloc[i].name)
    fixation_df.drop(indices_to_drop, inplace=True)

    return fixation_df

def find_x_changes(fixation_df):

    # find changes to previous line

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

def remove_close_reg_upper_line(fixation_df):

    # we assume regression was unintended when it was to the previous line (<y) and forward (>x),
    # and the number of pixels that the saccade goes forward (ie., change in x) is below the
    # 90th percentile for x changes for forward saccades
    # check distance of reg not in the same line and remove if to line above and close to origin

    indices_to_drop = []
    x_changes = find_x_changes(fixation_df)
    threshold_x_change = np.percentile(x_changes, 90)
    for id, group in fixation_df.groupby(['uniform_id', 'trialid']):
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

    fixation_df.drop(indices_to_drop, inplace=True)

    return fixation_df

def pre_process_fixation_data(fixation_filepath, language):

    if fixation_filepath.endswith('.rda'):
        fixation_filepath = convert_rdm_to_csv(fixation_filepath)

    fixation_df = pd.read_csv(fixation_filepath)

    if 'lang' in fixation_df.columns:
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
                               'ia.reg.in',
                               'ia.reg.in.from']]

    fixation_df = fixation_df.rename(columns={'ia.reg.out': 'reg.out',
                                              'ia.reg.out.to': 'reg.out.to',
                                              'ia.reg.in': 'reg.in',
                                              'ia.reg.in.from': 'reg.in.from',
                                              'ia.firstskip': 'skip'})

    # Remove first fixation of a trial that is also triggers a regression (likely noise)
    fixation_df = fixation_df.drop(fixation_df[(fixation_df['fixid'] == 1) & (fixation_df['reg.out'] == 1)].index)

    # Remove regression that is subsequently followed by another regression (likely noise)
    fixation_df = remove_subsequent_regressions(fixation_df)

    # Deal with noise in regressions to upper lines
    fixation_df = remove_close_reg_upper_line(fixation_df)

    return fixation_df

### Functions to compute and add variables for analysis (e.g. distance, length, frequency)
def compute_distance(word_ids, regression, regression_to_or_from):

    distances = []
    for i, word_id in enumerate(word_ids):
        distance = None
        if regression[i] == 1:
            distance = abs(word_id - regression_to_or_from[i])
        distances.append(distance)

    return distances

def compute_all_distances(fixation_data):

    distances_out = []
    distances_in = []

    for id, group in fixation_data.groupby(['uniform_id', 'trialid']):
        distance_out = compute_distance(group['ianum'].tolist(), group['reg.out'].tolist(), group['reg.out.to'].tolist())
        distances_out.extend(distance_out)
        distance_in = compute_distance(group['ianum'].tolist(), group['reg.in'].tolist(), group['reg.in.from'].tolist())
        distances_in.extend(distance_in)

    return distances_out, distances_in

def create_line_change_out(fixation_df):

    all_line_change_out = []
    for id, group in fixation_df.groupby(['uniform_id', 'trialid']):
        all_reg_out = group['reg.out'].tolist()
        all_line_change = group['line.change'].tolist()
        all_reg_in = group['reg.in'].tolist()
        all_reg_in_ianum = group['ianum'].tolist()
        all_reg_out_to = group['reg.out.to'].tolist()
        line_change_out = None
        for i, reg_out in enumerate(all_reg_out):
            if reg_out == 1:
                if all_reg_in[i+1] == 1 and all_reg_in_ianum[i+1] == all_reg_out_to[i]:
                    line_change_out = all_line_change[i+1]
            all_line_change_out.append(line_change_out)

    return all_line_change_out

def categorize_distance(row):

    if row['reg.dist'] == 0:
        return "NoDistance"
    elif (row['reg.dist'] == 1) & (row['reg.out.line.change'] == 0):
        return "Short"
    elif (row['reg.dist'] in [2, 3]) & (row['reg.out.line.change'] == 0):
        return "Mid-short"
    elif (row['reg.dist'] >= 4) & (row['reg.out.line.change'] == 0):
        return "Mid-long"
    elif (row['reg.dist'] != 0) & (row['reg.out.line.change'] <= -1):
        return "Long"
    else:
        return np.nan

def categorize_distance_in(row):

    if row['reg.dist.in'] == 0:
        return "NoDistance"
    elif (row['reg.dist.in'] == 1) & (row['line.change'] == 0):
        return "Short"
    elif (row['reg.dist.in'] in [2, 3]) & (row['line.change'] == 0):
        return "Mid-short"
    elif (row['reg.dist.in'] >= 4) & (row['line.change'] == 0):
        return "Mid-long"
    elif (row['reg.dist.in'] != 0) & (row['line.change'] <= -1):
        return "Long"
    else:
        return np.nan

def add_variables(variables, df, resources, language):

    if 'distance' in variables:
        distances_out, distances_in = compute_all_distances(df)
        # distance of regressions out (in words)
        df['reg.dist'] = distances_out
        df["reg.dist"] = df["reg.dist"].fillna(0)
        # distance of regressions in (in words)
        df['reg.dist.in'] = distances_in
        df["reg.dist.in"] = df["reg.dist.in"].fillna(0)
        # create distance bins for distance model(s)
        df['reg.out.line.change'] = create_line_change_out(df)
        df['reg.dist.binned'] = df.apply(categorize_distance, axis=1)
        df = df.drop('reg.out.line.change', axis=1)
        df['reg.dist.in.binned'] = df.apply(categorize_distance_in, axis=1)
        # log distances
        distances_above_zero = df['reg.dist'][df['reg.dist'] > 0].tolist()
        reg_dist_no_zero = df['reg.dist'].apply(lambda x: np.mean(distances_above_zero) if x == 0 else x)
        df['reg.dist.log'] = np.log(np.array(reg_dist_no_zero))
        distances_above_zero = df['reg.dist.in'][df['reg.dist.in'] > 0].tolist()
        df["reg.dist.in"] = df['reg.dist.in'].apply(lambda x: np.mean(distances_above_zero) if x == 0 else x)
        df['reg.dist.in.log'] = np.log(np.array(df["reg.dist.in"].tolist()))
        # distance in letters
        df['reg.dist.let'] = [0 if reg_out == 0 else abs(sac_out) for reg_out, sac_out in zip(df['reg.out'], df['sac.out'])]
        distances_above_zero = df['reg.dist.let'][df['reg.dist.let'] > 0].tolist()
        reg_dist_no_zero = df['reg.dist.let'].apply(lambda x: np.mean(distances_above_zero) if x == 0 else x)
        df['reg.dist.let.log'] = np.log(np.array(reg_dist_no_zero))

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
        for id, group in df.groupby(['uniform_id', 'trialid']):
            reg = [group['reg.out'].tolist()[i + 1] if i + 1 < len(group['reg.out'].tolist()) else None for
                   i, reg_out in enumerate(group['reg.out'].tolist())]
            reg_out_plus_one.extend(reg)
        df['reg.out.plus.one'] = reg_out_plus_one

        if 'distance' in variables:
            reg_out_plus_one_dist, reg_out_plus_one_dist_bin = [], []
            for id, group in df.groupby(['uniform_id', 'trialid']):
                reg = [group['reg.dist.log'].tolist()[i + 1] if i + 1 < len(group['reg.dist.log'].tolist()) else None
                       for i, reg_out in enumerate(group['reg.dist.log'].tolist())]
                reg_bin = [group['reg.dist.binned'].tolist()[i + 1] if i + 1 < len(group['reg.dist.binned'].tolist()) else None
                       for i, reg_out in enumerate(group['reg.dist.binned'].tolist())]
                reg_out_plus_one_dist.extend(reg)
                reg_out_plus_one_dist_bin.extend(reg_bin)
            df['reg.dist.log.plus.one'] = reg_out_plus_one_dist
            df['reg.dist.binned.plus.one'] = reg_out_plus_one_dist_bin

    if 'word+2' in variables:
        reg_out_plus_two = []
        for id, group in df.groupby(['uniform_id', 'trialid']):
            reg = [group['reg.out'].tolist()[i + 2] if i + 2 < len(group['reg.out'].tolist()) else None for
                   i, reg_out in enumerate(group['reg.out'].tolist())]
            reg_out_plus_two.extend(reg)
        df['reg.out.plus.two'] = reg_out_plus_two

        if 'distance' in variables:
            reg_out_plus_two_dist, reg_out_plus_two_dist_bin = [], []
            for id, group in df.groupby(['uniform_id', 'trialid']):
                reg = [group['reg.dist.log'].tolist()[i + 2] if i + 2 < len(group['reg.dist.log'].tolist()) else None
                       for i, reg_out in enumerate(group['reg.dist.log'].tolist())]
                reg_bin = [group['reg.dist.binned'].tolist()[i + 2] if i + 2 < len(group['reg.dist.binned'].tolist()) else None
                       for i, reg_out in enumerate(group['reg.dist.binned'].tolist())]
                reg_out_plus_two_dist.extend(reg)
                reg_out_plus_two_dist_bin.extend(reg_bin)
            df['reg.dist.log.plus.two'] = reg_out_plus_two_dist
            df['reg.dist.binned.plus.two'] = reg_out_plus_two_dist_bin

    return df

###############################################################

def pre_process_corpus(texts_filepath, words_filepath, fixation_report_filepath, fixation_filepath, frequency_filepath, language = 'en'):

    # Generate a dataset with each word of each text as row.
    # Columns: trialid (the id of the text); texts (the text the word belongs to); ianum (id of the word); ia (word)
    original_df = create_original_texts_dataframe(texts_filepath, language)
    original_df = clean_original_texts(original_df)
    words_df = create_ianum_from_original_texts(original_df)
    words_df.to_csv(words_filepath, index=False)

    # Generate a dataset with each fixation as a row and add variables for analysis
    resources = {'frequency': frequency_filepath}
    fixation_df = pre_process_fixation_data(fixation_report_filepath, language)
    fixation_df = add_variables(['distance', 'length', 'frequency', 'word+1', 'word+2'], fixation_df, resources, language)
    fixation_df.to_csv(fixation_filepath, index=False)

    return fixation_df, words_df

