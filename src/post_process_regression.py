import numpy as np
import pandas as pd

def categorize_distance(distance, line_change):

    if (distance == 1) & (line_change == 0):
        return "Short"
    elif (distance in [2, 3]) & (line_change == 0):
        return "Mid-short"
    elif (distance >= 4) & (line_change == 0):
        return "Mid-long"
    elif (distance != 0) & (line_change == 1):
        return "Long"
    else:
        return np.nan

def convert_saliency_string_into_array(saliency_array):

    saliency_array = saliency_array.replace('[', '').replace(']', '')
    saliency_array = saliency_array.split(',')
    saliency_array = [float(s) for s in saliency_array[:-1]]
    return saliency_array

def create_saliency_dataframe(fixation_df, importance_df):

    importance_df.rename(columns={'text_id': 'trialid', 'token_id': 'ianum', 'token': 'ia'}, inplace=True)

    frequency_map = dict()
    line_map = dict()
    for trial_id, word_id, freq, line in zip(fixation_df['trialid'].tolist(),
                                               fixation_df['ianum'].tolist(),
                                               fixation_df['frequency'].tolist(),
                                               fixation_df['line'].tolist()):
        frequency_map[f'{trial_id}-{word_id}'] = freq
        line_map[f'{trial_id}-{word_id}'] = line

    fixation_df = fixation_df[(fixation_df['reg.out']) == 1.0]

    fixation_importance = {'uniform_id': [],
                           'trialid': [],
                           'ianum': [],
                           'ia': [],
                           'previous.ia': [],
                           'previous.ianum': [],
                           'reg.in': [],
                           'reg.dist.in': [],
                           'reg.dist.in.log': [],
                           'reg.dist.in.bin': [],
                           'dist': [],
                           'dist.log': [],
                           'dist.bin': [],
                           'length': [],
                           'frequency': [],
                           'saliency': [],
                           'saliency.minus.one': [],
                           'saliency.plus.two': []}

    for id, group in fixation_df.groupby(['uniform_id', 'trialid']):

        importance_df_trialid = importance_df[(importance_df['trialid'] == id[1]-1)]

        for fixated_word, fixated_word_id, reg_out_to, reg_dist, reg_dist_log, reg_dist_bin, line in \
                zip(group['ia'].tolist(),
                    group['ianum'].tolist(),
                    group['reg.out.to'].tolist(),
                    group['reg.dist'].tolist(),
                    group['reg.dist.log'].tolist(),
                    group['reg.dist.binned'].tolist(),
                    group['line'].tolist()):

            fixated_word_df = importance_df_trialid[(importance_df_trialid['ianum'] == fixated_word_id)]
            saliency_array = fixated_word_df['distributed_saliency'].tolist()[0]
            if type(saliency_array) == str:
                saliency_array = convert_saliency_string_into_array(saliency_array)
            fixation_importance['saliency'].extend(saliency_array)
            fixation_importance['previous.ianum'].extend(importance_df_trialid['ianum'].tolist()[:len(saliency_array)])
            fixation_importance['previous.ia'].extend(importance_df_trialid['ia'].tolist()[:len(saliency_array)])
            fixation_importance['ianum'].extend([fixated_word_id for i in range(len(saliency_array))])
            fixation_importance['ia'].extend([fixated_word for i in range(len(saliency_array))])
            fixation_importance['trialid'].extend([id[1] for i in range(len(saliency_array))])
            fixation_importance['uniform_id'].extend([id[0] for i in range(len(saliency_array))])

            # variables for extra checks in the saliency analysis
            saliency_plus_two, saliency_minus_one = np.zeros(len(saliency_array)), np.zeros(len(saliency_array))
            saliency_plus_two[saliency_plus_two == 0] = np.nan
            saliency_minus_one[saliency_minus_one == 0] = np.nan
            # context words to fixated word (text order)
            if fixated_word_id + 2 in importance_df_trialid['ianum'].tolist():
                fixated_word_plus_two_df = importance_df_trialid[
                    (importance_df_trialid['ianum'] == fixated_word_id + 2)]
                saliency_plus_two = fixated_word_plus_two_df['distributed_saliency'].tolist()[0]
                if type(saliency_plus_two) == str:
                    saliency_plus_two = convert_saliency_string_into_array(saliency_plus_two)
                saliency_plus_two = saliency_plus_two[:len(saliency_array)]
            if fixated_word_id - 1 in importance_df_trialid['ianum'].tolist() and fixated_word_id - 1 != 1:
                fixated_word_minus_one_df = importance_df_trialid[
                    (importance_df_trialid['ianum'] == fixated_word_id - 1)]
                saliency_minus_one = fixated_word_minus_one_df['distributed_saliency'].tolist()[0]
                if type(saliency_minus_one) == str:
                    saliency_minus_one = convert_saliency_string_into_array(saliency_minus_one)
                saliency_minus_one = saliency_minus_one[:len(saliency_array)]
                saliency_minus_one.append(np.nan)
            assert len(saliency_plus_two) == len(saliency_array), print(fixated_word_id, saliency_array, saliency_plus_two)
            assert len(saliency_minus_one) == len(saliency_array), print(fixated_word_id, saliency_array, saliency_minus_one)
            fixation_importance['saliency.plus.two'].extend(saliency_plus_two)
            fixation_importance['saliency.minus.one'].extend(saliency_minus_one)


            # add specific info on each previous word
            reg_in, all_reg_dist, all_reg_dist_log, all_reg_dist_bin, lengths, frequencies = [], [], [], [], [], []
            distances, distance_bins = [], []
            for previous_id, previous_ia in zip(importance_df_trialid['ianum'].tolist()[:len(saliency_array)],
                                                importance_df_trialid['ia'].tolist()[:len(saliency_array)]):
                if f"{id[1]}-{float(previous_id)}" in frequency_map.keys():
                    frequencies.append(frequency_map[f"{id[1]}-{float(previous_id)}"])
                else:
                    frequencies.append(None)
                if previous_id == reg_out_to:
                    reg_in.append(1)
                    all_reg_dist.append(reg_dist)
                    all_reg_dist_log.append(reg_dist_log)
                    all_reg_dist_bin.append(reg_dist_bin)
                    lengths.append(len(previous_ia))
                else:
                    reg_in.append(0)
                    all_reg_dist.append(0)
                    all_reg_dist_log.append(np.mean(fixation_df.loc[fixation_df['reg.dist'] > 0]['reg.dist'].tolist()))
                    all_reg_dist_bin.append('NoDistance')
                    lengths.append(len(previous_ia))
                distance = int(fixated_word_id) - int(previous_id)
                line_change = None
                if f"{id[1]}-{float(previous_id)}" in line_map.keys():
                    if line_map[f"{id[1]}-{float(previous_id)}"] < line:
                        line_change = 1
                    elif line_map[f"{id[1]}-{float(previous_id)}"] == line:
                        line_change = 0
                distances.append(distance)
                distance_bins.append(categorize_distance(distance, line_change))
            fixation_importance['reg.in'].extend(reg_in)
            fixation_importance['length'].extend(lengths)
            fixation_importance['frequency'].extend(frequencies)
            fixation_importance['reg.dist.in'].extend(all_reg_dist)
            fixation_importance['reg.dist.in.log'].extend(all_reg_dist_log)
            fixation_importance['reg.dist.in.bin'].extend(all_reg_dist_bin)
            fixation_importance['dist'].extend(distances)
            fixation_importance['dist.log'].extend(np.log(np.array(distances)))
            fixation_importance['dist.bin'].extend(distance_bins)

    fixation_importance_df = pd.DataFrame.from_dict(fixation_importance)

    return fixation_importance_df

def filter_saliency_df(df, cut_off=.01):

    # filter instances to only from distances for which we have enough regressions (above 1% of the non-regressions)

    distances = []
    for dist, group in df.groupby('dist'):
        n_instances = len(group['reg.in'].tolist())
        if 1 in group['reg.in'].value_counts().keys():
            n_reg_in = group['reg.in'].value_counts()[1]
            if n_reg_in >= n_instances * cut_off:
                distances.append(dist)
    df = df.loc[df['dist'].isin(distances)]
    print(f'Distances: {distances}')
    print(f'{df["dist"].value_counts()}')

    return df

def create_regression_saliency_df(model_name, corpus_df, importance_df):

    regression_importance_df = create_saliency_dataframe(corpus_df, importance_df)
    filtered_df = filter_saliency_df(regression_importance_df)
    filtered_df.to_csv(f'../data/MECO/saliency_{model_name}_regIn.csv')