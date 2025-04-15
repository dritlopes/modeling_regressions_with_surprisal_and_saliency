import numpy as np
import pandas as pd

def convert_saliency_string_into_array(saliency_array:str) -> list[float]:

    """
    Convert saliency array string into an actual array
    :param saliency_array: saliency array string
    :return: list of saliency scores
    """

    saliency_array = saliency_array.replace('[', '').replace(']', '')
    saliency_array = saliency_array.replace('np.float64', '').replace('(', '').replace(')', '').replace(' np.float64', '')
    saliency_array = saliency_array.split(',')
    saliency_array = [float(s) for s in saliency_array[:-1]]

    return saliency_array


def add_saliency_relative_to_context_words(saliency_array:list[float], importance_df_trialid:pd.DataFrame, fixated_word_id:int)-> (list[float],list[float],list[float],list[float]):

    """
    Adding saliency relative to words surrounding regression source
    :param saliency_array: list of saliency scores
    :param importance_df_trialid: dataset with saliency values for a specific text
    :param fixated_word_id: which word is the regression source
    :return: saliency score relative to words in the surrounding context of the regression source
    """

    saliency_plus_one, saliency_plus_two, saliency_minus_one, saliency_minus_two = np.zeros(
        len(saliency_array)), np.zeros(len(saliency_array)), np.zeros(len(saliency_array)), np.zeros(
        len(saliency_array))

    saliency_plus_one[saliency_plus_one == 0] = np.nan
    saliency_plus_two[saliency_plus_two == 0] = np.nan
    saliency_minus_one[saliency_minus_one == 0] = np.nan
    saliency_minus_two[saliency_minus_two == 0] = np.nan

    # context words to fixated word (text order)
    # N + 1
    if fixated_word_id + 1 in importance_df_trialid['ianum'].tolist():
        fixated_word_plus_one_df = importance_df_trialid[
            (importance_df_trialid['ianum'] == fixated_word_id + 1)]
        saliency_plus_one = fixated_word_plus_one_df['distributed_saliency'].tolist()[0]
        if type(saliency_plus_one) == str:
            saliency_plus_one = convert_saliency_string_into_array(saliency_plus_one)
        saliency_plus_one = saliency_plus_one[:len(saliency_array)]
    # N + 2
    if fixated_word_id + 2 in importance_df_trialid['ianum'].tolist():
        fixated_word_plus_two_df = importance_df_trialid[
            (importance_df_trialid['ianum'] == fixated_word_id + 2)]
        saliency_plus_two = fixated_word_plus_two_df['distributed_saliency'].tolist()[0]
        if type(saliency_plus_two) == str:
            saliency_plus_two = convert_saliency_string_into_array(saliency_plus_two)
        saliency_plus_two = saliency_plus_two[:len(saliency_array)]
    # N - 1
    if fixated_word_id - 1 in importance_df_trialid['ianum'].tolist() and fixated_word_id - 1 != 1:
        fixated_word_minus_one_df = importance_df_trialid[
            (importance_df_trialid['ianum'] == fixated_word_id - 1)]
        saliency_minus_one = fixated_word_minus_one_df['distributed_saliency'].tolist()[0]
        if type(saliency_minus_one) == str:
            saliency_minus_one = convert_saliency_string_into_array(saliency_minus_one)
        saliency_minus_one = saliency_minus_one[:len(saliency_array)]
        saliency_minus_one.append(np.nan)
    # N - 2
    if fixated_word_id - 2 in importance_df_trialid['ianum'].tolist() and fixated_word_id - 2 != 1:
        fixated_word_minus_two_df = importance_df_trialid[
            (importance_df_trialid['ianum'] == fixated_word_id - 2)]
        saliency_minus_two = fixated_word_minus_two_df['distributed_saliency'].tolist()[0]
        if type(saliency_minus_two) == str:
            saliency_minus_two = convert_saliency_string_into_array(saliency_minus_two)
        saliency_minus_two = saliency_minus_two[:len(saliency_array)]
        saliency_minus_two.append(np.nan)
        saliency_minus_two.append(np.nan)

    assert len(saliency_plus_one) == len(saliency_array), print(fixated_word_id, saliency_array,
                                                                saliency_plus_one)
    assert len(saliency_minus_one) == len(saliency_array), print(fixated_word_id, saliency_array,
                                                                 saliency_minus_one)
    assert len(saliency_plus_two) == len(saliency_array), print(fixated_word_id, saliency_array,
                                                                saliency_plus_two)
    assert len(saliency_minus_two) == len(saliency_array), print(fixated_word_id, saliency_array,
                                                                 saliency_minus_two)

    return saliency_plus_one, saliency_plus_two, saliency_minus_one, saliency_minus_two

def create_saliency_dataframe(fixation_df: pd.DataFrame, importance_df: pd.DataFrame) -> pd.DataFrame:

    """
    Given a regression, pair each regression source with each previous word and register info on context word.

    :param fixation_df: dataframe containing fixation data.
    :param importance_df: dataframe containing saliency data.
    :return: dataframe containing analysis data.
    """

    importance_df.rename(columns={'text_id': 'trialid', 'token_id': 'ianum', 'token': 'ia'}, inplace=True)

    # find frequency, line and surprisal values from fixation data
    frequency_map, surprisal_map, sent_map = dict(), dict(), dict()
    for trial_id, word_id, freq, surprisal, sentnum in zip(fixation_df['trialid'].tolist(),
                                                                fixation_df['ianum'].tolist(),
                                                                fixation_df['frequency'].tolist(),
                                                                fixation_df['surprisal'].tolist(),
                                                                fixation_df['sentnum'].tolist()):
        frequency_map[f'{trial_id}-{float(word_id)}'] = freq
        surprisal_map[f'{trial_id}-{float(word_id)}'] = surprisal
        sent_map[f'{trial_id}-{float(word_id)}'] = sentnum

    # only regressions
    fixation_df = fixation_df[(fixation_df['reg.out']) == 1.0]

    fixation_importance = {'participant_id': [], # participant
                           'participant_id_int': [], # participant without str to run gam
                           'trialid': [], # text id
                           'source.ianum': [], # regression source id
                           'source.ia': [], # regression source word
                           'context.ia': [], # context word (candidate regression target)
                           'context.ianum': [], # context word id
                           'sent.change': [], # whether they are in different sentences
                           'source.ia.length': [], # reg source length
                           'source.ia.frequency': [], # reg source frequency
                           'context.ia.length': [], # length of context word
                           'context.ia.frequency': [], # frequency of context word
                           'dur': [],  # duration of fixation previous to regression
                           'dist': [],  # distance between regression source and context word
                           'source.ia.surprisal': [], # reg source surprisal
                           'context.ia.surprisal': [], # surprisal of context word
                           'saliency': [], # saliency of context word in relation to regression source
                           'saliency.rank': [], # position of saliency of context word in relation to saliency of all context words
                           'saliency.minus.one': [], # saliency of context word in relation to one word previous to regression source
                           'saliency.minus.two': [], # saliency of context word in relation to two words previous to regression source
                           'saliency.plus.one': [], # saliency of context word in relation to one word next to regression source
                           'saliency.plus.two': [], # saliency of context word in relation to two words next to regression source
                           'reg.in': []}  # whether the context word is the regression target

    for id, group in fixation_df.groupby(['participant_id', 'trialid']):

        # filter saliency values for words in current text
        importance_df_trialid = importance_df[(importance_df['trialid'] == id[1])]

        # loop through fixations (regressions)
        for i, row in group.iterrows():

            # ---------- Find saliency values relative to current regression source
            fixated_word_importance = importance_df_trialid[(importance_df_trialid['ianum'] == row['ianum'])]
            saliency_array = fixated_word_importance['distributed_saliency'].tolist()[0]
            if type(saliency_array) == str:
                saliency_array = convert_saliency_string_into_array(saliency_array)
            saliency_sorted = np.sort(saliency_array)[::-1]
            saliency_rank = [np.where(saliency_sorted == value)[0][0] + 1 for value in saliency_array]
            # save each saliency value and the relative context word
            fixation_importance['saliency'].extend(saliency_array)
            fixation_importance['saliency.rank'].extend(saliency_rank)
            fixation_importance['context.ianum'].extend(importance_df_trialid['ianum'].tolist()[:len(saliency_array)])
            fixation_importance['context.ia'].extend(importance_df_trialid['ia'].tolist()[:len(saliency_array)])
            fixation_importance['source.ianum'].extend([row['ianum'] for i in range(len(saliency_array))])
            fixation_importance['source.ia'].extend([row['ia'] for i in range(len(saliency_array))])
            fixation_importance['trialid'].extend([id[1] for i in range(len(saliency_array))])
            fixation_importance['participant_id'].extend([id[0] for i in range(len(saliency_array))])
            fixation_importance['participant_id_int'].extend([int(id[0].replace('en_','').replace('sub','')) for i in range(len(saliency_array))])
            fixation_importance['dur'].extend([row['dur'] for i in range(len(saliency_array))])
            fixation_importance['source.ia.length'].extend([row['length'] for i in range(len(saliency_array))])
            fixation_importance['source.ia.frequency'].extend([row['frequency'] for i in range(len(saliency_array))])
            fixation_importance['source.ia.surprisal'].extend([row['surprisal'] for i in range(len(saliency_array))])

            # --------- Variables for extra checks in the saliency analysis
            saliency_plus_one, saliency_plus_two, saliency_minus_one, saliency_minus_two = (
                add_saliency_relative_to_context_words(saliency_array, importance_df_trialid, row['ianum']))
            fixation_importance['saliency.plus.one'].extend(saliency_plus_one)
            fixation_importance['saliency.plus.two'].extend(saliency_plus_two)
            fixation_importance['saliency.minus.one'].extend(saliency_minus_one)
            fixation_importance['saliency.minus.two'].extend(saliency_minus_two)

            # --------- Add specific info on each previous word
            reg_in, dist, lengths, frequencies, surprisals, sent_changes = [], [], [], [], [], []
            # for each context word
            for previous_id, previous_ia in zip(importance_df_trialid['ianum'].tolist()[:len(saliency_array)],
                                                importance_df_trialid['ia'].tolist()[:len(saliency_array)]):
                # frequency
                if f"{id[1]}-{float(previous_id)}" in frequency_map.keys():
                    frequencies.append(frequency_map[f"{id[1]}-{float(previous_id)}"])
                else:
                    frequencies.append(None)
                # surprisal
                if f"{id[1]}-{float(previous_id)}" in surprisal_map.keys():
                    surprisals.append(surprisal_map[f"{id[1]}-{float(previous_id)}"])
                else:
                    surprisals.append(None)
                # save regression info
                if previous_id == row['reg.out.to']:
                    reg_in.append(1) # whether word is the regression target
                else:
                    reg_in.append(0)
                # distance and distance category
                distance = int(row['ianum']) - int(previous_id)
                sent_change = None
                if f"{id[1]}-{float(previous_id)}" in sent_map.keys():
                    if sent_map[f"{id[1]}-{float(previous_id)}"] < row['sentnum']:
                        sent_change = 1
                    elif sent_map[f"{id[1]}-{float(previous_id)}"] == row['sentnum']:
                        sent_change = 0
                dist.append(distance) # distance between word and regression source
                lengths.append(len(previous_ia)) # length
                sent_changes.append(sent_change)
            fixation_importance['reg.in'].extend(reg_in)
            fixation_importance['context.ia.length'].extend(lengths)
            fixation_importance['context.ia.frequency'].extend(frequencies)
            fixation_importance['context.ia.surprisal'].extend(surprisals)
            fixation_importance['dist'].extend(dist)
            fixation_importance['sent.change'].extend(sent_changes)

    fixation_importance_df = pd.DataFrame.from_dict(fixation_importance)

    return fixation_importance_df

def bin_distances(df:pd.DataFrame) -> pd.DataFrame:

    """
    Group words based on distance to regression source.
    :param df: dataframe containing context words to regression sources
    :return: dataframe with column dist.bins, which defines which distance group each word belongs to.
    """

    # only regression targets
    df_reg = df.copy().loc[df['reg.in'] == 1]
    # make bins based on quantities of regression targets per distance
    res, bins = pd.qcut(df_reg['dist'], 20, duplicates='drop', retbins=True)
    # print(res.value_counts())

    # apply bins to all words (also not regression targets)
    dist_bins = []
    for dist in df['dist'].tolist():
        if dist < bins[-1]: # some longer distances do not have any regression target, thus are excluded from this analysis
            for i, bin in enumerate(bins[:-1]):
                if bin <= dist < bins[i+1]:
                    dist_bins.append(f'{bins[i]}-{bins[i+1]}')
        else:
            dist_bins.append('')
    df['dist.bin'] = dist_bins

    # print(df['source.ianum'].value_counts())
    # for dist_bin in df['dist.bin'].unique():
    #     print('Intra-sentence')
    #     dist_bin_df = df.loc[(df['dist.bin'] == dist_bin) & (df['sent.change'] == 0)]
    #     print(dist_bin)
    #     print(len(dist_bin_df))
    #     print(dist_bin_df['reg.in'].value_counts(normalize=True))
    #     print(dist_bin_df['reg.in'].value_counts())
    #
    #     print('Inter-sentence')
    #     dist_bin_df = df.loc[(df['dist.bin'] == dist_bin) & (df['sent.change'] == 1)]
    #     print(dist_bin)
    #     print(len(dist_bin_df))
    #     print(dist_bin_df['reg.in'].value_counts(normalize=True))
    #     print(dist_bin_df['reg.in'].value_counts())
    #     print()

    return df

def create_incoming_regression_df(fixation_df:pd.DataFrame, importance_df:pd.DataFrame)-> pd.DataFrame:

    """
    Generate dataframe for incoming regression analysis by taking each word previous to a regression source and registering whether it's the regression target and other word-level variables.
    :param fixation_df: fixation dataframe (with surprisal values)
    :param importance_df: word dataframe with saliency values
    :return: fixation dataframe for incoming regression analysis
    """

    df = create_saliency_dataframe(fixation_df, importance_df)
    df = bin_distances(df)

    return df

def create_outgoing_regression_df(surprisal_df: pd.DataFrame, fixation_df: pd.DataFrame)-> pd.DataFrame():

    """
    Generate dataframe for outgoing regression analysis by adding surprisal values to fixation dataframe
    :param surprisal_df: word dataframe with surprisal values
    :param fixation_df: fixation dataframe
    :return: fixation dataframe with surprisal values
    """

    surprisal_fixation_df = pd.merge(fixation_df, surprisal_df[['trialid', 'ianum', 'surprisal']], how='left',
                                     on=['trialid', 'ianum'])
    # add surprisal n-1
    surprisal_minus_one = []
    for i, row in surprisal_fixation_df.iterrows():
        if i - 1 >= 0:
            previous_row = surprisal_fixation_df.loc[i - 1]
            if previous_row.participant_id == row.participant_id and previous_row.trialid == row.trialid:
                surprisal_minus_one.append(previous_row.surprisal)
            else:
                surprisal_minus_one.append(np.nan)
        else:
            surprisal_minus_one.append(np.nan)
    surprisal_fixation_df['ia.minus.one.surprisal'] = surprisal_minus_one

    return surprisal_fixation_df