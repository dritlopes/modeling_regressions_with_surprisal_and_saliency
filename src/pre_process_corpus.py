import pandas as pd
import rdata
import numpy as np
import string
from scipy.stats import zscore
from statistics import median

class WordData:

    def __init__(self,
                 corpus:str,
                 filepath:str):
        self.corpus = corpus
        self.filepath = filepath
        self.data = None

    def _create_texts_meco_df(self):

        """
        Create dataframe where each text word is row. Columns: trialid (the id of the text); texts (the text the word belongs to); ianum (id of the word); ia (word))
        :return: words_df
        """

        data =  pd.read_csv(self.filepath)

        data.drop(['Unnamed: 13', 'Unnamed: 14'], axis=1, inplace=True)
        data.columns = ['lang', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        # only English texts
        lan_filter = (data['lang'] == 'English')
        lan_texts_df = data.loc[lan_filter]
        # re-structure data so that each text becomes a row
        trialid_raw_df = lan_texts_df.stack().astype(str).reset_index(level=1)
        trialid_raw_df.rename(columns={'level_1': 'trialid', 0: 'text'}, inplace=True)
        trialid_raw_df = trialid_raw_df.reset_index(drop=False)
        trialid_raw_df.drop([0], inplace=True)
        trialid_raw_df.drop(['index'], axis=1, inplace=True)

        # do some cleaning on each text
        trialid_cleaning_df = trialid_raw_df.copy()
        # replace with "space" the "\\n" at the beginning of a word
        trialid_cleaning_df["text"] = trialid_cleaning_df["text"].str.replace(" \\n", " ")
        # replace with "space" the "\\n" between words as "word\\nword"
        trialid_cleaning_df["text"] = trialid_cleaning_df["text"].str.replace("\\n", " ")
        # when "word-word" add a space after first word, then the words would be separated equally
        trialid_cleaning_df["text"] = trialid_cleaning_df["text"].str.replace("-", "- ")
        # replace with a empty string all the quotation marks
        trialid_cleaning_df["text"] = trialid_cleaning_df["text"].str.replace('"', '')

        # create dataframe with each row being a word
        trialid, text, ia_new, ianum_new = [], [], [], []
        # to interact between a list of rows
        for _, row in trialid_cleaning_df.iterrows():
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
        return words_df

    def _create_texts_provo_df(self) -> pd.DataFrame:

        """
        Create dataframe where each text word is row. Columns: trialid (the id of the text); texts (the text the word belongs to); ianum (id of the word); ia (word))
        :return: words_df
        """

        data = pd.read_csv(self.filepath, encoding="ISO-8859-1")

        # make some corrections on word ids
        sent_map = dict()
        data['Word_Number'] = data.apply(
            lambda x: int(x["Word_Number"]) - 1 if (int(x["Word_Number"]) > 44) & (int(x["Text_ID"]) == 3) else int(x["Word_Number"]), axis=1)
        data['Word_Number'] = data.apply(
            lambda x: int(x["Word_Number"]) - 1 if (int(x["Word_Number"]) > 18) & (int(x["Text_ID"]) == 13) else int(x["Word_Number"]), axis=1)
        for row in data.itertuples():
            sent_map[f'{row.Text_ID}-{row.Word_Number}'] = int(row.Sentence_Number)
        sent_map['18-51'] = 3 # 'evolution' in text 13 not present in provo predictability data
        sent_map['55-3'] = 1

        # create dataframe with each row being a word
        trialids, texts, words, word_ids, sent_ids = [], [], [], [], []
        for i, text_info in data.groupby('Text_ID'):

            trialid = int(i)

            text = text_info['Text'].tolist()[0]
            # fix errors in texts in raw data
            text = text.replace(' Ñ', '')
            text = text.replace('Õ', "'")

            text_words = text.split()
            text_words = [word.replace('"', '') for word in text_words]

            text_word_ids = [i + 1 for i in range(len(text_words))]

            sentence_ids = [1]
            sentence_ids.extend([sent_map[f'{trialid}-{word_id}'] for word_id in text_word_ids[1:]])

            trialids.extend([trialid for i in range(len(text_words))])
            texts.extend([text for i in range(len(text_words))])
            words.extend(text_words)
            word_ids.extend(text_word_ids)
            sent_ids.extend(sentence_ids)

        words_df = pd.DataFrame(data={'trialid': trialids,
                                      'texts': texts,
                                      'sentnum': sent_ids,
                                      'ianum': word_ids,
                                      'ia': words})
        return words_df

    def create_texts_df(self):

        if self.corpus == 'Provo':
            dataframe = self._create_texts_provo_df()
        elif self.corpus == 'MECO':
            dataframe = self._create_texts_meco_df()
        else:
            raise Exception(f'Corpus {self.corpus} not supported.')

        self.data = dataframe

        return self.data

class FixationData:

    def __init__(self, corpus:str, filepath:str):
        self.corpus = corpus
        self.filepath = filepath
        self.data = None

    @staticmethod
    def _compute_extra_variables_provo(data:pd.DataFrame) -> pd.DataFrame:

        """
        Compute reg.out, reg.in, reg.out.to, reg.in.from, and incoming and outgoing saccade distance in words for Provo
        data: fixation data
        :return: fixation dataframe with extra variables
        """

        out_regs, in_regs, out_reg_ias, in_regs_ias, sac_out_dist, sac_in_dist = [], [], [], [], [], []

        for i, row in data.iterrows():
            word_id = row['ianum']
            next_word_id = row['next.ianum']
            previous_word_id = row['previous.ianum']
            # register information on incoming saccade, if there was a previous fixation
            if not pd.isna(previous_word_id):
                in_dist = int(word_id[0]) - int(previous_word_id)
                sac_in_dist.append(in_dist)
                # if distance between current fixation and previous fixation is negative, reg.in == 1
                if in_dist < 0:
                    in_regs.append(1)
                    in_regs_ias.append(previous_word_id)
                else:
                    in_regs.append(0)
                    in_regs_ias.append(np.nan)
            else:
                sac_in_dist.append(np.nan)
                in_regs.append(np.nan)
                in_regs_ias.append(np.nan)
            # register information on outgoing saccade, if there was a next fixation
            if not pd.isna(next_word_id):
                out_dist = int(next_word_id) - int(word_id)
                sac_out_dist.append(out_dist)
                # if distance between current fixation and next fixation is negative, reg.out == 1
                if out_dist < 0:
                    out_regs.append(1)
                    out_reg_ias.append(next_word_id)
                else:
                    out_regs.append(0)
                    out_reg_ias.append(np.nan)
            else:
                sac_out_dist.append(np.nan)
                out_regs.append(np.nan)
                out_reg_ias.append(np.nan)
        data['sac.out'] = sac_out_dist
        data['sac.in'] = sac_in_dist
        data['reg.in'] = in_regs
        data['reg.in.from'] = in_regs_ias
        data['reg.out'] = out_regs
        data['reg.out.to'] = out_reg_ias

        return data

    @staticmethod
    def _convert_rdm_to_csv(original_filepath: str) -> str:

        """
        Convert rdm file to csv
        :param original_filepath: filepath to rdm file
        :return: filepath to csv file
        """

        converted = rdata.read_rda(original_filepath)
        converted_key = list(converted.keys())[0]
        df = pd.DataFrame(converted[converted_key])
        filepath = original_filepath.replace('rda', 'csv')
        df.to_csv(filepath)

        return filepath

    @staticmethod
    def _remove_subsequent_regressions(fixation_df: pd.DataFrame) -> pd.DataFrame:

        """
        Remove regressions that are subsequently followed by another regression.
        :param fixation_df: fixation dataframe
        :return: fixation dataframe with such regressions remmoved
        """
        indices_to_drop = []
        for id, group in fixation_df.groupby(['participant_id', 'trialid']):
            reg_in = group['reg.in'].tolist()
            for i, reg in enumerate(reg_in):
                if i + 1 < len(reg_in):
                    if reg_in[i] == 1 and reg_in[i + 1] == 1:
                        indices_to_drop.append(group.iloc[i].name)
        fixation_df.drop(indices_to_drop, inplace=True)

        return fixation_df

    @staticmethod
    def _remove_close_reg_upper_line(fixation_df: pd.DataFrame) -> pd.DataFrame:

        """
        Remove regressions to one line above and too close in x-range.
        We assume regression was unintended when it was to the previous line (<y) and forward (>x),
        and the number of pixels that the saccade goes forward (ie., change in x) is below the
        90th percentile for x changes for forward saccades.
        Check distance of reg not in the same line and remove if to line above and close to origin
        :param fixation_df: fixation dataframe
        :return: fixation dataframe with such regressions removed.
        """

        # find changes to previous line
        x_changes = []
        count_line_above = 0
        count_forward = 0

        for id, group in fixation_df.groupby(['participant_id', 'trialid']):
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

        # drop indices where x is too close
        indices_to_drop = []

        threshold_x_change = np.percentile(x_changes, 90)
        for id, group in fixation_df.groupby(['participant_id', 'trialid']):
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

    def _pre_process_provo_fixation_data(self) -> pd.DataFrame:

        """
        Create dataframe with fixation data from Provo to be analysed further.
        :return: pre-processed fixation dataframe
        """

        df = pd.read_csv(self.filepath, encoding="ISO-8859-1")

        # select columns
        df = df[['RECORDING_SESSION_LABEL', 'CURRENT_FIX_INDEX', 'CURRENT_FIX_INTEREST_AREA_INDEX',
                 'CURRENT_FIX_INTEREST_AREA_LABEL',
                 'CURRENT_FIX_INTEREST_AREA_DWELL_TIME', 'NEXT_FIX_INTEREST_AREA_INDEX', 'NEXT_FIX_INTEREST_AREA_LABEL',
                 'PREVIOUS_FIX_INTEREST_AREA_INDEX', 'PREVIOUS_FIX_INTEREST_AREA_LABEL', 'TRIAL_LABEL']]

        # rename columns
        df = df.rename(columns={'CURRENT_FIX_INDEX': 'fixid',
                                'CURRENT_FIX_INTEREST_AREA_DWELL_TIME': 'dur',
                                'RECORDING_SESSION_LABEL': 'participant_id',
                                'CURRENT_FIX_INTEREST_AREA_INDEX': 'ianum',
                                'CURRENT_FIX_INTEREST_AREA_LABEL': 'ia',
                                'NEXT_FIX_INTEREST_AREA_INDEX': 'next.ianum',
                                'NEXT_FIX_INTEREST_AREA_LABEL': 'next.ia',
                                'PREVIOUS_FIX_INTEREST_AREA_LABEL': 'previous.ia',
                                'PREVIOUS_FIX_INTEREST_AREA_INDEX': 'previous.ianum'})

        # filter data of participants to only contain the same participants of Provo_Corpus-Eyetracking_Data.csv
        to_include = [id for id in df['participant_id'].unique().tolist() if 'a' not in id and id != '80']
        df = df[df['participant_id'].isin(to_include)]

        # strip trailing spaces from words
        df['ia'] = df['ia'].apply(lambda x: x.strip())

        # fix character error
        df['ia'] = df['ia'].apply(lambda x: x.replace('Õ', "'"))
        df['ia'] = df['ia'].apply(lambda x: x.replace(' Ñ', ''))
        df['ia'] = df['ia'].apply(lambda x: x.replace('Ñ', '.'))
        df['ia'] = df['ia'].apply(lambda x: '.' if x == 'livres--a' else x)
        df['ia'] = df['ia'].apply(lambda x: '.' if x == 'profession--writing.' else x)

        # drop rows with empty cell ('.')
        df['ianum'] = df['ianum'].apply(lambda x: np.nan if x == '.' else x)
        df['ia'] = df['ia'].apply(lambda x: np.nan if x == '.' else x)
        df.dropna(subset=['ia', 'ianum'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # replace '.' by empty string in previous and next ianum
        df['next.ianum'] = df['next.ianum'].apply(lambda x: np.nan if x == '.' else x)
        df['previous.ianum'] = df['previous.ianum'].apply(lambda x: np.nan if x == '.' else x)

        # add participant code without str for gam analysis
        df['participant_id_int'] = [int(id.replace('sub', '')) for id in df['participant_id'].tolist()]

        # add median split of duration of each participant
        median_split = []
        for i, participant_fixations in df.groupby('participant_id'):
            median_split.extend(['long' if dur > median(participant_fixations['dur']) else 'short' for dur in
                                 participant_fixations['dur'].tolist()])
        df['dur.bin'] = median_split

        # compute regression info and outgoing saccade distance in words
        df = self._compute_extra_variables_provo(df)

        return df

    def _pre_process_meco_fixation_data(self) -> pd.DataFrame:

        """
        Create dataframe with fixation data from MECO to be analysed further.
        :return: pre-processed fixation dataframe
        """

        # convert fixation report to csv
        if self.filepath.endswith('.rda'):
            self.filepath = self._convert_rdm_to_csv(self.filepath)

        fixation_df = pd.read_csv(self.filepath)

        # filter out non-english data
        if 'lang' in fixation_df.columns:
            fixation_df = fixation_df[(fixation_df['lang'] == 'en')]

        # removed unnamed columns if existent
        fixation_df = fixation_df.loc[:, ~fixation_df.columns.str.contains('^Unnamed')]

        # define data columns
        fixation_df = fixation_df[['uniform_id',  # id of participant
                                   'trialid',  # text id
                                   'fixid',  # id of fixation
                                   'ia',  # form of fixated word
                                   'ianum',  # id of fixated word
                                   'sentnum',  # id of sentence word is in
                                   'dur',  # duration of fixation
                                   'xs',
                                   # raw x position of fixation point (in pixels); needed to filter out noisy regressions to upper lines
                                   'line',  # number of line fixation is in
                                   'line.change',
                                   # whether there was a line change from previous fixation to current fixation
                                   'sac.in',  # incoming saccade length (in letters)
                                   'sac.out',  # outgoing saccade length (in letters)
                                   'ia.reg.out',  # whether the next saccade is a regression
                                   'ia.reg.out.to',  # to which word id the eyes regressed to next
                                   'ia.reg.in',  # whether the previous saccade was a regression
                                   'ia.reg.in.from']]  # from which word id the eyes regressed from previously

        fixation_df = fixation_df.rename(columns={'ia.reg.out': 'reg.out',
                                                  'ia.reg.out.to': 'reg.out.to',
                                                  'ia.reg.in': 'reg.in',
                                                  'ia.reg.in.from': 'reg.in.from',
                                                  'uniform_id': 'participant_id'})

        # Remove quotation marks, as done for words_df
        fixation_df["ia"] = fixation_df["ia"].apply(lambda x: str(x).replace('"', ''))

        # Add participant code without str for gam analysis
        fixation_df['participant_id_int'] = [int(id.replace('en_', '')) for id in
                                             fixation_df['participant_id'].tolist()]

        # Register too short and too long fixations (outliers)
        z_scores = []
        for i, participant_fixations in fixation_df.groupby('participant_id'):
            z_scores.extend(zscore(participant_fixations['dur']))
        fixation_df['z_score'] = z_scores
        fixation_df[(fixation_df['z_score'] < -3) | (fixation_df['z_score'] > 3)].to_csv(
            f'../data/MECO/processed/fixation_outliers.csv', index=False)

        # Remove first fixation of a trial which also triggers a regression (likely noise)
        fixation_df = fixation_df.drop(fixation_df[(fixation_df['fixid'] == 1) & (fixation_df['reg.out'] == 1)].index)
        # Only keep columns we are going to use for the analysis
        fixation_df = fixation_df.drop(['xs'], axis=1)

        # Remove regression that is subsequently followed by another regression (likely noise)
        # fixation_df = self._remove_subsequent_regressions(fixation_df)

        # Deal with noise in regressions to upper lines
        fixation_df = self._remove_close_reg_upper_line(fixation_df)

        # Add median split of duration of each participant
        median_split = []
        for i, participant_fixations in fixation_df.groupby('participant_id'):
            median_split.extend(['long' if dur > median(participant_fixations['dur']) else 'short' for dur in
                                 participant_fixations['dur'].tolist()])
        fixation_df['dur.bin'] = median_split

        # Add distance in words
        in_dist, out_dist = [], []
        for id, group in fixation_df.groupby(['participant_id', 'trialid']):
            for i, ianum in enumerate(group['ianum'].tolist()):
                if i - 1 >= 0:
                    previous_ianum = group['ianum'].tolist()[i - 1]
                    in_distance = int(ianum) - int(previous_ianum)
                    in_dist.append(in_distance)
                else:
                    in_dist.append(None)
                if i + 1 < len(group['ianum'].tolist()):
                    next_ianum = group['ianum'].tolist()[i + 1]
                    out_distance = int(next_ianum) - int(ianum)
                    out_dist.append(out_distance)
                else:
                    out_dist.append(None)
        fixation_df['sac.in.dist'] = in_dist
        fixation_df['sac.out.dist'] = out_dist

        return fixation_df

    def pre_process_fixation_data(self):

        if self.corpus == 'Provo':
            data = self._pre_process_provo_fixation_data()
        elif self.corpus == 'MECO':
            data = self._pre_process_meco_fixation_data()
        else:
            raise Exception(f'Corpus {self.corpus} not supported.')
        self.data = data

    def add_variables(self, variables:list[str], frequency_filepath:str='') -> pd.DataFrame:

        """
        Add variables to the fixation dataframe.
        :param variables: list of names of variables to be added to dataframe
        :param frequency_filepath: filepath where word frequencies are located (SUBTLEX or MECO's frequency list)
        :return: dataframe with added variables
        """

        # Add length
        if 'length' in variables:
            # add length and frequency
            self.data['length'] = [len(word.strip(string.punctuation)) for word in self.data['ia'].tolist()]

        # Add frequency
        if 'frequency' in variables and frequency_filepath:

            if self.corpus == 'MECO':  # we use frequency file from meco corpus
                freq_col_name = 'zipf_freq'
                word_col_name = 'ia_clean'
                frequency_df = pd.read_csv(frequency_filepath, usecols=[freq_col_name, word_col_name])
                if 'lang' in frequency_df.columns:
                    frequency_df = frequency_df[frequency_df['lang'] == 'english']
            elif self.corpus == 'Provo':  # we use SUBTLEX-UK
                freq_col_name = 'LogFreq(Zipf)'
                word_col_name = 'Spelling'
                frequency_df = pd.read_csv(frequency_filepath, sep='\t', usecols=[freq_col_name, word_col_name],
                                           dtype={word_col_name: np.dtype(str)})
            else:
                raise NotImplementedError('Frequency resource or corpus not implemented.')

            frequency_col = []
            for word in self.data['ia'].tolist():
                word = ''.join(filter(lambda x: x.isalpha() or x.isdigit() or x.isspace(), str(word)))
                if word.isalpha():
                    word = word.lower()
                if word in frequency_df[word_col_name].tolist():
                    frequency_col.append(
                        frequency_df[freq_col_name].tolist()[frequency_df[word_col_name].tolist().index(word)])
                else:
                    frequency_col.append(None)
            self.data['frequency'] = frequency_col

        # Add N-1 variables
        if 'word-1' in variables:

            if 'length' in self.data.columns:
                lengths = []
                for i, row in self.data.iterrows():
                    if i - 1 >= 0:
                        previous_row = self.data.loc[i - 1]
                        if previous_row.participant_id == row.participant_id and previous_row.trialid == row.trialid:
                            lengths.append(len(previous_row.ia))
                        else:
                            lengths.append(np.nan)
                    else:
                        lengths.append(np.nan)
                self.data['ia.minus.one.length'] = lengths

            if 'frequency' in self.data.columns:
                frequencies = []
                for i, row in self.data.iterrows():
                    if i - 1 >= 0:
                        previous_row = self.data.loc[i - 1]
                        if previous_row.participant_id == row.participant_id and previous_row.trialid == row.trialid:
                            frequencies.append(previous_row.frequency)
                        else:
                            frequencies.append(np.nan)
                    else:
                        frequencies.append(np.nan)
                self.data['ia.minus.one.frequency'] = frequencies

        return self.data

def check_sent_length(fixation_df: pd.DataFrame):

    """"
    Check how long each sentence is in fixation report.
    """
    sent_lengths = []

    for id, group in fixation_df.groupby(['participant_id', 'trialid']):
        sent_lengths.extend(group['sentnum'].value_counts().values.tolist())

    print(np.mean(sent_lengths))
    print(np.std(sent_lengths))

def add_sent_ids_to_provo(fixation_df, words_df):

    """
    Add sentence ids to Provo fixation dataframe.
    :param fixation_df: dataframe with fixation data
    :param words_df: dataframe with words data
    :return: fixation dataframe with sentence ids added
    """

    if 'sentnum' not in fixation_df.columns:

        text_ids = []
        text_words = [set(group['ia'].tolist()) for i, group in words_df.groupby('trialid')]

        for row in fixation_df.itertuples():

            word_loc = words_df.index[(words_df['ianum'] == int(row.ianum)) & (words_df['ia'] == row.ia)].tolist()

            if word_loc and len(word_loc) == 1:
                trialid = words_df.iloc[word_loc]['trialid'].tolist()[0]
                text_ids.append(int(trialid))

            # e.g. in case the same ianum-ia combination appears more than once in words_df (ambiguous as to which text each belongs to)
            # find the text with the most overlap with words in this trial label in fixation report.
            else:
                trial_rows = fixation_df[
                    (fixation_df['participant_id'] == row.participant_id) & (
                            fixation_df['TRIAL_LABEL'] == row.TRIAL_LABEL)]
                trial_words = set(trial_rows['ia'].unique())
                overlap = []
                for words in text_words:
                    overlap.append(len(words.intersection(trial_words)))
                trialid = overlap.index(max(overlap)) + 1
                text_ids.append(trialid)

        fixation_df['trialid'] = text_ids
        fixation_df.to_csv('../data/Provo/processed/fixation_en_df.csv', index=False)
        # change ianums from text 55 to match words_df (because of error in tokens of fixation report: livre--as)
        fixation_df['ianum'] = fixation_df.apply(
            lambda x: int(x['ianum']) + 1 if (x['trialid'] == 55) & (int(x['ianum']) > 9) else int(x['ianum']), axis=1)
        # change ianums from text 36 to match words_df (because of error in tokens of fixation report: Ñ)
        fixation_df['ianum'] = fixation_df.apply(
            lambda x: int(x['ianum']) - 1 if (x['trialid'] == 36) & (int(x['ianum']) > 24) else int(x['ianum']), axis=1)
        fixation_df = fixation_df.drop(columns=['TRIAL_LABEL'])

        fixation_df.sort_values(by=['participant_id', 'trialid', 'fixid'], inplace=True)

        fixation_df = pd.merge(fixation_df, words_df[['trialid', 'ianum', 'sentnum']], how='left',
                                         on=['trialid', 'ianum'])

    return fixation_df

def check_alignment(words_df: pd.DataFrame, eye_df: pd.DataFrame):

    """
    Check alignment between word and fixation dataframes (whether word ids match).
    :param words_df: words dataframe
    :param eye_df: fixation dataframe
    """

    # create dict with text id and word id as keys and word form as value
    words_df_dict = dict()
    for trialid, group in words_df.groupby('trialid'):
        words_df_dict[trialid] = dict()
        for ia, ianum in zip(group['ia'].tolist(), group['ianum'].tolist()):
            words_df_dict[trialid][ianum] = ia

    # for each word if and word in eye-movement dataframe, check if it's the same in word dataframe
    for id, data in eye_df.groupby(['participant_id', 'trialid']):
        for eye_ia, eye_ianum in zip(data['ia'].tolist(), data['ianum'].tolist()):
            # in case word_id-word combination from eye-movement dataframe does not exist in words dataframe
            assert eye_ianum in words_df_dict[id[1]].keys(), print(
                f'Word id {eye_ianum} of text {id[1]} and participant '
                f'{id[0]} in eye-tracking data not in words dataframe;'
                f'{group}')
            # in case word from eye-movement dataframe does not match word with same id in words dataframe
            assert eye_ia == words_df_dict[id[1]][eye_ianum], print(
                f'Word {eye_ia} (id {eye_ianum} in text {id[1]} of participant '
                f'{id[0]}) in eye-tracking dataframe does not match word '
                f'of same text and id in words dataframe ({words_df_dict[id[1]][eye_ianum]}).')

def pre_process_corpus(texts_filepath:str, words_filepath:str, corpus_filepath:str, eye_filepath:str, frequency_filepath:str, corpus:str = 'MECO') -> (pd.DataFrame,pd.DataFrame):

    """
    Pre-process files from eye-tracking corpus (MECO or Provo) and generate two datasets: one where each word is a row (for surprisal and saliency computations), and one where each fixation is a row (for analysis).

    :param texts_filepath: filepath to file with texts used in corpus
    :param words_filepath: filepath where to save generated word dataframe
    :param corpus_filepath: filepath to file with eye movements from corpus
    :param eye_filepath: filepath where to save generated eye-movement dataframe
    :param frequency_filepath: filepath to word frequency resource (SUBTLEX-UK or MECO's frequency file)
    :param corpus: Provo or MECO
    :return: words_df (each word of each text as row) and eye_df (each fixation/word as a row and add variables for analysis)
    """

    # Word data
    word_data = WordData(corpus=corpus,
                         filepath=texts_filepath).create_texts_df()
    word_data.data.to_csv(words_filepath, index=False)

    # Fixation data
    fixation_obj = FixationData(corpus=corpus,
                            filepath=corpus_filepath)
    eye_data = fixation_obj.pre_process_fixation_data()
    if corpus == 'Provo':
        data = add_sent_ids_to_provo(eye_data, word_data)
        fixation_obj.data = data
    eye_data = fixation_obj.add_variables(['length', 'frequency', 'word-1'], frequency_filepath)
    eye_data.to_csv(eye_filepath, index=False)

    # Check alignment between words and fixation dataframe
    check_alignment(word_data, eye_data)

    return eye_data, word_data

