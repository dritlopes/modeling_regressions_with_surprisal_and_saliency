import numpy as np
import tensorflow as tf
import scipy.special
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import string

# ------------- Code based on code from Hollenstein & Beinborn (2021) -------------
# source: https://github.com/beinborn/relative_importance/blob/main/extract_model_importance/extract_saliency.py

def compute_sensitivity(model, embedding_matrix, tokenizer, words, word_ids):

    # # vocab_size = embedding_matrix.get_shape()[0]
    # vocab_size = embedding_matrix.num_embeddings
    # vocab_size = embedding_matrix.input_dim
    vocab_size = embedding_matrix.vocab_size
    sensitivity_data = []

    for word_index in range(len(words)):

        input_sequence = ' '.join(words[:word_index+1])
        token_ids = tokenizer.encode(input_sequence, add_special_tokens=False)
        target_token_index = len(token_ids)-1
        target_ids = tokenizer.encode(' ' + words[word_index], add_special_tokens=False)
        # in case target word is multi-token, take sensitivity to first token only
        if len(target_ids) > 1:
            target_token_index = len(token_ids) - len(target_ids)
        # print(input_sequence, token_ids[:target_token_index+1], tokenizer.convert_ids_to_tokens(token_ids[:target_token_index+1]), target_token_index)

        # TENSOR FOR MODEL PREDICTION
        # integers are not differentable, so use a one-hot encoding of the intput
        token_ids_tensor = tf.constant([token_ids[:target_token_index+1]], dtype='int32')
        token_ids_tensor_one_hot = tf.one_hot(token_ids_tensor, vocab_size)

        # TENSOR WITH CORRECT OUTPUT
        # To select the correct output, create a masking tensor.
        # tf.gather_nd could also be used, but this is easier.
        # output_mask = np.zeros((1, len(token_ids), vocab_size))
        output_mask = np.zeros((1, len(token_ids[:target_token_index+1]), vocab_size))
        output_mask[0, target_token_index, token_ids[target_token_index]] = 1
        output_mask_tensor = tf.constant(output_mask, dtype='float32')

        # COMPUTE GRADIENT of the logits of the correct target, w.r.t. the input
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(token_ids_tensor_one_hot)
            inputs_embeds = tf.matmul(token_ids_tensor_one_hot, embedding_matrix.weights)
            predict = model({"inputs_embeds": inputs_embeds}).logits
            predict_mask_correct_token = tf.reduce_sum(predict * output_mask_tensor)
        # compute the sensitivity and take l2 norm
        sensitivity_non_normalized = tf.norm(tape.gradient(predict_mask_correct_token, token_ids_tensor_one_hot), axis=2)
        # Normalize by the max
        sensitivity_tensor = (sensitivity_non_normalized / tf.reduce_max(sensitivity_non_normalized))
        sensitivity = sensitivity_tensor[0].numpy().tolist()
        # print(sensitivity)

        # MERGE MULTI-TOKENS IN SENSITIVITY ARRAY
        # find which words in corpus are multi-tokens
        multi_tokens = []
        for pos, word in enumerate(words[:word_index]):
            # word = word.translate(str.maketrans('', '', string.punctuation))
            if pos > 0:
                word = ' ' + word
            token_id = tokenizer.encode(word, add_special_tokens=False)
            # print(word, token_id, tokenizer.convert_ids_to_tokens(token_id))
            if len(token_id) > 1 and token_id not in multi_tokens:
                multi_tokens.append(token_id)
        # print(multi_tokens)
        # find which locations in array represent multi-tokens
        sensitivity_merge, all_indices = list(), list()
        for multi_token in multi_tokens:
            indices = [(i, i+len(multi_token)-1) for i in range(len(token_ids)) if token_ids[i:i+len(multi_token)] == multi_token]
            # print(indices)
            for occurrence in indices:
                sensitivity_merge.append(occurrence)
                # all_indices.extend([index for index in range(occurrence[0], occurrence[-1] + 1)])
        # print(sensitivity_merge)
        # check multi-tokens within other multi-tokens. Keep indices of longest multi-token.
        exclude = []
        for occurrence in sensitivity_merge:
            id_seq = list(range(occurrence[0], occurrence[-1]+1))
            # print(f'id_seq 1: {id_seq}')
            for occurrence2 in sensitivity_merge:
                id_seq2 = list(range(occurrence2[0], occurrence2[-1]+1))
                # print(f'id_seq 2: {id_seq2}')
                if len(id_seq) < len(id_seq2):
                    if any(id_seq == id_seq2[i:i + len(id_seq)] for i in range(len(id_seq2)-len(id_seq) + 1)):
                       exclude.append(id_seq)
        sensitivity_merge_updated = [i for i in sensitivity_merge if list(range(i[0],i[-1]+1)) not in exclude]
        all_indices = [index for occurrence in sensitivity_merge_updated for index in range(occurrence[0], occurrence[-1] + 1)]
        # print(all_indices)
        # create new array with locations of each multi-token merged into one
        sensitivity_updated = []
        for i in range(len(sensitivity)):
            if i in all_indices:
                for occurrence in sensitivity_merge_updated:
                    if i == occurrence[0]:
                        sensitivity_updated.append(np.mean(sensitivity[occurrence[0]:occurrence[-1]+1]))
            else:
                sensitivity_updated.append(sensitivity[i])
        # print(sensitivity_updated)
        # print(len(sensitivity_updated))
        # print({'word': words[word_index], 'word_id': word_ids[word_index], 'sensitivity': sensitivity_updated})
        # print()
        # if len(sensitivity_data) > 1 and len(sensitivity_updated) > len(sensitivity_data[-1]['sensitivity']) + 1:
        #     print('CHECK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #     exit()
        sensitivity_data.append({'word': words[word_index], 'word_id': word_ids[word_index], 'sensitivity': sensitivity_updated})

    return sensitivity_data

# We calculate relative saliency by summing the sensitivity a token has with all other tokens
def extract_relative_saliency(model, embeddings, tokenizer, words, word_ids):

    sensitivity_data = compute_sensitivity(model, embeddings, tokenizer, words, word_ids)

    distributed_sensitivity = [entry["sensitivity"] for entry in sensitivity_data]
    tokens = [entry["word"] for entry in sensitivity_data]
    token_ids = [entry["word_id"] for entry in sensitivity_data]

    # For each token, I sum the sensitivity values it has with all other tokens
    distributed_sensitivity_updated = []
    for item, dist_s in enumerate(distributed_sensitivity):
        dist = [s for s in dist_s]
        for i in range(len(dist), len(words)):
            dist.append(0) # make all arrays same length (length of sequence)
        distributed_sensitivity_updated.append(dist)
    saliency_sum = np.sum(distributed_sensitivity_updated, axis=0)
    saliency_mean = np.mean(distributed_sensitivity_updated, axis=0)

    # Taking the softmax does not make a difference for calculating correlation
    # It can be useful to scale the salience signal to the same range as the human attention
    # saliency = scipy.special.softmax(saliency)

    return tokens, token_ids, saliency_sum[:len(tokens)], saliency_mean[:len(tokens)], distributed_sensitivity

def extract_all_saliency(model, embeddings, tokenizer, texts, words, word_ids, outfile):

    all_text_ids, all_token_ids, all_tokens, all_saliency_sum, all_saliency_mean, all_dist_saliency = [], [], [], [], [], []

    for i, text in enumerate(texts):

        tokens, token_ids, saliency_sum, saliency_mean, dist_saliency = extract_relative_saliency(model, embeddings, tokenizer, words[i], word_ids[i])
        all_text_ids.extend([i for token in tokens])
        all_tokens.extend(tokens)
        all_token_ids.extend(token_ids)
        all_saliency_sum.extend(saliency_sum)
        all_saliency_mean.extend(saliency_mean)
        all_dist_saliency.extend(dist_saliency)

    df = pd.DataFrame({'text_id': all_text_ids,
                       'token_id': all_token_ids,
                       'token': all_tokens,
                       'distributed_saliency': all_dist_saliency,
                       'saliency_sum': all_saliency_sum,
                       'saliency_mean': all_saliency_mean})
    df.to_csv(outfile)
    return df

def create_saliency_dataframe(fixation_df, importance_df):

    importance_df.rename(columns={'text_id': 'trialid', 'token_id': 'ianum', 'token': 'ia'}, inplace=True)

    fixation_df = fixation_df[(fixation_df['reg.out']) == 1.0]

    fixation_importance = {'uniform_id': [],
                           'trialid': [],
                           'ianum': [],
                           'ia': [],
                           'previous.ia': [],
                           'previous.ianum': [],
                           'reg.in': [],
                           'saliency': []}

    for id, group in fixation_df.groupby(['uniform_id', 'trialid']):

        importance_df_trialid = importance_df[(importance_df['trialid'] == id[1]-1)]

        for fixated_word, fixated_word_id, reg_out_to in zip(group['ia'].tolist(), group['ianum'].tolist(), group['reg.out.to'].tolist()):

            fixated_word_df = importance_df_trialid[(importance_df_trialid['ianum'] == fixated_word_id)]
            saliency_array = fixated_word_df['distributed_saliency'].tolist()[0]

            if type(saliency_array) == str:
                saliency_array = saliency_array.replace('[','').replace(']','')
                saliency_array = saliency_array.split(',')
                saliency_array = [float(s) for s in saliency_array[:-1]]

            fixation_importance['saliency'].extend(saliency_array)
            fixation_importance['previous.ianum'].extend(importance_df_trialid['ianum'].tolist()[:len(saliency_array)])
            fixation_importance['previous.ia'].extend(importance_df_trialid['ia'].tolist()[:len(saliency_array)])
            fixation_importance['ianum'].extend([fixated_word_id for i in range(len(saliency_array))])
            fixation_importance['ia'].extend([fixated_word for i in range(len(saliency_array))])
            fixation_importance['trialid'].extend([id[1] for i in range(len(saliency_array))])
            fixation_importance['uniform_id'].extend([id[0] for i in range(len(saliency_array))])

            reg_in = []
            for previous_id in importance_df_trialid['ianum'].tolist()[:len(saliency_array)]:
                if previous_id == reg_out_to:
                    reg_in.append(1)
                else:
                    reg_in.append(0)
            fixation_importance['reg.in'].extend(reg_in)

    fixation_importance_df = pd.DataFrame.from_dict(fixation_importance)
    fixation_importance_df.to_csv('regression_importance.csv')

def main():

    models = ['gpt2']
    corpora = ['MECO']
    measures = ['saliency']

    for corpus in corpora:

        # corpus_df = pd.read_csv(f'../data/MECO/words_df.csv')
        corpus_df = pd.read_csv(f'../data/MECO/words_df.csv', index_col=0)
        fixation_df = pd.read_csv('../data/MECO/fixation_en_df.csv')

        texts = corpus_df.texts.unique()
        words, word_ids = [], []
        for text, group in corpus_df.groupby('trialid'):
            words.append(group['ia'].tolist())
            word_ids.append(group['ianum'].tolist())

        for modelname in models:

            if modelname == 'gpt2':
                model = TFGPT2LMHeadModel.from_pretrained(modelname, output_attentions=True)
                tokenizer = GPT2Tokenizer.from_pretrained(modelname)
                embeddings = model.get_input_embeddings()

                for measure in measures:
                    outfile_path = f'{modelname}_{measure}.csv'
                    print(f'Extract saliency for {corpus} with {modelname}')
                    saliency_df = extract_all_saliency(model, embeddings, tokenizer, texts, words, word_ids, outfile_path)
                    create_saliency_dataframe(fixation_df, saliency_df)

if __name__ == '__main__':
    main()

