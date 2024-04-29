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
            if len(token_id) > 1:
                multi_tokens.append(token_id)
        # print(multi_tokens)
        # find which locations in array represent multi-tokens
        sensitivity_merge, all_indices = list(), list()
        for multi_token in multi_tokens:
            indices = [(i, i+len(multi_token)-1) for i in range(len(token_ids)) if token_ids[i:i+len(multi_token)] == multi_token]
            # print(indices)
            for occurrence in indices:
                sensitivity_merge.append(occurrence)
                all_indices.extend(occurrence)
        # create new array with locations of each multi-token merged into one
        sensitivity_updated = []
        for i in range(len(sensitivity)):
            if i in all_indices:
                for occurrence in sensitivity_merge:
                    if i == occurrence[0]:
                        sensitivity_updated.append(np.mean(sensitivity[occurrence[0]:occurrence[-1]+1]))
            else:
                sensitivity_updated.append(sensitivity[i])
        # print(sensitivity_updated)
        # print({'word': words[word_index], 'word_id': word_ids[word_index], 'sensitivity': sensitivity_updated})
        # print()
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
    for dist_s in distributed_sensitivity:
        dist = [s for s in dist_s]
        for i in range(len(dist_s), len(words)):
            dist.append(0) # make all arrays same length (length of sequence)
        distributed_sensitivity_updated.append(dist)
    saliency = np.sum(distributed_sensitivity_updated, axis=0)

    # Taking the softmax does not make a difference for calculating correlation
    # It can be useful to scale the salience signal to the same range as the human attention
    # saliency = scipy.special.softmax(saliency)

    return tokens, token_ids, saliency[:len(tokens)], distributed_sensitivity

def extract_all_saliency(model, embeddings, tokenizer, texts, words, word_ids, outfile):

    all_text_ids, all_token_ids, all_tokens, all_saliency, all_dist_saliency = [], [], [], [], []

    for i, text in enumerate(texts):

        tokens, token_ids, saliency, dist_saliency = extract_relative_saliency(model, embeddings, tokenizer, words[i], word_ids[i])
        all_text_ids.extend([i for token in tokens])
        all_tokens.extend(tokens)
        all_token_ids.extend(token_ids)
        all_saliency.extend(saliency)
        all_dist_saliency.extend(dist_saliency)

    df = pd.DataFrame({'text_id': all_text_ids, 'token_id': all_token_ids, 'token': all_tokens, 'distributed_saliency': all_dist_saliency, 'saliency': all_saliency})
    df.to_csv(outfile)

def main():

    models = ['gpt2']
    corpora = ['MECO']
    measures = ['saliency']

    for corpus in corpora:

        corpus_df = pd.read_csv(f'../data/{corpus}/words_df.csv', index_col=0)
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
                    outfile_path = f'../data/{corpus}/{modelname}_{measure}.csv'
                    print(f'Extract saliency for {corpus} with {modelname}')
                    extract_all_saliency(model, embeddings, tokenizer, texts, words, word_ids, outfile_path)

if __name__ == '__main__':
    main()

