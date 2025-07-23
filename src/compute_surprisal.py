from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from torch import nn
import string
import pandas as pd

# Calculate the surprisal value for each word from original texts (df)
def calculate_surprisal_values(df: pd.DataFrame, corpus_name:str, model_name:str)->pd.DataFrame:

    """
    Compute surprisal values for each word in dataset.
    :param df: words dataset
    :param corpus_name: name of eye-tracking corpus
    :param model_name: name of langauge model with which to compute surprisal
    :return: word dataframe with surprisal values
    """

    #see https://huggingface.co/docs/transformers/model_doc/gpt2 for gpt2 documentation

    if 'gpt2' in model_name:
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    else:
        raise Exception('Model name must be gpt2 or gpt2-large.')

    # Process text with language model
    model_tokens, corpus_tokens = [], [] # lists to save which words in the corpus are multi-tokens in the model
    surprisal_values = []

    print(f'Extracting surprisal values from {model_name}...')
    for text, rows in df.groupby('trialid'):

        previous_context = ''

        for i, next_word in enumerate(rows['ia'].tolist()):

            if i == 0:
                surprisal_values.append(None)  # first word in text does not have context to compute surprisal
                previous_context = next_word

            else:
                next_word = ' ' + next_word
                next_word = next_word.strip(string.punctuation)
                # tokenize next word
                next_word_id = tokenizer(next_word, return_tensors='pt')["input_ids"][0]

                # to deal with multi-token words
                total_word_surprisal = 0.0
                for token_id in next_word_id:
                    # tokenize previous context
                    encoded_input = tokenizer(previous_context, return_tensors='pt')
                    # turn off dropout layers
                    model.eval()
                    output = model(**encoded_input)
                    # logits are scores from output layer of shape (batch_size, sequence_length, vocab_size)
                    logits = output.logits[:, -1, :]
                    # convert raw scores into probabilities (between 0 and 1)
                    probabilities = nn.functional.softmax(logits,
                                                          dim=1)  # softmax transforms the values from logits into percentages
                    next_token_prob = probabilities[0, token_id]
                    next_token_prob = next_token_prob.cpu().detach().numpy()
                    surprisal = -np.log2(next_token_prob)
                    total_word_surprisal += surprisal
                    previous_context += tokenizer.decode([token_id])
                surprisal_values.append(total_word_surprisal)

                # check which words in the corpus are multi-tokens in the model
                if len(next_word_id) > 1:
                    corpus_tokens.append(next_word)
                    model_tokens.append([tokenizer.decode(token_id) for token_id in
                                         next_word_id])

    df['surprisal'] = surprisal_values

    # write out which words in the corpus are multi-tokens in the model
    with open(f'../data/{corpus_name}/processed/multi_tokens_{model_name}.csv', 'w') as outfile:
        outfile.write(f'CORPUS_TOKEN\tMODEL_TOKEN\n')
        for model_token, corpus_token in zip(model_tokens, corpus_tokens):
            outfile.write(f'{corpus_token}\t{model_token}\n')

    return df