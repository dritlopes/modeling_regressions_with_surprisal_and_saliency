from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from torch import nn
import string
import pandas as pd

# Calculate the surprisal value for each word from original texts (df)
def calculate_surprisal_values(df: pd.DataFrame, corpus_name, model_name):

    # Here some lines were adapted from Dri's PROVO script
    # Initialize language model #see https://huggingface.co/docs/transformers/model_doc/gpt2 for gpt2 documentation

    if model_name == 'gpt2':
        MODEL_NAME = model_name
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    # Process text with language model
    # previous_context = ""  # cumulator, to start a for loop you need an empty variable to include something in each loop
    model_tokens, corpus_tokens = [], [] # lists to save which words in the corpus are multi-tokens in the model
    surprisal_values = []

    # df = df.loc[(df['uniform_id'] == 'en_10') & (df['trialid'] == 1)]

    # for participant_and_text, rows in df.groupby(['uniform_id', 'trialid']):

    # for index, row in df.iterrows():  # for an id that I will not use "index"", check each line

    for text, rows in df.groupby('trialid'):

        previous_context = ''

        for i, next_word in enumerate(rows['ia'].tolist()):

            if i == 0:
                surprisal_values.append(None)  # first word in text does not have context to compute surprisal
                previous_context = next_word

            else:
                next_word = ' ' + next_word
                next_word = next_word.strip(string.punctuation)
                # this line takes the first word and tokenizes it in the form PyTorch
                encoded_input = tokenizer(previous_context, return_tensors='pt')
                # the list of IDs from the tokenizer
                next_word_id = tokenizer(next_word, return_tensors='pt')["input_ids"][0]
                # get GPT2 output, see https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel for details on the output
                model.eval()  # turn off dropout layers
                output = model(**encoded_input)
                # logits are scores from output layer of shape (batch_size, sequence_length, vocab_size)
                logits = output.logits[:, -1, :]
                # convert raw scores into probabilities (between 0 and 1)
                probabilities = nn.functional.softmax(logits, dim=1)  # softmax transforms the values from logits into percentages
                # take probability of next token in the text (averaging probabilities for multi-token words)
                token_probabs = []
                for token_id in next_word_id:
                    probability = probabilities[0, token_id]
                    probability = probability.detach().numpy()
                    token_probabs.append(probability)
                probability = np.mean(token_probabs)
                # convert probability into surprisal
                surprisal = -np.log2(probability)
                surprisal_values.append(surprisal)
                # increase context for next surprisal
                previous_context = previous_context + next_word
                # check which words in the corpus are multi-tokens in the model
                if len(next_word_id) > 1:
                    corpus_tokens.append(next_word)
                    model_tokens.append([tokenizer.decode(token_id) for token_id in
                                         next_word_id])

        # ianum = row["ianum"]  # the id from df
        # if ianum == 1:  # ignore the first word, because the first word would never have a surprisal value, it needs context
        #     previous_context = row["ia"]  # but use it as context
        #     continue
        # next_word = ' ' + row["ia"]  # to construct a new text use it with space
        # next_word = next_word.strip(string.punctuation)
        # # this line takes the first word and tokenizes it in the form PyTorch
        # encoded_input = tokenizer(previous_context, return_tensors='pt')
        # # the same for second word and inputs_ids is the reference number for an individual token that comes from
        # # the list of IDs from the tokenizer, and [0] is to select only the first one
        # next_word_id = tokenizer(next_word, return_tensors='pt')["input_ids"][0]
        # # check which words in the corpus are multi-tokens in the model
        # if len(tokenizer(next_word, return_tensors='pt')["input_ids"][0]) > 1:
        #     corpus_tokens.append(next_word)
        #     model_tokens.append([tokenizer.decode(token_id) for token_id in tokenizer(next_word, return_tensors='pt')["input_ids"][0]])
        # # get GPT2 output, see https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel for details on the output
        # # model.eval()  # turn off dropout layers
        # # output = model(**encoded_input)
        # # # logits are scores from output layer of shape (batch_size, sequence_length, vocab_size)
        # # logits = output.logits[:, -1, :]
        # # # convert raw scores into probabilities (between 0 and 1)
        # # probabilities = nn.functional.softmax(logits, dim=1)  # softmax transforms the values from logits into percentages
        # # # take probability of next word in the text
        # # probability = probabilities[0, next_word_id]
        # # probability = probability.detach().numpy()[0]
        # # # print(probability)
        # # # convert probability into surprisal
        # # surprisal = -np.log2(probability)
        # # # print(surprisal)
        # # check which words in the corpus are multi-tokens in the model
        # if len(tokenizer(next_word, return_tensors='pt')["input_ids"][0]) > 1:
        #     corpus_tokens.append(next_word)
        #     model_tokens.append([tokenizer.decode(token_id) for token_id in
        #                          tokenizer(next_word, return_tensors='pt')["input_ids"][0]])
        # # # to add the surprisal values to the cumulator df[surprisal] it's a new column in the dataframe
        # # df.at[index, 'surprisal'] = surprisal
        # previous_context = previous_context + next_word

    df['surprisal'] = surprisal_values
    df.to_csv("../data/MECO/surprisal_df2.csv", sep='\t')

    # write out which words in the corpus are multi-tokens in the model
    with open(f'../data/{corpus_name}/multi_tokens_{model_name}.csv', 'w') as outfile:
        outfile.write(f'CORPUS_TOKEN\tMODEL_TOKEN\n')
        for model_token, corpus_token in zip(model_tokens, corpus_tokens):
            outfile.write(f'{corpus_token}\t{model_token}\n')

    return df