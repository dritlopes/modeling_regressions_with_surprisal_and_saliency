import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# # merge saliency and word-based data
# saliency_filepath = f'GroNLP_gpt2-small-dutch_saliency.csv'
# eye_tracking_filepath = f'corpus_du_df.csv'
# saliency_df = pd.read_csv(saliency_filepath)
# saliency_df['trialid'] = saliency_df['trialid'].apply(lambda x: x + 1)
# eye_tracking_df = pd.read_csv(eye_tracking_filepath)
# df = pd.merge(eye_tracking_df, saliency_df[['trialid', 'ianum', 'saliency_mean', 'saliency_sum']], how='left', on=['trialid', 'ianum'])
# df.to_csv('eye_tracking_saliency_du.csv')
# # df = pd.read_csv('eye_tracking_saliency_du.csv')
#
# # run correlation
# df_lm = df.dropna(subset=["dur", "saliency_mean", "frequency"])
# model_sm_dur = smf.mixedlm("dur ~ saliency_mean + length + frequency", df_lm, groups=df_lm["uniform_id"]).fit()
# print(model_sm_dur.summary())
# df_glm = df.dropna(subset=["skip", "saliency_mean", "frequency"])
# model_sm_skip = sm.BinomialBayesMixedGLM.from_formula('skip ~ saliency_mean + length + frequency', {'uniform_id': '0 + C(uniform_id)'}, df_glm).fit_vb()
# print(model_sm_skip.summary())

# # correlation reg.in and saliency
# reg_in_df = pd.read_csv('../data/MECO/regression_importance.csv')
# reg_in_df['length'] = [len(ia) for ia in reg_in_df['previous.ia']] # add length as co-variate
# reg_in_df = reg_in_df.rename(columns={"reg.in":"reg_in"})
# # create a unique id for each word that triggered a regression
# ids = []
# counter = 1
# for id, group in reg_in_df.groupby(['uniform_id','trialid','ianum']):
#     ids.extend([counter for item in range(len(group))])
#     counter += 1
# assert len(ids) == len(reg_in_df['saliency'].tolist())
# reg_in_df['iaid'] = ids
# df_glm = reg_in_df.dropna(subset=["reg_in", "saliency"])
# model_sm_skip = sm.BinomialBayesMixedGLM.from_formula('reg_in ~ saliency + length', {'uniform_id:iaid': '0 + C(iaid)'}, df_glm).fit_vb()
# print(model_sm_skip.summary())