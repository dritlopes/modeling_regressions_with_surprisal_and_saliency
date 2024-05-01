import pandas as pd
# Calculate regression out rates and correlation

# merged_df = pd.read_csv("/Users/anm/code/RegressionMECO/data/MECO/merged_df.csv", sep='\t', index_col=0)
def calculate_mean_ia_regression(meco_df: pd.DataFrame):

    # Filter to remove the first 3 texts because they presented different words with the same Id (ianum)
    filt = (meco_df['trialid'] > 3)
    eight_merged_df = meco_df.loc[filt] # .loc is something specific for reading dataframe, check the tutorials

    # Filter to keep only the words that were indeed read
    read_filter = (eight_merged_df['blink'] == 0.0) & (eight_merged_df['skip'] == 0.0)
    read_eight_merged_df = eight_merged_df.loc[read_filter] # .loc is something specific for reading dataframe, check the tutorials

    # FINAL DATA WOULD HAVE 36494 ROWS

    # Regression out rates
    mean_read_eight_merged_df = read_eight_merged_df.groupby(['trialid', 'ianum', 'ia'])['reg.out'].mean()

    # mean_read_eight_merged_df.to_csv("/Users/anm/code/RegressionMECO/data/MECO/sup_reg_df.csv", sep='\t')
    return mean_read_eight_merged_df.reset_index()