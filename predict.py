import pandas as pd


def predict_sentiment(user_id,recc_df):
    predict_df=pd.read_csv('sample30_tfidf_predict',index_col='Product')
    dataframe_df=predict_df[predict_df.index.isin(recc_df.loc[user_id].sort_values(ascending=False)[0:20].index)]
    return dataframe_df 
