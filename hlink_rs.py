# Libraries for data preparation & visualization
import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio
import random
import argparse
pio.renderers.default = "png"


import warnings 
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, SVD, SVDpp,NormalPredictor
from surprise import accuracy

def filter_languages(df, languages=[]):
    if not languages:
        # No languages specified, return the original DataFrame
        return df
    else:
        # Construct the filtering condition dynamically based on specified languages
        condition = df['Language'].isin(languages)
        filtered_df = df[condition]
        return filtered_df

def preprocessing(languages=[]):
    # Define column names for your data (replace with your actual column names)
    column_names_hlink_type = ["Hlink_type_ID", "Title"]
    column_names_rating = ["Article_ID", "Hlink_type_ID", "Rating"]
    column_names_article = ["Article_ID", "Q_ID", "Language"]#"Title_a",

    # Load the data with specified column names
    article = pd.read_csv("Hlink_RS/Data/Articles.csv", names=column_names_article)
    hlink_type = pd.read_csv("Hlink_RS/Data/Hlink_types.csv", names=column_names_hlink_type)
    rating = pd.read_csv("Hlink_RS/Data/Hlink_type_ratings.csv", names=column_names_rating)
    #Choose languages for dataset
    
    # article = article[(article['Language'] == 'vi')]#| (article['Language'] == 'ja')]
    article = filter_languages(article,languages)
    # Merge rating with article and hyperlink data
    rating = rating.merge(hlink_type, on="Hlink_type_ID")[['Article_ID','Hlink_type_ID', 'Title', 'Rating']]
    rating = rating.merge(article, on="Article_ID")[['Article_ID','Q_ID','Hlink_type_ID', 'Title','Language', 'Rating']]
    #Filter duplicates
    rating.drop_duplicates(inplace=True)
    #Filter with rating from 1 to 10:
    rating = rating[rating['Rating'] < 11] 
    return rating

def statistic(languages=[]):

    # Count the number of rating in languages
    rating = preprocessing(languages)
    print("number of ratings")
    for i in languages:
        print(len(rating[(rating['Language'] == i)]))
     
    # Count the distinct hyperlink types in languages
    distinct_hyperlinks_by_language = rating.groupby('Language')['Hlink_type_ID'].nunique().reset_index()
    distinct_hyperlinks_by_language.columns = ['Language', 'Distinct_Hyperlink_Count']
    print(distinct_hyperlinks_by_language)

    
def build():
    #Reader
    rating = preprocessing()
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(rating[['Article_ID', 'Title', 'Rating']], reader)
    # Model 
    raw_ratings = data.raw_ratings
    random.shuffle(raw_ratings)                 # shuffle dataset
    #Calculate threshold for train and test data:
    threshold   = int(len(raw_ratings)*0.8)

    train_raw_ratings = raw_ratings[:threshold] # 80% of data is trainset

    test_raw_ratings  = raw_ratings[threshold:] # 20% of data is testset

    data.raw_ratings = train_raw_ratings        # data is now the trainset
    trainset         = data.build_full_trainset() 
    testset          = data.construct_testset(test_raw_ratings)
    train_df = pd.DataFrame(trainset.all_ratings(), columns=['Article_ID', 'Title', 'Rating'])

    # Convert the testset to a DataFrame
    test_df = pd.DataFrame(testset, columns=['Article_ID', 'Title', 'Rating'])

    # sim_options = {
    #     "name": "cosine",
    #     "user_based": True,  # compute  similarities between items
    # }
    # models=[KNNBasic(sim_options=sim_options),KNNWithMeans(sim_options=sim_options),KNNWithZScore(sim_options=sim_options),KNNBaseline(sim_options=sim_options),SVD(),SVDpp(), NormalPredictor()] #SlopeOne(),
    models=[KNNBasic(),KNNWithMeans(),KNNWithZScore(),KNNBaseline(),SVD(),SVDpp(), NormalPredictor()] #SlopeOne(),

    results = {}

    for model in models:
        # perform 5 fold cross validation
        # evaluation metrics: mean absolute error & root mean square error
        CV_scores = cross_validate(model, data, measures=["MAE","RMSE"], cv=5, n_jobs=-1)  
        
        # storing the average score across the 5 fold cross validation for each model
        result = pd.DataFrame.from_dict(CV_scores).mean(axis=0).\
                rename({'test_mae':'MAE', 'test_rmse': 'RMSE'})
        results[str(model).split("algorithms.")[1].split("object ")[0]] = result

    performance_df = pd.DataFrame.from_dict(results)
    print("Model Performance: \n")
    print(performance_df.T.sort_values(by='MAE'))

# Function to generate recommendations
def generate_recommendations(model, user_id, k=10):
    user_items = set([item for item, _ in trainset.ur[user_id]])
    all_items = set([i for i in range(trainset.n_items)])
    non_interacted_items = list(all_items - user_items)
    
    predictions = []
    for item_id in non_interacted_items:
        prediction = model.predict(uid=user_id, iid=item_id)
        predictions.append((item_id, prediction.est))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    return [item_id for item_id, _ in predictions[:k]]



def main():
    parser = argparse.ArgumentParser(description='Code of Languges')
    parser.add_argument('--languages', nargs='+', type=str, help='List of languages')
    parser.add_argument('--action', choices=['statistic', 'build'], default='statistic', help='Choose the action to perform')

    args = parser.parse_args()

    args = parser.parse_args()

    languages = args.languages

    
    if args.action == 'statistic':
        if languages:
            print(f'List of languages: {languages}')
            statistic(languages)
        else:
            print('No given languages')
    elif args.action == 'build':
        if languages:
            print(f'List of languages: {languages}')
            build(languages)
        else:
            print('No given languages')

if __name__ == '__main__':
    main()

#python hlink_rs.py --languages en ja vi --action statistic
#python hlink_rs.py --languages en ja vi --action build

