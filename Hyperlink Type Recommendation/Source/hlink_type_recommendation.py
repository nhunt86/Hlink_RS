import numpy as np
import pandas as pd
import random
import argparse
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, SVD, NormalPredictor
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
    article = pd.read_csv("../Data/Rich_case/Articles.csv", names=column_names_article)
    hlink_type = pd.read_csv("../Data/Rich_case/Hlink_types.csv", names=column_names_hlink_type)
    rating = pd.read_csv("../Data/Rich_case/Ratings.csv", names=column_names_rating)

    #Choose languages for dataset
    article = filter_languages(article,languages)
    
    # Merge rating with article and hyperlink data
    rating = rating.merge(hlink_type, on="Hlink_type_ID")[['Article_ID','Hlink_type_ID', 'Title', 'Rating']]
    rating = rating.merge(article, on="Article_ID")[['Article_ID','Q_ID','Hlink_type_ID', 'Title','Language', 'Rating']]
    #Filter duplicates
    rating.drop_duplicates(inplace=True)
    
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
    random.shuffle(raw_ratings)                 
    #Calculate threshold for train and test data:
    threshold   = int(len(raw_ratings)*0.8)

    train_raw_ratings = raw_ratings[:threshold] # 80% of data is trainset

    test_raw_ratings  = raw_ratings[threshold:] # 20% of data is testset

    data.raw_ratings = train_raw_ratings        
    trainset         = data.build_full_trainset() 
    testset          = data.construct_testset(test_raw_ratings)
    train_df = pd.DataFrame(trainset.all_ratings(), columns=['Article_ID', 'Title', 'Rating'])

    # Convert the testset to a DataFrame
    test_df = pd.DataFrame(testset, columns=['Article_ID', 'Title', 'Rating'])
    # Similarities between items
    sim_options = {
        "name": "msd",
        "user_based": False,  
    }
    models=[KNNBasic(k=50,sim_options=sim_options),KNNWithMeans(k=50,sim_options=sim_options),KNNWithZScore(k=50,sim_options=sim_options),KNNBaseline(k=50,sim_options=sim_options),SVD()]
    # models=[KNNBasic(),KNNWithMeans(),KNNWithZScore(),KNNBaseline(),SVD(),SVDpp(), NormalPredictor()] #SlopeOne(),

    results = {}

    for model in models:
        # perform 5 fold cross validation and caclculate mae and rmse of models
    
        CV_scores = cross_validate(model, data, measures=["MAE","RMSE"], cv=5, n_jobs=-1)  
        result = pd.DataFrame.from_dict(CV_scores).mean(axis=0).\
                rename({'test_mae':'MAE', 'test_rmse': 'RMSE'})
        results[str(model).split("algorithms.")[1].split("object ")[0]] = result

    performance_df = pd.DataFrame.from_dict(results)
    print("Model Performance: \n")
    print(performance_df.T.sort_values(by='MAE'))


#Tunning
def finetune(languages=[], trainset, testset):
    data = preprocessing(languages)
    param_grid = { 'sim_options' : {'name': ['msd','cosine'], \
                                'min_support': [3,5], \
                                'user_based': [False, True]}
             }

    gridsearchKNNWithMeans = GridSearchCV(KNNBaseline, param_grid, measures=['mae', 'rmse'], \
                                        cv=5, n_jobs=-1)
                                        
    gridsearchKNNWithMeans.fit(data)

    print(f'MAE Best Parameters:  {gridsearchKNNWithMeans.best_params["mae"]}')
    print(f'MAE Best Score:       {gridsearchKNNWithMeans.best_score["mae"]}\n')

    print(f'RMSE Best Parameters: {gridsearchKNNWithMeans.best_params["rmse"]}')
    print(f'RMSE Best Score:      {gridsearchKNNWithMeans.best_score["rmse"]}\n')
    sim_options = {'name':'cosine','min_support':3,'user_based':False}
    final_model = KNNBaseline(k=50,sim_options=sim_options)

    # Fitting the model on trainset & predicting on testset, printing test accuracy
    pred = final_model.fit(trainset).test(testset)

    print(f'\nUnbiased Testing Performance:')
    print(f'MAE: {accuracy.mae(pred)}, RMSE: {accuracy.rmse(pred)}')


# Generate recommendations
def KNN_Recommendation(trainset,userID, like_recommend=5):
    # Compute item based similarity matrix
    sim_options       = {'name':'msd','min_support':3,'user_based':False}
    similarity_matrix = KNNBaseline(k=50,sim_options=sim_options).fit(trainset).\
                        compute_similarities() 
    # converts the raw userID to innerID
    userID      = trainset.to_inner_uid(userID)    
    userRatings = trainset.ur[userID]              
    
    # sort rating by decreasing order. Then top 'like_recommend' items & ratings are extracted
    
    temp_df = pd.DataFrame(userRatings).sort_values(by=1, ascending=False).\
              head(like_recommend)
    userRatings = temp_df.to_records(index=False) 
    
    recommendations   = {}

    for user_top_item, user_top_item_rating  in userRatings:

        all_item_indices          =   list(pd.DataFrame(similarity_matrix)[user_top_item].index)
        all_item_weighted_rating  =   list(pd.DataFrame(similarity_matrix)[user_top_item].values*\
                                          user_top_item_rating)
        
        all_item_weights          =   list(pd.DataFrame(similarity_matrix)[user_top_item].values)
        
        for index in range(len(all_item_indices)):
            if index in recommendations:
                # sum of weighted ratings
                recommendations[index] += all_item_weighted_rating[index]        
            else:                        
                recommendations[index]  = all_item_weighted_rating[index]

    
    for index in range(len(all_item_indices)):                               
            if all_item_weights[index]  !=0:
                # final ratings (sum of weighted ratings/sum of weights)
                recommendations[index]   =recommendations[index]/\
                                          (all_item_weights[index]*like_recommend)
                    
    temp_df = pd.Series(recommendations).reset_index().sort_values(by=0, ascending=False)
    recommendations = list(temp_df.to_records(index=False))
    final_recommendations = []
    count = 0
    
    for item, score in recommendations:
        flag = True
        for userItem, userRating in trainset.ur[userID]:
            if item == userItem: 
                flag = False       # If item in recommendations has not been rated by user, 
                break              # add to final_recommendations
        if flag == True:
            final_recommendations.append(trainset.to_raw_iid(item)) 
            count +=1              # trainset has the items stored as inner id,  
                                   # convert to raw id & append 
      
    return(final_recommendations)

def rec(languages = []):

    rating = preprocessing(languages)
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(rating[['Article_ID','Title', 'Rating']], reader)
    trainset = data.build_full_trainset()

    min_article_id = rating['Article_ID'].min()
    print(min_article_id)
   
    max_article_id = rating['Article_ID'].max()
    print(max_article_id)
    article_id_range = range(min_article_id, max_article_id + 1)
    total_article_ids = rating['Article_ID'].nunique()
    print(total_article_ids)
    sample_size = 50
    # List of user IDs for which you want to generate recommendations
    user_ids_to_recommend = random.sample(article_id_range, sample_size)
    # filtered_user_ids = []
    # for user_id in user_ids_to_recommend:
    #     if user_id < 1001 or user_id > 2000:# for ja + vi data
    #         filtered_user_ids.append(user_id)
    print(user_ids_to_recommend)
    l = {}
    for userID in user_ids_to_recommend:#filtered_user_ids:#
        l[userID]= KNN_Recommendation(trainset,userID,like_recommend=5)
        
        
    #Print user and number of recommendation
    for k,v in l.items():
        print(k,len(v))
    
 
def mae_rmse_user_trained(languages=[], user_range=(0, 999)):
    rating = preprocessing(languages)
    reader = Reader(rating_scale=(1, 10))
    data = Dataset.load_from_df(rating[['Article_ID', 'Hlink_type_ID', 'Rating']], reader)

    # Run cross-validation
    model = SVD()#KNNBasic(K=50)
    cv_results = cross_validate(model, data, measures=["MAE", "RMSE"], cv=5, n_jobs=-1, return_train_measures=True)
    
    trainset = data.build_full_trainset()
    model.fit(trainset)
    testset = trainset.build_testset()
    predictions = model.test(testset)

    # Create a dictionary to store 
    user_true_ratings = defaultdict(list)
    user_pred_ratings = defaultdict(list)

    for pred in predictions:
        if int(pred.uid) in range(user_range[0], user_range[1] + 1):
            user_true_ratings[pred.uid].append(pred.r_ui)
            user_pred_ratings[pred.uid].append(pred.est)
    
    # Calculate MAE và RMSE values of users in a range
    
    user_metrics = {}
    for user_id in user_true_ratings:
        mae = mean_absolute_error(user_true_ratings[user_id], user_pred_ratings[user_id])
        rmse = mean_squared_error(user_true_ratings[user_id], user_pred_ratings[user_id], squared=False)
        user_metrics[user_id] = {'MAE': mae, 'RMSE': rmse}
    print(len(user_metrics))
    
    # Average MAE and RMSE values of users
    mae_values = [metrics['MAE'] for metrics in user_metrics.values()]
    rmse_values = [metrics['RMSE'] for metrics in user_metrics.values()]

    average_mae = sum(mae_values) / len(mae_values) if mae_values else None
    average_rmse = sum(rmse_values) / len(rmse_values) if rmse_values else None

    print(f'Average MAE for users in range {user_range}: {average_mae}')
    print(f'Average RMSE for users in range {user_range}: {average_rmse}')

def main():
    parser = argparse.ArgumentParser(description='Code of Languges')
    parser.add_argument('--languages', nargs='+', type=str, help='List of languages')
    parser.add_argument('--action', choices=['statistic', 'build','rec','mae_rmse_user_trained'], default='statistic', help='Choose the action to perform')

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
    elif args.action == 'rec':
        if languages:
            print(f'List of languages: {languages}')
            rec(languages)
        else:
            print('No given languages')
     elif args.action == 'mae_rmse_user_trained':
        if languages:
            print(f'List of languages: {languages}')
            mae_rmse_user_trained(languages)
        else:
            print('No given languages')


if __name__ == '__main__':
    main()

#python hlink_recommendation.py --languages en ja vi --action statistic
#python hlink_hlink_recommendation.py --languages en ja vi --action build

