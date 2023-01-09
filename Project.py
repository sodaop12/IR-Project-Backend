import json
import pickle
import string
import lightgbm as lgb
import optuna
import pandas as pd
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from nltk import PorterStemmer, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from elasticsearch import Elasticsearch
import numpy as np
from sklearn.model_selection import train_test_split

import M3 as m3

app = Flask(__name__)
CORS(app, origins=['http://localhost:8080'], methods=['GET', 'POST'])
app.secret_key = 'eyJhbGciOiJIUzI1NiJ9.eyJSb2xlIjoiQWRtaW4iLCJJc3N1ZXIiOiJJc3N1ZXIiLCJVc2VybmFtZSI6IkphdmFJblVzZSIsImV4cCI6MTY3MjAzNTYwNiwiaWF0IjoxNjcyMDM1NjA2fQ.tnrAPu4Z7B2Oq8OFrLR4lp1pFSWAstv_QMeC4MenqOs'
@app.route('/login', methods=['POST'])
def login():
    # Get the username and password from the request
    data = request.get_json()
    username = data['username']
    password = data['password']

    # Validate the user's credentials
    if valid_login(username, password):
        # Return a success response
        session['logged_in'] = True
        return jsonify({'success': True})
    else:
        # Return an error response
        return jsonify({'success': False, 'error': 'Invalid username or password'})

@app.route('/logout', methods=['POST'])
def logout():
    # Clear the user's login status
    clear_login_status()
    return jsonify({'success': True})

def valid_login(username, password):
    # Replace this with your own logic to validate the user's credentials
    valid_credentials = [
        {'username': 'user1', 'password': 'pass1'},
        {'username': 'user2', 'password': 'pass2'}
    ]
    for cred in valid_credentials:
        if cred['username'] == username and cred['password'] == password:
            return True
    return False

def clear_login_status():
    # Replace this with your own logic to clear the user's login status
    session.pop('logged_in', None)

def preProcess(s):
    ps = PorterStemmer()
    s = word_tokenize(s)
    stopwords_set = set(stopwords.words())
    s = [w for w in s if w not in stopwords_set]
    # s = [w for w in s if not w.isdigit()]
    s = [ps.stem(w) for w in s]
    s = ' '.join(s)
    return s

def getanimedata():
    anime = pd.read_csv('resource/anime_withpic.csv')
    rating = pd.read_csv('resource/anime_rating_1000_users.csv')
    anime_features = ['mal_id','title','type','score','status','duration','studios','synopsis','main_picture']
    anime = anime[anime_features]

    merged_df = anime.merge(rating, left_on='mal_id', right_on='anime_id', how='inner')
    genre_names = ['Action', 'Adventure','Comedy', 'Drama','Sci-Fi',
                   'Game', 'Space', 'Music', 'Mystery', 'School', 'Fantasy',
                   'Horror', 'Kids', 'Sports', 'Magic', 'Romance',]
    return genre_names,merged_df,anime,rating

def SerchanimeByName():
    genre_names, merged_df, anime, rating = getanimedata()
    anime['title'] = anime['title'].apply(preProcess)
    AnimeName = anime['title']
    vectorizer = TfidfVectorizer()
    BM25 = m3.BM25
    bm25AnimeName = BM25(vectorizer)
    bm25AnimeName.fit(AnimeName)
    query = ("full")
    score = bm25AnimeName.transform(query)
    rank = np.argsort(score)[::-1]
    print(anime.iloc[rank[:10]].to_markdown())
    pickle.dump(bm25AnimeName, open('resource/Bm25SerchByName.pkl', 'wb'))


def SerchanimeBySynopsis():
    genre_names, merged_df, anime, rating = getanimedata()
    anime['synopsis'] = anime['synopsis'].astype(str)
    anime['synopsis'] = anime['synopsis'].apply(preProcess)
    AnimeSynopsis = anime['synopsis']
    vectorizer = TfidfVectorizer()
    BM25 = m3.BM25
    bm25AnimeSynopsis = BM25(vectorizer)
    bm25AnimeSynopsis .fit(AnimeSynopsis)
    query = ("Shinji Ikari is left emotionally comatose after the death of a dear friend.")
    score = bm25AnimeSynopsis .transform(query)
    rank = np.argsort(score)[::-1]
    print(anime.iloc[rank[:10]].to_markdown())
    pickle.dump(bm25AnimeSynopsis, open('resource/Bm25SerchBySynopsis.pkl', 'wb'))
def learningtorank():
    anime = pd.read_csv('resource/anime.csv')
    rating = pd.read_csv('resource/anime_rating_1000_users.csv')
    anime_features = ['MAL_ID', 'English name', 'Japanese name', 'Score', 'Genres', 'Popularity', 'Members','Name',
                      'Favorites', 'Watching', 'Completed', 'On-Hold', 'Dropped', 'Score-1', 'Score-2', 'Score-3',
                      'Score-4', 'Score-5', 'Score-6', 'Score-7', 'Score-8', 'Score-9', 'Score-10', ]

    anime = anime[anime_features]

    merged_df = anime.merge(rating, left_on='MAL_ID', right_on='anime_id', how='inner')
    genre_names = ['Action', 'Adventure','Comedy', 'Drama','Sci-Fi',
                   'Game', 'Space', 'Music', 'Mystery', 'School', 'Fantasy',
                   'Horror', 'Kids', 'Sports', 'Magic', 'Romance',]
    return genre_names,merged_df,anime,rating

def genre_to_category(df):
    genre_names,merged_df,anime,rating = learningtorank()
    d = {name: [] for name in genre_names}
    def f(row):
      genres = row.Genres.split(',')
      for genre in genre_names:
         if genre in genres:
            d[genre].append(1)
         else:
            d[genre].append(0)

    df.apply(f, axis=1)

    genre_df = pd.DataFrame(d, columns=genre_names)
    df = pd.concat([df, genre_df], axis=1)
    return df

def make_anime_feature(df):
    df['Score'] = df['Score'].apply(lambda x: np.nan if x == 'Unknown' else float(x))
    for i in range(1, 11):
        df[f'Score-{i}'] = df[f'Score-{i}'].apply(lambda x: np.nan if x == 'Unknown' else float(x))


    df = genre_to_category(df)
    return  df
def make_user_feature(df):
    df['rating_count'] = df.groupby('user_id')['anime_id'].transform('count')
    df['rating_mean'] = df.groupby('user_id')['rating'].transform('mean')
    return df

def preprocesst(merged_df):
    merged_df = make_anime_feature(merged_df)
    merged_df = make_user_feature(merged_df)
    return merged_df
def prepare():
    SEED = 0
    genre_names, merged_df, anime, rating = learningtorank()
    merged_df = preprocesst(merged_df)
    train, test = train_test_split(merged_df, test_size=0.2, random_state=SEED)

    features = ['Score', 'Popularity', 'Members', 'Favorites', 'Watching', 'Completed', 'On-Hold', 'Dropped',
                'Score-1', 'Score-2', 'Score-3', 'Score-4', 'Score-5', 'Score-6', 'Score-7', 'Score-8', 'Score-9',
                'Score-10', 'rating_count', 'rating_mean']

    features += genre_names
    user_col = 'user_id'
    item_col = 'anime_id'
    target_col = 'rating'

    train = train.sort_values('user_id').reset_index(drop=True)
    test = test.sort_values('user_id').reset_index(drop=True)

    train_query = train[user_col].value_counts().sort_index()
    test_query = test[user_col].value_counts().sort_index()
    return features,train,train_query,test,test_query,target_col
def objectives(trial):
    SEED = 10
    features, train, train_query, test, test_query, target_col = prepare()
    # search param
    param = {
        'reg_alpha': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1),
        #'subsample': trial.suggest_uniform('subsample', 1e-8, 1),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    model = lgb.LGBMRanker(n_estimators=1000, **param, random_state=SEED,)
    model.fit(
        train[features],
        train[target_col],
        group=train_query,
        eval_set=[(test[features], test[target_col])],
        eval_group=[list(test_query)],
        eval_at=[1, 3, 5, 10, 20],  # calc validation ndcg@1,3,5,10,20
        early_stopping_rounds=50,
        verbose=10
    )
    scores = []
    for name, score in model.best_score_['valid_0'].items():
        scores.append(score)
    return np.mean(scores)

def test():
    SEED = 10
    features, train, train_query, test, test_query, target_col = prepare()
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED)  # fix random seed
                                )
    study.optimize(objectives, n_trials=10)
    best_params = study.best_trial.params
    model = lgb.LGBMRanker(n_estimators=1000, **best_params, random_state=SEED,)
    model.fit(
        train[features],
        train[target_col],
        group=train_query,
        eval_set=[(test[features], test[target_col])],
        eval_group=[list(test_query)],
        eval_at=[1, 3, 5, 10, 20],
        early_stopping_rounds=50,
        verbose=10
    )

    model.predict(test.iloc[:10][features])


def predict(user_df,top_k, anime, rating):
    genre_names, merged_df, anime, rating = learningtorank()
    features, train, train_query, test, test_query, target_col = prepare()
    model = pickle.load(open('resource/modelanime.pkl', 'rb'))
    user_anime_df = anime.merge(user_df, left_on='MAL_ID', right_on='anime_id')
    user_anime_df = make_anime_feature(user_anime_df)
    excludes_genres = list(np.array(genre_names)[np.nonzero([user_anime_df[genre_names].sum(axis=0) <= 1])[1]])
    pred_df = make_anime_feature(anime.copy())
    pred_df = pred_df.loc[pred_df[excludes_genres].sum(axis=1) == 0]

    for col in user_df.columns:
       if col in features:
          pred_df[col] = user_df[col].values[0]

    preds = model.predict(pred_df[features])
    pickle.dump(model, open('resource/modelanime.pkl', 'wb'))
    topk_idx = np.argsort(preds)[::-1][:top_k]

    recommend_df = pred_df.iloc[topk_idx].reset_index(drop=True)

    print('---------- Recommend ----------')
    for i, row in recommend_df.iterrows():
         print(f'{i + 1}: {row["Japanese name"]}:{row["English name"]}')

    print('---------- Rated ----------')
    user_df = user_df.merge(anime, left_on='anime_id', right_on='MAL_ID', how='inner')
    for i, row in user_df.sort_values('rating',ascending=False).iterrows():
         print(f'rating:{row["rating"]}: {row["Japanese name"]}:{row["English name"]}')

    return recommend_df

def predicttest():
    genre_names, merged_df, anime, rating = learningtorank()
    user_df = rating.copy().loc[rating['user_id'] == 12]
    user_df = make_user_feature(user_df)
    predict(user_df, 10, anime, rating)

app.vecterizer = pickle.load(open('resource/Bm25SerchByName.pkl', 'rb'))
@app.route('/SerachByName', methods=['POST'])
def FlaskSearhByName():
    genre_names, merged_df, anime, rating = getanimedata()
    response_object = {'status': 'success'}
    requests = request.get_json()
    query = json.dumps(requests)
    parsed_json = json.loads(query)
    query = parsed_json.get("Name")
    print(query)
    score = app.vecterizer.transform(query)
    rank = np.argsort(score)[::-1]
    response_object = anime.iloc[rank[:10]].to_json()
    return response_object

app.vecterizersyn = pickle.load(open('resource/Bm25SerchBySynopsis.pkl', 'rb'))
@app.route('/SerachBySysnopsis', methods=['POST'])
def FlaskSearhBySysnopsis():
    genre_names, merged_df, anime, rating = getanimedata()
    response_object = {'status': 'success'}
    requests = request.get_json()
    query = json.dumps(requests)
    parsed_json = json.loads(query)
    query = parsed_json.get("Synopsis")
    print(query)
    score = app.vecterizersyn.transform(query)
    rank = np.argsort(score)[::-1]
    response_object = anime.iloc[rank[:10]].to_json()
    return response_object

@app.route('/predictanime', methods=['POST'])
def predictanime():
    response_object = {'status': 'success'}
    genre_names, merged_df, anime, rating = learningtorank()
    data = request.get_json()
    print(data)
    array = json.dumps(data)
    python_obj = json.loads(array)
    df = pd.DataFrame.from_dict(python_obj)
    user_df = make_user_feature(df)
    pre = predict(user_df, 10, anime, rating)
    results = []
    print('---------- Recommend ----------')
    for i, row in pre.iterrows():
        results.append(f'{i + 1}: {row["Name"]}')

    response_object = jsonify(results)

    return response_object
if __name__ == '__main__':
    app.run(debug=True)