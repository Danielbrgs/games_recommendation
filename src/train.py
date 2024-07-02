import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def get_steam_data(file_path:str) -> pd.DataFrame:
    try:
        column_names = ['user_id', 'item_id', 'behaviour', 'hours']
        df = pd.read_csv(file_path, header=None, names=column_names, usecols=range(4))
        return df
    except:
        column_names = ['user_id', 'item_id', 'behaviour', 'hours']
        df = pd.read_csv('../data/steam-200k.csv', header=None, names=column_names, usecols=range(4))
        return df
def get_ratings_df(df:pd.DataFrame) -> pd.DataFrame:
    df_play = df.query('behaviour == "play"')
    df_user_total_hours = df_play.groupby(['user_id'])['hours'].sum().reset_index()
    df_user_total_hours.rename(columns={'hours': 'total'}, inplace=True)
    df_play = df_play.merge(df_user_total_hours, on='user_id',how='left')
    df_play = df_play.drop(columns=['behaviour'])
    df_play['rating'] = df_play['hours']/df_play['total']
    df_play.drop(columns=['hours', 'total'], inplace=True)
    lista_duplicatas = []
    for i in df_play['user_id'].unique():  # esta função procura duplicatas de jogos para cada usuário
        uid = "user_id == " + str(i)
        ps = df_play.query(uid).duplicated('item_id')
        has_true_values = ps.any()
        if has_true_values:
            lista_duplicatas.append(i)
    for i in lista_duplicatas:
        user_data = df_play[df_play['user_id'] == i]
        user_data_no_duplicates = user_data.drop_duplicates(subset='item_id', keep='first')
        df_play.loc[user_data.index, :] = user_data_no_duplicates
    df_play = df_play.dropna()
    ratings_matrix = df_play.pivot(index='item_id', columns='user_id', values='rating').fillna(0)
    return ratings_matrix

def get_similarity(ratings_matrix:pd.DataFrame) -> pd.DataFrame:
    item_similarity = cosine_similarity(ratings_matrix)
    item_similarity_df = pd.DataFrame(item_similarity, index=ratings_matrix.index, columns=ratings_matrix.index)
    return item_similarity_df

class RecommenderSystem:
    def __init__(self):
        self.similarity_matrix = joblib.load('data/item_similarity_df.pkl')
        self.game_list = joblib.load('data/game_list.pkl')
        
    def recommend_items(self, item_id, num_recommendations=10):
        try:
            similar_items = self.similarity_matrix[item_id].sort_values(ascending=False)
            similar_items = similar_items.drop(item_id)
            return similar_items.head(num_recommendations)
        except KeyError:
            return f'Item {item_id} not found in the dataset.'
    
    def list_games(self):
        return self.game_list
    
# salvando a matriz de similiaridade
file_path = 'data/steam-200k.csv'
item_similarity_df = get_similarity(get_ratings_df(get_steam_data(file_path)))
joblib.dump(item_similarity_df, 'data/item_similarity_df.pkl')

# obtendo lista de jogos
list = get_steam_data(file_path)['item_id'].unique().tolist()
joblib.dump(list, 'data/game_list.pkl')

# Crie uma instância da classe
recommender = RecommenderSystem()

# Salvando o modelo
joblib.dump(recommender, 'models/recommender_system.pkl')