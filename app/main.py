import sys
import joblib
import uvicorn
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse



# Inicia API
app = FastAPI()

# carrega a classe de recomendação
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


# Carrega modelo
# sys.path.append('src') # procura módulos de python na pasta src
recommender = joblib.load('models/recommender_system.pkl')

# Cria página inicial
@app.get('/', response_class=HTMLResponse)
def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Steam Game Recommender</title>
    </head>
    <body>
        <h1>Welcome to the Steam Game Recommender app!</h1>
        <ul>
            <li><a href="/list_games">All Games</a></li>
        </ul>
        <form action="/search_games" method="get">
            <input type="text" name="pattern" placeholder="Enter game name or part of the name">
            <button type="submit">Search</button>
        </form>
        <br>
        <h2>Search Results</h2>
        <ul>
            </ul>
    </body>
    </html>
    """
    return HTMLResponse(html_content, status_code=200)

# Lista jogos
@app.get('/list_games', response_class=HTMLResponse)
async def list_games():
    # Retrieve the list of games from the recommender system
    games = recommender.list_games()

    # Create an empty string to store the HTML content
    html_content = """<html>
        <head>
            <title>Steam Game List:</title>
        </head>
        <body>
            <h1>Steam Game List</h1>
            <ul>
                """

    # Iterate through each game in the list
    for game in games:
        # Encode the recommended game name for URL safety
        game_encoded = urllib.parse.quote(game)

        # Generate a URL for the recommendation page using the game name
        recommendation_url = f"/recommend?game={game_encoded}"

        # Create an HTML link element with the game name and recommendation URL
        game_link = f"<li><a href='{recommendation_url}'>{game}</a></li>"

        # Append the link to the HTML content string
        html_content += game_link


    # Close the unordered list tag
    html_content += """
                </ul>
            </body>
        </html>
    """

    # Return the HTML content as a string response
    return html_content


# Procura jogos por substring
@app.get('/search_games')
def search_games(pattern: str):
    pattern = pattern.lower()
    games = pd.Series(recommender.list_games())
    games_matched = games[games.str.lower().str.contains(pattern)]
        # Criar o conteúdo HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pesquisa de jogos</title>
    </head>
    <body>
        <h1>"Resultados da pesquisa: "</h1>
        <ul>
    """

    # Iterar pela pesquisa
    for game in games_matched:
        # Encode the recommended game name for URL safety
        game_encoded = urllib.parse.quote(game)


        # Gerar URL para a página de recomendações do jogo recomendado
        recommendation_url = f"/recommend?game={game_encoded}"

        # Criar um elemento de link HTML com o nome do jogo e a URL
        game_link = f"<li><a href='{recommendation_url}'>{game}</a></li>"

        # Adicionar o link ao conteúdo HTML
        html_content += game_link + "\n"

    html_content += """
        </ul>
    </body>
    </html>
    """

    return HTMLResponse(html_content, status_code=200, media_type="text/html")

# Recomenda jogo
@app.get('/recommend')
async def recommend(request: Request, game: str, max_recommendations: int = 10):
    # Obter as recomendações para o jogo selecionado
    recommendations = recommender.recommend_items(game, max_recommendations)

    # Filtrar o dicionário para obter apenas os nomes dos jogos
    game_names = recommendations.keys().tolist()

    # Criar o conteúdo HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Recomendações de Jogos</title>
    </head>
    <body>
        <h1>Recomendações para "{game}"</h1>
        <ul>
    """

    # Iterar pelas recomendações
    for recommended_game in game_names:

        # Encode the recommended game name for URL safety
        recommended_game_encoded = urllib.parse.quote(recommended_game)

        # Gerar URL para a página de recomendações do jogo recomendado
        recommendation_url = f"/recommend?game={recommended_game_encoded}"

        # Criar um elemento de link HTML com o nome do jogo e a URL
        game_link = f"<li><a href='{recommendation_url}'>{recommended_game}</a></li>"

        # Adicionar o link ao conteúdo HTML
        html_content += game_link + "\n"

    html_content += """
        </ul>
    </body>
    </html>
    """

    # Retornar o conteúdo HTML como uma resposta HTML
    return HTMLResponse(html_content, status_code=200, media_type="text/html")


# Executa API
if __name__ == '__main__':
    uvicorn.run(app)