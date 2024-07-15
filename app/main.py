import sys
import joblib
import uvicorn
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse

# Create a FastAPI web application instance
app = FastAPI()

# Class to encapsulate recommender system logic
class RecommenderSystem:
    def __init__(self):
        # Load the pre-trained similarity matrix and game list
        self.similarity_matrix = joblib.load('data/item_similarity_df.pkl')
        self.game_list = joblib.load('data/game_list.pkl')

    def recommend_items(self, item_id, num_recommendations=10):
        try:
            # Find similar items based on cosine similarity
            similar_items = self.similarity_matrix[item_id].sort_values(ascending=False)
            # Remove the original item from the recommendations
            similar_items = similar_items.drop(item_id)
            # Return the top N similar items
            return similar_items.head(num_recommendations)
        except KeyError:
            # Handle case where item ID is not found
            return f'Item {item_id} not found in the dataset.'

    def list_games(self):
        # Return the list of all games
        return self.game_list

# Load the pre-trained recommender system model
recommender = joblib.load('models/recommender_system.pkl')

# Define the home page route
@app.get('/')
def home():
    # Generate HTML content for the home page
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

# Define the route to list all games
@app.get('/list_games')
async def list_games():
    # Get the list of games from the recommender system
    games = recommender.list_games()

    # Create the HTML content for the game list
    html_content = """<html>
        <head>
            <title>Steam Game List:</title>
        </head>
        <body>
            <h1>Steam Game List</h1>
            <ul>
            """

    # Create HTML links for each game with recommendations
    for game in games:
        game_encoded = urllib.parse.quote(game)
        recommendation_url = f"/recommend?game={game_encoded}"
        game_link = f"<li><a href='{recommendation_url}'>{game}</a></li>"
        html_content += game_link

    html_content += """
            </ul>
        </body>
    </html>
    """

    return HTMLResponse(html_content, status_code=200)

# Define the route to search for games
@app.get('/search_games')
def search_games(pattern: str):
    # Search for games matching the search pattern
    pattern = pattern.lower()
    games = pd.Series(recommender.list_games())
    games_matched = games[games.str.lower().str.contains(pattern)]

    # Create the HTML content for the search results
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

    # Create HTML links for each matching game with recommendations
    for game in games_matched:
        game_encoded = urllib.parse.quote(game)
        recommendation_url = f"/recommend?game={game_encoded}"
        game_link = f"<li><a href='{recommendation_url}'>{game}</a></li>"
        html_content += game_link + "\n"

    html_content += """
        </ul>
    </body>
    </html>
    """

    return HTMLResponse(html_content, status_code=200, media_type="text/html")

# Define the route to recommend games
@app.get('/recommend')
async def recommend(request: Request, game: str, max_recommendations: int = 10):
    # Get recommendations for the selected game
    recommendations = recommender.recommend_items(game, max_recommendations)

    # Create the HTML content for the recommendations
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

    # Create HTML links for each recommended game
    for recommended_game in recommendations.keys():
        recommended_game_encoded = urllib.parse.quote(recommended_game)
        recommendation_url = f"/recommend?game={recommended_game_encoded}"
        game_link = f"<li><a href='{recommendation_url}'>{recommended_game}</a></li>"
        html_content += game_link + "\n"

    html_content += """
        </ul>
    </body>
    </html>
    """

    return HTMLResponse(html_content, status_code=200, media_type="text/html")

# Run the FastAPI application
if __name__ == '__main__':
    uvicorn.run(app, port=8080)
