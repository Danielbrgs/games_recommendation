# Steam Game Recommendation System
### Overview
This project implements a collaborative filtering based item recommendation system using the Steam dataset and constructs an API to interact with the model. The goal is to allow users to input a game name and receive the 10 most similar games as recommendations.

### Key Points:

* Collaborative Filtering: Leverages user-item interactions (time spent playing) to identify similar games.
* Item Similarity Matrix: Generates a similarity matrix based on user interactions.
* Recommendation Engine: Utilizes the similarity matrix to recommend similar games.
* API Interface: Provides an API endpoint to interact with the recommendation system.
### Dataset and Preprocessing
The Steam dataset is used to train the recommendation model. The dataset includes user-game interactions, such as the time spent playing each game. This data is preprocessed to extract relevant information and calculate implicit ratings for each game.

### Model Training and Evaluation
The recommendation model is trained using the preprocessed data and the item similarity matrix. The model is evaluated based on its ability to recommend relevant and similar games.

### Model Creation and PKL Files

The model creation process is outlined in the test.ipynb notebook, located in the project's root directory. This notebook utilizes the preprocessed data to generate the item similarity matrix and trains a collaborative filtering model using the Surprise library. The trained model is then saved as a pickle file (.pkl) using the joblib library.

### Notebooks Used:

* test.ipynb (Root Directory): Creates the item similarity matrix, trains the recommendation model, and saves the model as a .pkl file.
* notebooks/reco_note.ipynb: Provides data exploration and initial testing of the recommendation system.
### Implementation and Deployment
The recommendation system is implemented using Python and deployed as an API using FastAPI. The API provides an endpoint for users to submit game names and receive recommendations. The .pkl file containing the trained model is loaded into the API to generate recommendations.

### Running the App Locally
1. Activate Conda Environment:

```bash
conda activate steam-rec
```

2. Install Dependencies:

```bash
pip install -r requirements.txt
```
3. Run the App:
```bash
python app\main.py
```
4. Access the App:

Open a web browser and navigate to http://localhost:8000 to access the app.

### Additional Notes:

Ensure you have Python 3.11 installed. If using a different Python version, you may need to adjust the python= parameter in the Conda environment creation step.
If you encounter any errors, check the terminal output and project logs for more information.
### Troubleshooting:

If you face issues starting the server, ensure you are running the command from the correct project directory and that the main.py file is present in the app folder.
If you encounter permission errors, try running the commands with elevated privileges (using sudo or running from an administrator account).
### App Usage
The app provides two main functionalities:

* List of Games: Users can access a list of all available games.
* Game Search: Users can search for a specific game and receive 10 similar game recommendations.
Each recommendation list displays 10 similar games, and each recommended game can be clicked to view its own set of 10 recommendations.

### Recommendation System Explanation
The recommendation system utilizes collaborative filtering to identify similar games based on user interactions. It calculates implicit ratings for each game based on the time spent playing and generates an item similarity matrix. When a user inputs a game name, the system retrieves the corresponding game's similarity vector from the matrix and recommends the 10 most similar games.

**Para saber mais:**

* [Conecte-se comigo no LinkedIn](https://www.linkedin.com/in/daniel-braga-reis-725aa012a/)
* [Explore meus projetos no GitHub](https://github.com/Danielbrgs?tab=repositories)

