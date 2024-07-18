from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load and preprocess the data
df = pd.read_csv("netflix_titles.csv")
df.drop(["show_id", "director", "cast", "country", "date_added", "release_year", "rating", "duration"], axis=1, inplace=True)
df['title'] = df['title'].str.lower()

vectoriser = TfidfVectorizer()
matrix = vectoriser.fit_transform(df['description'])
cosine_similarities = linear_kernel(matrix, matrix)
movie_title = df['title']
indices = pd.Series(df.index, index=df['title'])

def get_recommendations(title, cosine_sim=cosine_similarities):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get the top 10 items
    item_indices = [i[0] for i in sim_scores]
    rec_movie = df['title'].iloc[item_indices]
    return df.iloc[item_indices][['title', 'type', 'listed_in']]

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        movie_name = request.form['movie_name'].lower()
        recommendations_df = get_recommendations(movie_name)
        return render_template('index.html', recommendations=recommendations_df.to_dict('records'), movie_name=movie_name)
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
