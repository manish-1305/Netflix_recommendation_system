import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st




df = pd.read_csv("netflix_titles.csv")
df.drop(["show_id","director","cast","country","date_added","release_year","rating","duration"],axis=1,inplace=True)
df['title'] = df['title'].str.lower()

vectoriser =TfidfVectorizer()
matrix = vectoriser.fit_transform(df['description'])
cosine_similarities = linear_kernel(matrix,matrix)
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





def main():
    st.title("Netflix Recommendation System")
    st.subheader("Content Based filtering")
    ask = st.text_input("Enter the movie name:")
    if st.button('Get Result'):
        
        recommendations_df = get_recommendations(ask)
        st.write(recommendations_df)
    
    
    
    
if __name__ == '__main__':
    main()    
    




