import streamlit as st
import json
from Classifier import KNearestNeighbour
from operator import itemgetter

# Load data and movies list from corresponding JSON files
with open(r'data.json', 'r+', encoding='utf-8') as f:
    data = json.load(f)
with open(r'titles.json', 'r+', encoding='utf-8') as f:
    movie_titles = json.load(f)


def knn(test_point, k):  # test_pont: genre based or movie based

    # create dummy target variable for the knn classifier
    target = [0 for item in movie_titles]

    # instantiate object for the classifier
    model = KNearestNeighbour(data, target, test_point, k= k)

    # run the algorithm
    model.fit()

    # distancs to the most distant movie
    max_dist = sorted(model.distances, key= itemgetter(0))[-1]

    # print list of 10 recommemdation < change value of k for a diffrent number>
    table = list()

    for i in model.indices:
        # return back the movie title and imdb link
        table.append([movie_titles[i][0], movie_titles[i][2]])
        # adding 0th index and 2nd index data
    return table

if __name__  == '__main__':
    genres = ['Action','Adventure','Animation','Biography','Comedy','Crime','Documentary', 'Drama', 'Family',
              'Fantasy','Film-Noir', 'Game-Show', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News',
              'Reality-TV','Romance', 'Sci-Fi','Short', 'Sport', 'Thriller', 'War', 'Western']

    movies = [title[0] for title in movie_titles]
    st.header(" The Movie Recommendation System ")

    apps = ["--Select--", "Movie Based", "Genres Based"]
    # index(select)==0, indx(Movie based)==1, index(Genres based)==2
    app_options = st.selectbox("Select Application: ", apps)

    if app_options == 'Movie based':
        movie_select = st.selectbox('Select movie:', ['--Select--'] + movies)
        if movie_select == '--Select--':
            st.write('Select a movie')
        else:
            n = st.number_input('Number of movies:', min_value=5, max_value=20, step=1)
            genres = data[movies.index(movie_select)]
            test_point = genres
            table = knn(test_point, n) # test_point , n = nearest neighbour
            for movie, link in table:
                # Displays movie title with link to imdb
                st.markdown(f"[{movie}]({link})")


    elif app_options == apps[2]:  # apps[2] --> index 2--> means genres
        options = st.multiselect('Select genres:', genres)
        if options:
            imdb_score = st.slider('IMDb score:', 1, 10, 8)
            # imdb score --> min 1, Max 10, default 8
            n = st.number_input('Number of movies:', min_value=5, max_value=20, step=1)
            test_point = [1 if genre in options else 0 for genre in genres]
            test_point.append(imdb_score)
            table = knn(test_point, n)
            for movie, link in table:
                # Displays movie title with link to imdb
                st.markdown(f"[{movie}]({link})")

        else:
            st.write("This is a Movie Recommender application.  "
                     "You can select the genres and change the IMDb score.")

    else:
        st.write('Select option')








