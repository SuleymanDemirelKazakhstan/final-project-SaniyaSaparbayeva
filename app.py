import  streamlit as st
import pickle
import pandas as pd
from PIL import Image
st.title('Movie Recommendation')


def fetch_poster(movie_id):
    #url = "https://api.themoviedb.org/3/movie/popular?api_key='549ce4ee8cfb0a14d2f676b9bd4562e2'&language=en-US".format(movie_id)
    #data = requests.get(url)
    #data = data.json()
    poster_path = movies[movies['id'] == movie_id]['poster_path'].values[0]
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_movie_posters = []

    for i in movies_list:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movie_posters.append(fetch_poster(movie_id))

    return recommended_movies,recommended_movie_posters

movies_dict = pickle.load(open('movies_1.pkl','rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity_1.pkl','rb'))


selected_movie_name = st.selectbox(
    "Type or select a movie from the dropdown",
    movies['title'].values
)
if st.button('Show Recommendation'):
    names,posters = recommend(selected_movie_name)

    #display with the columns
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])
        
st.markdown('##')
st.header('Data')
st.write("The data was collected from [TMDB web-site](https://www.themoviedb.org/movie).")
st.write("Dataframe columns: ")
st.markdown("""
            - id - movie id
            - title - movie title 
            - overview - movie overview
            - cast - actors
            - crew - director
            - keywords - movie specific words
            - genre_name - movie genre
            """)
            
st.markdown('**First 5 rows**:')
df = pd.read_csv('movies_10k.csv',index_col=0)
st.write(df.head())

st.write(str(df.shape) +" Dataframe has 10000 rows and 8 columns")


st.header('Preprocessing')
st.write("The function convert remove curly brackets from columns keywods and crew. And return to this columns only name values")
code = ''' def convert(object):
    list=[]
    for i in ast.literal_eval(object):
        list.append(i['name'])
    return list '''
st.code(code, language='python')

st.write("The function convert3 remove curly brackets from column cast. And return to this column only 3 main actors")
code = ''' def convert3(obj):
    L = []
    c=0
    for i in ast.literal_eval(obj):
        if(c!=3):
            L.append(i['name'])
            c+=1
        else:
            break
    return L
movies['cast'] = movies['cast'].apply(convert3) '''
st.code(code, language='python')


st.write("Merge columns into one column that will be used in our model for recommendation")
code = ''' movies['tags'] = movies['overview']+ movies['genre_name'] + movies['keywords'] + movies['cast'] + movies['crew'] '''
st.code(code, language='python')    


st.markdown('**Final dataframe**:')
df2 = pd.read_csv('final_df.csv',index_col=0)
st.write(df2.head())
st.write(str(df2.shape) +" Dataframe has 9960 rows and 4 columns")

st.markdown('##')

st.header('Create recommendation system')
st.write("Transform words in tags column to vectors")
code = ''' v = CountVectorizer(max_features =5000, stop_words='english') 
vectors=v.fit_transform(new_df['tags']).toarray()'''
st.code(code, language='python')

image = Image.open('vectors.jpg')
st.image(image, caption='array of vectors has 9960 rows and 5000 columns')

st.write("Using the cosine_similarity create weights for vectors")
code = '''from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(vectors)'''
st.code(code, language='python')

image = Image.open('cosine_vectors.jpg')
st.image(image)

st.write("The function that makes recommendation")
code = ''' def recommend(movie):
    #find the index of the movies
    movie_index = new_df[new_df['title']==movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    #to fetch movies from indeces
    for i in movies_list:
        print(new_df.iloc[i[0]].title) '''
st.code(code, language='python')