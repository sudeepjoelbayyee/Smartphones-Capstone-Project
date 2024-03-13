import streamlit as st
st.set_page_config(page_title='Price Prediction App')
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import time
from xgboost import XGBRegressor

with open(r'./df.pkl','rb') as file:
    df = pickle.load(file)

df1 = pd.read_csv(r'./Data Cleaning/smartphones_feature_selection_cleaned_v3_linksphotos.csv')

with open(r'./pipeline.pkl','rb') as file:
    pipeline = pickle.load(file)

st.subheader("Enter your inputs")

# Brand Name
brand_name = st.selectbox('Brand Name',df['brand_name'].str.capitalize().unique()).lower()

# Has 5G
has_5g = st.radio("Has 5G?",['Yes','No'],horizontal=True)
if has_5g == 'Yes':
    has_5g = 1
else:
    has_5g = 0

# Has NFC
has_nfc = st.radio("Has NFC? (Near Field Communication)",['Yes','No'],horizontal=True)
if has_nfc == 'Yes':
    has_nfc = 1
else:
    has_nfc = 0

# Processor Brand
processor_brand = st.selectbox("Processor Brand",df['processor_brand'].str.capitalize().unique()).lower()

# Processor Speed
processor_speed = st.slider("Processor Speed",min_value=np.floor(df['processor_speed'].min()),max_value=np.ceil(df['processor_speed'].max()))

# Ram capacity
ram_capacity = st.select_slider("Ram Capacity",options=df['ram_capacity'].value_counts().index.sort_values().to_list())

# Internal Memory Capacity
internal_memory = st.select_slider("Internal Memory Capacity",options=df['internal_memory'].value_counts().index.sort_values().to_list())

# Battery Capacity
battery_capacity = st.slider("Battery Capacity",min_value=1500,max_value=7200,step=10)

# Fast Charging
fast_charging = st.select_slider("Fast Charging",options=df['fast_charging'].value_counts().index.sort_values().to_list())

# Number of rear cameras
num_rear_cameras = st.radio("Number of Rear Cameras",[1,2,3,4],horizontal=True)

# Screen Size
screen_size = st.slider("Screen Size",min_value=4.0,max_value=8.5)

# Resolution
resolution = st.selectbox("Resolution",['HD','HD+','FHD','FHD+','QHD','UHD'])

# Refresh Rate
refresh_rate = st.selectbox("Refresh Rate",[60,90,120,144,165])

# Operating System
os = st.radio("Operating System",['iOS','Android','Other'],horizontal=True).lower()

# Extended Memory Capacity Upto
extended_memory = st.select_slider("Extended Memory Capacity Upto",options=[0, 32, 64, 128, 256, 400, 512, 1024, 2048])

# Primary Camera Rear
primary_camera_rear = st.select_slider("Rear Primary Camera (MP)",options=df['primary_camera_rear'].value_counts().index.sort_values().to_list())

# Primary Camera Front
primary_camera_front = st.select_slider("Front Primary Camera (MP)",options=df['primary_camera_front'].value_counts().index.sort_values().to_list())


if st.button("Predict"):
    # Form a dataframe

    data = [[brand_name, has_5g, has_nfc, processor_brand, processor_speed, ram_capacity, internal_memory, battery_capacity, fast_charging, num_rear_cameras, screen_size, resolution, refresh_rate, os, extended_memory, primary_camera_rear, primary_camera_front]]
    columns = ['brand_name', 'has_5g', 'has_nfc', 'processor_brand',
               'processor_speed', 'ram_capacity', 'internal_memory',
               'battery_capacity', 'fast_charging', 'num_rear_cameras', 'screen_size',
               'resolution', 'refresh_rate', 'os', 'extended_upto',
               'primary_camera_rear', 'primary_camera_front']

    one_df = pd.DataFrame(data, columns=columns)

    # Predict
    base_price = np.expm1(pipeline.predict(one_df))[0]
    low = base_price - 1200
    high = base_price + 2200

    # Display
    st.success("The Price of Smartphone is in Between {} and {}".format(round(low),round(high)))

    time.sleep(0.5)
    loading_placeholder = st.empty()
    loading_placeholder.subheader("Recommending More...")

    ## Recommender System

    user_input = one_df.iloc[0,:].values.tolist()

    # Tokenize the dataset and user input
    def tokenize_data(data):
        tokenized_data = []
        for item in data:
            tokens = [str(element) for element in item]
            tokenized_data.append(tokens)
        return tokenized_data


    tokenized_dataset = tokenize_data(df.values)
    tokenized_user_input = [str(element) for element in user_input]

    # Train Word2Vec model
    model = Word2Vec(sentences=tokenized_dataset, vector_size=1000, window=5, min_count=1, workers=4)

    def get_embeddings(data, model):
        embeddings = []
        for item in data:
            item_embeddings = []
            for token in item:
                if token in model.wv:
                    item_embeddings.append(model.wv[token])
            embeddings.append(np.mean(item_embeddings, axis=0))
        return embeddings

    user_input_embeddings = np.array(get_embeddings([tokenized_user_input], model)[0])
    dataset_embeddings = np.array(get_embeddings(tokenized_dataset, model))


    def calculate_similarity(user_input, dataset):
        similarities = []
        user_vector = user_input.reshape(1, -1)
        for idx, item in enumerate(dataset):
            item_vector = item.reshape(1, -1)
            similarity = cosine_similarity(user_vector, item_vector)[0][0]
            similarities.append((idx, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities


    # Get top N recommendations
    def get_recommendations(user_input_embeddings, dataset_embeddings, dataset, n=5):
        similarities = calculate_similarity(user_input_embeddings, dataset_embeddings)
        recommendations = similarities[:n]
        recommended_phones = [(dataset[idx], similarity) for idx, similarity in recommendations]
        return recommended_phones


    # Convert DataFrame to a list of lists
    dataset_list = df.values.tolist()

    # Example of getting top 5 recommendations
    top_recommendations = get_recommendations(user_input_embeddings, dataset_embeddings, dataset_list, n=5)

    recommendation_indices = []
    for j in range(5):
        for i in range(len(df.values)):
            if df.values[i].tolist() == top_recommendations[j][0]:
                recommendation_indices.append(df.index[i])

    loading_placeholder.subheader("")
    st.subheader("Here are More SmartPhone Recommendations for You!")
    X = df1.iloc[recommendation_indices].sort_values('price')

    col1, col2, col3, col4, col5 = st.columns(5)

    # Define URLs for each image
    image_urls = X['links'].tolist()

    # Ensure all URLs are absolute
    base_url = "https://"  # Replace this with the base URL of your source pages
    absolute_urls = [base_url + url if not url.startswith('http') else url for url in image_urls]

    # Display clickable images
    for i, col in enumerate([col1, col2, col3, col4, col5]):
        with col:
            # Wrap the image inside an anchor tag
            st.markdown(f'<a href="{absolute_urls[i]}" target="_blank"><img src="{X["images"].tolist()[i]}" width="120"><br>{X["model"].tolist()[i]}</a>', unsafe_allow_html=True)
