import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title='Insights Module')

st.title("Smartphones Insights")

with open(r'./df.pkl','rb') as file:
    df = pickle.load(file)

y_log = pd.read_csv(r"./y_log.csv")
coef_df = pd.read_csv(r"./coef_df.csv")
X_scaled = pd.read_csv(r"./X_scaled.csv")
X = pd.read_csv(r"./X_df.csv")

columnss = X.columns.tolist()
columnss.insert(0,'None')
# Allow users to select features for comparison
selected_feature = st.selectbox("Select features to compare",columnss)


if selected_feature == 'brand_name':
    brand_name = st.selectbox('Brand Name', options=df['brand_name'].str.capitalize().unique().tolist()).lower()
    unscaled_value = coef_df[coef_df['features'] == f'brand_name_{brand_name}']['coef'].values[0]
    scaled_value = unscaled_value * (np.expm1(y_log).std() / X_scaled['brand_name'].std())
    if scaled_value.values[0] > 0:
        st.success("Price Increases by {}".format(round(scaled_value.values[0])))
    else:
        st.error("Price Decreases by {}".format(round(scaled_value.values[0])))

elif selected_feature == 'processor_brand':
    processor_brand = st.selectbox("Processor Brand",df['processor_brand'].str.capitalize().unique()).lower()
    unscaled_value = coef_df[coef_df['features'] == f'processor_brand_{processor_brand}']['coef'].values[0]
    scaled_value = unscaled_value * (np.expm1(y_log).std() / X_scaled['processor_brand'].std())
    if scaled_value.values[0] > 0:
        st.success("Price Increases by {}".format(round(scaled_value.values[0])))
    else:
        st.error("Price Decreases by {}".format(round(scaled_value.values[0])))

elif selected_feature == 'resolution':
    resolution = st.selectbox("Resolution",['HD','HD+','FHD','FHD+','QHD','UHD'])
    unscaled_value = coef_df[coef_df['features'] == f'resolution_{resolution}']['coef'].values[0]
    scaled_value = unscaled_value * (np.expm1(y_log).std() / X_scaled['resolution'].std())
    if scaled_value.values[0] > 0:
        st.success("Price Increases by {}".format(round(scaled_value.values[0])))
    else:
        st.error("Price Decreases by {}".format(round(scaled_value.values[0])))

elif selected_feature == 'os':
    os = st.radio("Operating System",['iOS','Android','Other'],horizontal=True).lower()
    unscaled_value = coef_df[coef_df['features'] == f'os_{os}']['coef'].values[0]
    scaled_value = unscaled_value * (np.expm1(y_log).std() / X_scaled['os'].std())
    if scaled_value.values[0] > 0:
        st.success("Price Increases by {}".format(round(scaled_value.values[0])))
    else:
        st.error("Price Decreases by {}".format(round(scaled_value.values[0])))

elif selected_feature == 'has_5g':
    unscaled_value = coef_df[coef_df['features'] == 'has_5g']['coef'].values[0]
    scaled_value = unscaled_value * (np.expm1(y_log).std() / X['has_5g'].std())
    if scaled_value.values[0] > 0:
        st.success("Price Increases by {}".format(round(scaled_value.values[0])))
    else:
        st.error("Price Decreases by {}".format(round(scaled_value.values[0])))

elif selected_feature == 'has_nfc':
    unscaled_value = coef_df[coef_df['features'] == 'has_nfc']['coef'].values[0]
    scaled_value = unscaled_value * (np.expm1(y_log).std() / X['has_nfc'].std())
    if scaled_value.values[0] > 0:
        st.success("Price Increases by {}".format(round(scaled_value.values[0])))
    else:
        st.error("Price Decreases by {}".format(round(scaled_value.values[0])))

elif selected_feature == 'processor_speed':
    processor_speed = st.slider("Processor Speed", min_value=np.floor(df['processor_speed'].min()),max_value=np.ceil(df['processor_speed'].max()))
    unscaled_value = coef_df[coef_df['features'] == 'processor_speed']['coef'].values[0]
    scaled_value = unscaled_value * (np.expm1(y_log).std() / X['processor_speed'].std()) * processor_speed
    if scaled_value.values[0] > 0:
        st.success("Price Increases by {}".format(round(scaled_value.values[0])))
    else:
        st.error("Price Decreases by {}".format(round(scaled_value.values[0])))

elif selected_feature == 'ram_capacity':
    ram_capacity = st.select_slider("Ram Capacity",options=df['ram_capacity'].value_counts().index.sort_values().to_list())
    unscaled_value = coef_df[coef_df['features'] == 'ram_capacity']['coef'].values[0]
    scaled_value = unscaled_value * (np.expm1(y_log).std() / X['ram_capacity'].std()) * ram_capacity
    if scaled_value.values[0] > 0:
        st.success("Price Increases by {}".format(round(scaled_value.values[0])))
    else:
        st.error("Price Decreases by {}".format(round(scaled_value.values[0])))

elif selected_feature == 'internal_memory':
    internal_memory = st.select_slider("Internal Memory Capacity",options=df['internal_memory'].value_counts().index.sort_values().to_list())
    unscaled_value = coef_df[coef_df['features'] == 'internal_memory']['coef'].values[0]
    scaled_value = unscaled_value * (np.expm1(y_log).std() / X['internal_memory'].std()) * internal_memory
    if scaled_value.values[0] > 0:
        st.success("Price Increases by {}".format(round(scaled_value.values[0])))
    else:
        st.error("Price Decreases by {}".format(round(scaled_value.values[0])))

elif selected_feature == 'battery_capacity':
    battery_capacity = st.slider("Battery Capacity",min_value=1500,max_value=7200,step=10)
    unscaled_value = coef_df[coef_df['features'] == 'battery_capacity']['coef'].values[0]
    scaled_value = unscaled_value * (np.expm1(y_log).std() / X['battery_capacity'].std())*(battery_capacity)
    if scaled_value.values[0] > 0:
        st.success("Price Increases by {}".format(round(scaled_value.values[0])))
    else:
        st.error("Price Decreases by {}".format(round(scaled_value.values[0])))

elif selected_feature == 'fast_charging':
    fast_charging = st.select_slider("Fast Charging",options=df['fast_charging'].value_counts().index.sort_values().to_list())
    unscaled_value = coef_df[coef_df['features'] == 'fast_charging']['coef'].values[0]
    scaled_value = unscaled_value * (np.expm1(y_log).std() / X['fast_charging'].std())*(fast_charging)
    if scaled_value.values[0] > 0:
        st.success("Price Increases by {}".format(round(scaled_value.values[0])))
    else:
        st.error("Price Decreases by {}".format(round(scaled_value.values[0])))

elif selected_feature == 'num_rear_cameras':
    num_rear_cameras = st.radio("Number of Rear Cameras",[1,2,3,4],horizontal=True)
    unscaled_value = coef_df[coef_df['features'] == 'num_rear_cameras']['coef'].values[0]
    scaled_value = unscaled_value * (np.expm1(y_log).std() / X['num_rear_cameras'].std())*(num_rear_cameras)
    if scaled_value.values[0] > 0:
        st.success("Price Increases by {}".format(round(scaled_value.values[0])))
    else:
        st.error("Price Decreases by {}".format(round(scaled_value.values[0])))

elif selected_feature == 'num_rear_cameras':
    num_rear_cameras = st.radio("Number of Rear Cameras",[1,2,3,4],horizontal=True)
    unscaled_value = coef_df[coef_df['features'] == 'num_rear_cameras']['coef'].values[0]
    scaled_value = unscaled_value * (np.expm1(y_log).std() / X['num_rear_cameras'].std())*(num_rear_cameras)
    if scaled_value.values[0] > 0:
        st.success("Price Increases by {}".format(round(scaled_value.values[0])))
    else:
        st.error("Price Decreases by {}".format(round(scaled_value.values[0])))

elif selected_feature == 'screen_size':
    screen_size = st.slider("Screen Size",min_value=4.0,max_value=8.5)
    unscaled_value = coef_df[coef_df['features'] == 'screen_size']['coef'].values[0]
    scaled_value = unscaled_value * (np.expm1(y_log).std() / X['screen_size'].std())*(screen_size)
    if scaled_value.values[0] > 0:
        st.success("Price Increases by {}".format(round(scaled_value.values[0])))
    else:
        st.error("Price Decreases by {}".format(round(scaled_value.values[0])))

elif selected_feature == 'refresh_rate':
    refresh_rate = st.selectbox("Refresh Rate",[60,90,120,144,165])
    unscaled_value = coef_df[coef_df['features'] == 'refresh_rate']['coef'].values[0]
    scaled_value = unscaled_value * (np.expm1(y_log).std() / X['refresh_rate'].std())*(refresh_rate)
    if scaled_value.values[0] > 0:
        st.success("Price Increases by {}".format(round(scaled_value.values[0])))
    else:
        st.error("Price Decreases by {}".format(round(scaled_value.values[0])))

elif selected_feature == 'extended_upto':
    extended_memory = st.select_slider("Extended Memory Capacity Upto",options=[0, 32, 64, 128, 256, 400, 512, 1024, 2048])
    unscaled_value = coef_df[coef_df['features'] == 'extended_upto']['coef'].values[0]
    scaled_value = unscaled_value * (np.expm1(y_log).std() / X['extended_upto'].std())*(extended_memory)
    if scaled_value.values[0] > 0:
        st.success("Price Increases by {}".format(round(scaled_value.values[0])))
    else:
        st.error("Price Decreases by {}".format(round(scaled_value.values[0])))

elif selected_feature == 'primary_camera_rear':
    primary_camera_rear = st.select_slider("Rear Primary Camera (MP)",options=df['primary_camera_rear'].value_counts().index.sort_values().to_list())
    unscaled_value = coef_df[coef_df['features'] == 'primary_camera_rear']['coef'].values[0]
    scaled_value = unscaled_value * (np.expm1(y_log).std() / X['primary_camera_rear'].std()) * (primary_camera_rear)
    if scaled_value.values[0] > 0:
        st.success("Price Increases by {}".format(round(scaled_value.values[0])))
    else:
        st.error("Price Decreases by {}".format(round(scaled_value.values[0])))

elif selected_feature == 'primary_camera_front':
    primary_camera_front = st.select_slider("Front Primary Camera (MP)",options=df['primary_camera_front'].value_counts().index.sort_values().to_list())
    unscaled_value = coef_df[coef_df['features'] == 'primary_camera_front']['coef'].values[0]
    scaled_value = unscaled_value * (np.expm1(y_log).std() / X['primary_camera_front'].std()) * (primary_camera_front)
    if scaled_value.values[0] > 0:
        st.success("Price Increases by {}".format(round(scaled_value.values[0])))
    else:
        st.error("Price Decreases by {}".format(round(scaled_value.values[0])))
