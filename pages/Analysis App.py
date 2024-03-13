import streamlit as st
st.set_page_config(page_title='Analytics App')
import pandas as pd
import pickle
import plotly.express as px

df = pd.read_csv("./Data Cleaning/smartphones_feature_selection_cleaned.csv")


# Plotting brands against prices
def brand_by_price(df):
    avg_price_by_brand = df.groupby(['brand_name'])
    avg_price_by_brand = avg_price_by_brand['price'].mean().sort_values(ascending=False).reset_index()
    return avg_price_by_brand
avg_price_by_brand = brand_by_price(df)
fig1 = px.bar(avg_price_by_brand, x='brand_name', y='price', title='Average Prices of Phones by Brand')
st.plotly_chart(fig1)


# Brand Vs Processor Speed
def brand_by_processor_speed(df):
    avg_specs_by_brand = df.groupby(['brand_name'])
    avg_specs_by_brand = avg_specs_by_brand['processor_speed'].mean().sort_values(ascending=False).reset_index()
    return avg_specs_by_brand
avg_specs_by_brand = brand_by_processor_speed(df)
fig2 = px.bar(avg_specs_by_brand, x='brand_name', y='processor_speed',title='Average Processor Speed by Brand')
st.plotly_chart(fig2)
