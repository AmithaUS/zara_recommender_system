import streamlit as st
import pickle
import pandas as pd
from src.recommender import recommend

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Zara AI Recommender",
    page_icon="🛍",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
train_df = pickle.load(open("models/train_df.pkl", "rb"))

if not isinstance(train_df, pd.DataFrame):
    st.error("train_df.pkl is not a valid DataFrame.")
    st.stop()

# ---------------- AUTO COLUMN DETECTION ----------------
def detect_column(possible_names):
    for col in possible_names:
        if col in train_df.columns:
            return col
    return None

name_col = detect_column(["name", "product_name", "title"])
price_col = detect_column(["price", "Price"])
url_col = detect_column(["url", "link", "product_url"])

if name_col is None:
    st.error("No product name column found in dataset.")
    st.stop()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.product-card {
    background-color: #111827;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
    transition: transform 0.2s ease-in-out;
    margin-bottom: 25px;
    min-height: 180px;
}
.product-card:hover {
    transform: translateY(-5px);
}
.price {
    color: #10B981;
    font-weight: bold;
    font-size: 18px;
    margin-top: 10px;
}
.similarity {
    background-color: #1F2937;
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 12px;
    display: inline-block;
    margin-top: 10px;
}
.view-link {
    text-decoration: none;
    font-weight: bold;
    display: block;
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("🔎 Filter Options")

search_query = st.sidebar.text_input("Search Product")

if search_query:
    filtered_df = train_df[
        train_df[name_col].str.contains(search_query, case=False, na=False)
    ]
else:
    filtered_df = train_df

# ---------------- MAIN TITLE ----------------
st.title("🛍 Zara Fashion Product Recommendation System")
st.write("Hybrid ML Recommender (TF-IDF + Price Similarity)")

selected_product = st.selectbox(
    "Select a Product",
    filtered_df[name_col].values
)

# ---------------- RECOMMEND BUTTON ----------------
if st.button("✨ Get Recommendations"):

    try:
        results = recommend(selected_product)
    except Exception as e:
        st.error(f"Recommendation Error: {e}")
        st.stop()

    st.markdown("---")
    st.subheader("🔥 Recommended Products")

    cols = st.columns(3)

    for i, item in enumerate(results):
        with cols[i % 3]:
            st.markdown('<div class="product-card">', unsafe_allow_html=True)

            # PRODUCT NAME
            st.markdown(f"### {item.get(name_col, 'Unknown Product')}")

            # PRICE
            if price_col:
                st.markdown(
                    f"<div class='price'>💰 ${item.get(price_col, 'N/A')}</div>",
                    unsafe_allow_html=True
                )

            # SIMILARITY SCORE
            if "score" in item:
                st.markdown(
                    f"<div class='similarity'>📊 Similarity: {round(item['score'], 2)}</div>",
                    unsafe_allow_html=True
                )

            # VIEW PRODUCT LINK
            if url_col and url_col in item and item[url_col]:
                st.markdown(
                    f"<a class='view-link' href='{item[url_col]}' target='_blank'>🔗 View Product</a>",
                    unsafe_allow_html=True
                )

            st.markdown('</div>', unsafe_allow_html=True)

