#  Zara Product Recommendation System

A content-based product recommendation engine built on Zara's fashion catalogue, designed to suggest visually and semantically similar products to users based on item features such as category, color, fabric, and description.

---

##  Overview

This project develops a product recommendation system for Zara's e-commerce catalogue using content-based filtering techniques. By analysing product metadata and textual descriptions, the system identifies and recommends similar products — mimicking the "You may also like" feature found in modern fashion retail platforms.

The project demonstrates the application of natural language processing (NLP), feature engineering, and similarity computation on a real-world retail dataset.

---

##  Objectives

- Build a content-based filtering recommender system for fashion products
- Extract meaningful features from product names, descriptions, categories, and attributes
- Compute item-to-item similarity using TF-IDF vectorization and cosine similarity
- Return the top-N most similar products for any given input product

---

##  Dataset

- **Source:** Zara product catalogue (scraped / publicly available dataset)
- **Features used:**
  - Product name
  - Category / subcategory
  - Color and fabric details
  - Product description
  - Price

---

##  Tech Stack

| Area | Tools / Libraries |
|---|---|
| Language | Python 3 |
| Data Handling | Pandas, NumPy |
| NLP & Vectorization | Scikit-learn (TF-IDF) |
| Similarity Metric | Cosine Similarity |
| Visualization | Matplotlib, Seaborn |
| Notebook Environment | Jupyter Notebook / Google Colab |

---

##  Methodology

### 1. Data Preprocessing
- Handled missing values in product descriptions and categorical fields
- Standardized text (lowercasing, punctuation removal, stopword removal)
- Combined relevant text features into a unified "tag" or "soup" column per product

### 2. Feature Engineering
- Applied **TF-IDF Vectorization** on the combined text features to convert product descriptions into numerical vectors
- Each product is represented as a high-dimensional vector reflecting the importance of each term relative to the catalogue

### 3. Similarity Computation
- Computed **cosine similarity** between all product vectors to produce an item-to-item similarity matrix
- Higher cosine similarity scores indicate more similar products

### 4. Recommendation Generation
- For a given product, the system retrieves the top-N products with the highest similarity scores
- Results are filtered and ranked to exclude duplicates or identical items

---

##  Results

- The system successfully returns contextually relevant product recommendations
- Products from the same category with similar colour or style descriptors consistently rank highest
- Recommendations reflect both semantic similarity (description language) and categorical similarity (product type, fabric)

---

##  Project Structure

```
zara_recommender_system/
│
├── data/
│   └── zara_products.csv          # Raw product dataset
│
├── notebooks/
│   └── zara_recommender.ipynb     # Main analysis and modelling notebook
│
├── src/
│   └── recommender.py             # Core recommendation logic (if modularised)
│
├── requirements.txt               # Python dependencies
└── README.md
```

---

##  How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/AmithaUS/zara_recommender_system.git
   cd zara_recommender_system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the notebook**
   ```bash
   jupyter notebook notebooks/zara_recommender.ipynb
   ```

4. **Get recommendations**
   ```python
   get_recommendations("Floral Midi Dress", top_n=5)
   ```

---

##  Sample Output

```
Input Product: Floral Midi Dress

Top 5 Recommendations:
1. Printed Wrap Dress        — Similarity: 0.87
2. Floral Mini Skirt         — Similarity: 0.81
3. Boho Printed Maxi Dress   — Similarity: 0.79
4. Satin Slip Dress          — Similarity: 0.74
5. Floral Crop Top           — Similarity: 0.68
```

---

##  Future Improvements

- Integrate **collaborative filtering** to incorporate user purchase and browsing history
- Add **image-based similarity** using CNN embeddings for visual recommendation
- Deploy as a **web application** using Streamlit or Flask
- Incorporate **user feedback loops** for personalised ranking

---

## 👩 Author

**Amitha U S**
BSc Statistics, Sree Sankara College, Kalady (2025)
[LinkedIn](https://linkedin.com/in/amithausamudra) | [GitHub](https://github.com/AmithaUS)

---

##  License

This project is for educational and portfolio purposes.
