# Olist Brazilian E-Commerce Delivery Prediction 🇧🇷📦

## Project Overview
This project focuses on predicting the **actual delivery time** for e-commerce orders in Brazil using the **Olist Dataset**. Accurate delivery estimation is critical for customer satisfaction and logistics optimization. 

The solution involves a robust Machine Learning pipeline that:
1.  Ingests and merges data from **9 relational tables** (100k+ orders).
2.  Engineers geospatial, temporal, and seller-performance features.
3.  Trains an optimized **XGBoost Regressor** to predict delivery days.

## 🛠️ Tech Stack & Tools
-   **Python**: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn.
-   **Geospatial**: Haversine formula for distance calculation.
-   **Machine Learning**: XGBoost, RandomizedSearchCV, TimeSeriesSplit.
-   **Engineering**: Feature engineering, data cleaning, pipeline development.

## 📊 Key Results
-   **Model**: XGBoost Regressor (Hyperparameter Tuned).
-   **Performance**:
    -   **MAE**: ~5.6 Days (Average error in prediction).
    -   **RMSE**: ~8-9 Days.
    -   *Note: This significantly outperforms baseline heuristics by capturing complex non-linear relationships like distance and seller reliability.*
    -   **Outlier Strategy**: Validated >3σ outliers as valid long-haul deliveries (avg 1087km) rather than data errors, preserving critical signal for remote regions.

## 🚀 Pipeline Steps

### 1. Data Ingestion & Merging
We merged 9 separate CSVs into a single "Master Table" effectively reconstructing the relational database schema:
-   `orders` + `order_items` + `products` + `sellers` + `customers` + `geolocation` + `reviews` + `payments`.

### 2. Feature Engineering
-   **Geospatial Distance**: Calculated the Haversine distance (km) between seller and customer using latitude/longitude data.
-   **Seller Performance**: Created cumulative features like `seller_historical_delay` to capture seller reliability without data leakage.
-   **Product Volume**: Synthesized `product_length_cm * height_cm * width_cm` to estimate package size.
-   **Temporal Features**: Extracted `purchase_weekday`, `purchase_hour`, and seasonality.

### 3. Model Training
-   **Time-Series Split**: Used time-based splitting (training on past data, testing on future data) to strictly prevent look-ahead bias/data leakage.
-   **XGBoost**: Selected for its efficiency with tabular data and ability to handle missing values.
-   **Tuning**: Optimized `learning_rate`, `max_depth`, `n_estimators`, and `subsample` using `RandomizedSearchCV`.

## 📈 Feature Importance
The most influential predictors were:
1.  **Distance (km)**: Physical distance is the strongest driver of delivery time.
2.  **Seller Historical Delay**: Past performance of sellers strongly correlates with future delivery speed.
3.  **Freight Value**: Higher freight often correlates with shipping complexity.

## 📂 Repository Structure
-   `eda.ipynb`: Exploratory Data Analysis, schema inspection, and cleaning.
-   `feature_engineering.ipynb`: Feature creation and dataset merging pipeline.
-   `delivery_model.ipynb`: Model training, hyperparameter tuning, and evaluation.
