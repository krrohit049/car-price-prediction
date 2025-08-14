import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
st.title("🚗 Car Price Prediction")
st.write("Enter details to get an estimated price for your car.")

st.set_page_config(page_title="Car Price Prediction", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "car_price_dataset.csv")
    return pd.read_csv(csv_path)

df = load_data()

# Tabs
tab1, tab2, tab3 = st.tabs(["📄 Dataset", "⚙️ Train Model", "🔮 Predict Price"])

# -------------------- Dataset Tab --------------------
with tab1:
    st.title("📄 Dataset Preview")
    st.dataframe(df.head())

# -------------------- Train Model Tab --------------------
with tab2:
    st.title("Model Training")

    st.subheader("Select Features for Training")
    all_features = [col for col in df.columns if col.lower() != "price"]
    features = st.multiselect("Choose features", all_features)

    test_size = st.slider("Test Size (%)", 10, 50, 20)
    model_choice = st.selectbox("Choose Model", ["Linear Regression", "Decision Tree", "Random Forest"])
    train_button = st.button("Train Model")

    if train_button:
        if not features:
            st.error("Please select at least one feature.")
        else:
            X = pd.get_dummies(df[features], drop_first=True)
            y = df["Price"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

            if model_choice == "Linear Regression":
                model = LinearRegression()
            elif model_choice == "Decision Tree":
                model = DecisionTreeRegressor()
            else:
                model = RandomForestRegressor()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):,.2f}")
            st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):,.2f}")
            st.write(f"**R²:** {r2_score(y_test, y_pred):.2f}")

            # Save to session state
            st.session_state.model = model
            st.session_state.features = features
            st.session_state.df = df

# -------------------- Predict Price Tab --------------------
with tab3:
    st.title("Predict Car Price")

    if "model" not in st.session_state:
        st.warning("Please train a model first in the 'Train Model' tab.")
    else:
        model = st.session_state.model
        features = st.session_state.features
        df = st.session_state.df

        input_data = {}
        for col in features:
            if df[col].dtype == 'object':
                input_data[col] = st.selectbox(f"Select {col}", df[col].unique())
            else:
                input_data[col] = st.number_input(f"Enter {col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))

        if st.button("Predict Price"):
            input_df = pd.DataFrame([input_data])
            input_df = pd.get_dummies(input_df, drop_first=True)

            # Align with training columns
            train_cols = pd.get_dummies(df[features], drop_first=True).columns
            input_df = input_df.reindex(columns=train_cols, fill_value=0)

            prediction = model.predict(input_df)[0]
            st.success(f"💰 Predicted Price: ₹{prediction:,.2f}")
