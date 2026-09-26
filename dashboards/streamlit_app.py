"""MVP dashboard (fast demo before the Tableau version).

Run: streamlit run dashboards/streamlit_app.py
"""
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Analytics Suite", layout="wide")
st.title("Customer Analytics Suite — MVP Dashboard")

tab1, tab2, tab3 = st.tabs(["Executive Overview", "Customer Explorer", "Offer Performance"])

with tab1:
    st.subheader("Executive Overview")
    st.info("TODO: load scored customer base and show churn rate / avg CLV / trend.")

with tab2:
    st.subheader("Customer Explorer")
    st.info("TODO: filterable table by segment, risk tier, priority score.")
    demo = pd.DataFrame({
        "customer_id": ["C1", "C2", "C3"],
        "churn_probability": [0.78, 0.22, 0.55],
        "clv": [1200, 300, 900],
        "priority_score": [0, 0, 0],
    })
    demo["priority_score"] = demo["churn_probability"] * demo["clv"]
    st.dataframe(demo)

with tab3:
    st.subheader("Offer Performance")
    st.info("TODO: hit rate per offer from NBO module results.")
