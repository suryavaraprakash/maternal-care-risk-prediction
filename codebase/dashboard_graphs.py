
import streamlit as st
import pandas as pd
import plotly.express as px

class MaternalHealthDashboard:
    def __init__(self, csv_path="Maternal Health Risk Data Set.csv"):
        self.csv_path=csv_path
        self.maternal_health_data=self.fetch_data()

    def fetch_data(self):
        try:
            return pd.read_csv(self.csv_path)
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return None

    def create_bubble_chart(self):
        if self.maternal_health_data is None: return
        df=self.maternal_health_data
        st.subheader("Age vs Blood Sugar (Bubble size = Heart Rate)")
        fig=px.scatter(df,x="Age",y="BS",size="HeartRate",color="RiskLevel",hover_data=["SystolicBP","DiastolicBP","BodyTemp"])
        st.plotly_chart(fig,use_container_width=True)

    def create_pie_chart(self):
        if self.maternal_health_data is None:return
        st.subheader("Risk Level Distribution")
        counts=self.maternal_health_data["RiskLevel"].value_counts().reset_index()
        counts.columns=["RiskLevel","Count"]
        fig=px.pie(counts,names="RiskLevel",values="Count")
        st.plotly_chart(fig,use_container_width=True)

    def get_bubble_chart_data(self):
        return "Scatter plot showing relationship between Age and Blood Sugar. Bubble size indicates Heart Rate and color indicates Pregnancy Risk Level."

    def get_pie_graph_data(self):
        return "Pie chart showing distribution of Low, Mid and High pregnancy risk cases in the dataset."
