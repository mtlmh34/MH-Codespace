import streamlit as st
import pandas as pd

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay,precision_recall_curve,PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score

from ucimlrepo import fetch_ucirepo 





def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Streamlit WebApp")
    st.markdown("Are your mashrooms edible or poisionous? 🍄")
    st.sidebar.markdown("Some Text is needed")
    
    # load data
    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv("/Users/minghaooo/Documents/GitHub Repos/MH Codespace/App/mushrooms.csv")
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    df = load_data()
    
    if st.sidebar.checkbox("Show raw data: ", False):
        st.subheader("Mushroom Dataset")
        st.write(df)    


if __name__ == '__main__':
    main()