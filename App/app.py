import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, RocCurveDisplay,precision_recall_curve,PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score

from ucimlrepo import fetch_ucirepo 





def main():
    ### UI
    st.title("Binary Classification Web App")
    st.sidebar.title("Streamlit WebApp")
    st.markdown("Are your mashrooms edible or poisionous? 🍄")
    st.sidebar.markdown("Some Text is needed")
    
    ### load data
    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv("/Users/minghaooo/Documents/GitHub Repos/MH Codespace/App/mushrooms.csv")
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    
    ### Split dataset
    @st.cache_data(persist=True)
    def split(df):
        y = df['class']
        x = df.drop(columns = ['class'])
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=1)
        return x_train, x_test, y_train, y_test

    # model eval
    def plot_metrics(metrics_list):
        # Prepare which plots to show and their order
        plot_types = []
        if 'Confusion Matrix' in metrics_list:
            plot_types.append('cm')
        if 'ROC Curve' in metrics_list:
            plot_types.append('roc')
        if 'Precision-Recall Curve' in metrics_list:
            plot_types.append('pr')
        n = len(plot_types)
        if n == 0:
            return

        fig, axes = plt.subplots(1, n, figsize=(5*n, 8))
        
        st.subheader(f"Matrix Evaluation: {', '.join(metrics_list)}")
        
        if n == 1:
            axes = [axes]  # make axes always iterable

        for idx, plot_type in enumerate(plot_types):
            ax = axes[idx]
            if plot_type == 'cm':
                cm = confusion_matrix(y_test, y_pred, labels=class_names)
                display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
                display.plot(ax=ax, cmap='Blues', colorbar=False)
                ax.set_title("Confusion Matrix")
            elif plot_type == 'roc':
                fpr, tpr, thresholds = roc_curve(y_test, y_score)
                roc_auc=roc_auc_score(y_test,y_score)
                display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
                display.plot(ax=ax)
                ax.set_title("ROC Curve")
                ax.legend(loc="lower right")
            elif plot_type == 'pr':
                precision, recall, thresholds = precision_recall_curve(y_test, y_score)
                display = PrecisionRecallDisplay(precision=precision, recall=recall)
                display.plot(ax=ax)
                ax.set_title("Precision-Recall Curve")

        st.pyplot(fig)
            

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = [0,1]
    
    st.sidebar.subheader('Choose Classifier')
    classifier = st.sidebar.selectbox("Classifier", ("SVM", "Logistic Regression", "Random Forest"))
    
    # Hypermarameter Tuning
    if classifier == "SVM":
        st.sidebar.subheader("Model Hyperparameters")
        
        selected_C = st.sidebar.number_input("C (Regularization Parameter)", 0.01, 10.0, step=0.01, key='C')
        selected_kernel = st.sidebar.radio("Kernel", options=("rbf", "linear"), key = 'kernel')
        selected_gamma = st.sidebar.radio("Gamma", ("scale", "auto"), key='gamma')
        
        metrics = st.sidebar.multiselect("Choose metrics to plot:", ("Confusion Matrix", "ROC Curve","Precision-Recall Curve"))
        
        # Button to refresh model results
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Show SVM Results")
            
            model = SVC(C=selected_C, kernel=selected_kernel, gamma=selected_gamma)
            model.fit(x_train,y_train)
            
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            y_score = model.decision_function(x_test)
            
            st.write("Accuracy: ", round(accuracy,3))
            st.write("Precision: ", round(precision_score(y_test,y_pred, labels=class_names),3))
            st.write("Recall: ", round(recall_score(y_test,y_pred, labels=class_names),3))
            plot_metrics(metrics)
    
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Dataset")
        st.write(df)    


if __name__ == '__main__':
    main()