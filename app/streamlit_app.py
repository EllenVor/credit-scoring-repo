import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix


st.title("Welcome to the Credit Scoring App")


@st.cache_data
def load_processed_df():
    data_path = Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'cs-scoring-processed.csv'
    df = pd.read_csv(data_path)
    df.drop('Unnamed: 0', axis=1, inplace=True, errors='ignore')
    return df


@st.cache_resource
def train_model(model_type):
    df = load_processed_df()
    if 'SeriousDlqin2yrs' not in df.columns:
        raise ValueError("Target 'SeriousDlqin2yrs' missing from dataset")
    X = df.loc[:, df.columns != 'SeriousDlqin2yrs']
    y = df['SeriousDlqin2yrs']
    feature_names = X.columns.tolist()

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    if model_type == 'Logistic Regression':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'Random Forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'XGBoost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    model.fit(X_train, y_train)
    return model, scaler, feature_names, X_test, y_test


def get_feature_importance_df(model, feature_names):
    """Return pandas DataFrame with columns: feature, importance, abs_importance sorted by abs_importance desc."""
    try:
        import pandas as _pd
    except Exception:
        import pandas as _pd

    # Logistic regression coefficients
    if hasattr(model, 'coef_'):
        coefs = model.coef_.ravel()
        df_fi = _pd.DataFrame({'feature': feature_names, 'importance': coefs})
        df_fi['abs_importance'] = df_fi['importance'].abs()
    # scikit-learn ensemble models and some sklearn wrappers
    elif hasattr(model, 'feature_importances_'):
        fi = model.feature_importances_
        df_fi = _pd.DataFrame({'feature': feature_names, 'importance': fi})
        df_fi['abs_importance'] = df_fi['importance']
    # xgboost core Booster via get_booster/get_score
    elif hasattr(model, 'get_booster'):
        booster = model.get_booster()
        try:
            score = booster.get_score(importance_type='weight')
            values = []
            for i in range(len(feature_names)):
                key = f'f{i}'
                values.append(score.get(key, 0))
            df_fi = _pd.DataFrame({'feature': feature_names, 'importance': values})
            df_fi['abs_importance'] = df_fi['importance']
        except Exception:
            # fallback
            fi = getattr(model, 'feature_importances_', None)
            if fi is not None:
                df_fi = _pd.DataFrame({'feature': feature_names, 'importance': fi})
                df_fi['abs_importance'] = df_fi['importance']
            else:
                raise
    else:
        # last resort: try to access feature_importances_
        fi = getattr(model, 'feature_importances_', None)
        if fi is not None:
            df_fi = _pd.DataFrame({'feature': feature_names, 'importance': fi})
            df_fi['abs_importance'] = df_fi['importance']
        else:
            raise ValueError('Model does not provide feature importances or coefficients')

    return df_fi.sort_values('abs_importance', ascending=False).reset_index(drop=True)


def plot_feature_importances_df(df_fi, top_n=20, signed=False, title=None):
    """Plot a horizontal bar chart with top_n features from df_fi DataFrame.
    If signed=True, bars will show signed coefficients (red negative, green positive)."""
    top_n = max(1, top_n)
    df_top = df_fi.head(top_n).copy()
    df_top = df_top.iloc[::-1]  # reverse so largest on top
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(df_top))))
    if signed:
        colors = df_top['importance'].apply(lambda x: 'red' if x < 0 else 'green')
        ax.barh(df_top['feature'], df_top['importance'], color=colors)
        ax.set_xlabel('Coefficient (signed)')
    else:
        ax.barh(df_top['feature'], df_top['importance'], color='steelblue')
        ax.set_xlabel('Importance')
    ax.set_title(title or 'Feature Importances')
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig


st.sidebar.header("Select Model & Show Metrics")
model_type = st.sidebar.radio("Choose model:", ["Logistic Regression", "Random Forest", "XGBoost"])
if st.sidebar.button(f"Train {model_type} & Show ROC"):
    st.info(f"Training {model_type} and generating ROC plot...")
    try:
        model, scaler, feature_names, X_test, y_test = train_model(model_type)
    except Exception as e:
        st.error(str(e))
    else:
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['feature_names'] = feature_names
        st.session_state['model_type'] = model_type

        y_scores = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        auc = roc_auc_score(y_test, y_scores)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'{model_type} ROC (AUC = {auc:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {model_type}')
        ax.legend(loc='lower right')
        ax.grid(True)
        st.pyplot(fig)
        st.write(f"ROC AUC: {auc:.3f}")
        y_pred = model.predict(X_test)
        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        # Feature importances / coefficients
        try:
            fi_df = get_feature_importance_df(model, feature_names)
            signed = hasattr(model, 'coef_')
            fig_fi = plot_feature_importances_df(fi_df, top_n=20, signed=signed, title=f'{model_type} Feature Importances')
            st.pyplot(fig_fi)
            if signed:
                st.caption('For Logistic Regression: positive coefficients increase probability of default (class 1)')
            if st.checkbox('Show feature importance table'):
                st.dataframe(fi_df)
        except Exception as e:
            st.warning(f"Cannot compute feature importances: {e}")


st.markdown("---")
st.header("New Client Prediction")

if 'model' not in st.session_state or st.session_state.get('model_type') != model_type:
    try:
        model, scaler, feature_names, _, _ = train_model(model_type)
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['feature_names'] = feature_names
        st.session_state['model_type'] = model_type
    except Exception as e:
        st.error(f"Cannot prepare model: {e}")


df = load_processed_df()
features = st.session_state.get('feature_names', df.columns[df.columns != 'SeriousDlqin2yrs'].tolist())

with st.form('predict_form'):
    st.write(f"Please fill in the client's information below for {model_type}:")
    user_inputs = {}
    for feat in features:
        default_val = float(df[feat].median()) if np.issubdtype(df[feat].dtype, np.number) else ''
        if np.issubdtype(df[feat].dtype, np.number):
            user_inputs[feat] = st.number_input(feat, value=default_val, format="%.2f")
        else:
            user_inputs[feat] = st.text_input(feat, value=default_val)

    submitted = st.form_submit_button('Predict')
    if submitted:
        X_new = pd.DataFrame([user_inputs], columns=features)
        for c in X_new.columns:
            X_new[c] = pd.to_numeric(X_new[c], errors='coerce').fillna(0)

        scaler = st.session_state['scaler']
        model = st.session_state['model']
        X_new_scaled = scaler.transform(X_new)
        prob = model.predict_proba(X_new_scaled)[:, 1][0]
        pred = model.predict(X_new_scaled)[0]
        decision = 'Approved' if pred == 0 else 'Rejected'
        st.success(f"Decision: {decision}")
        st.write(f"Probability of default (class 1): {prob:.3f}")

        if st.checkbox('Show model raw prediction and class'):
            st.write({'predicted_class': int(pred), 'probability_default': float(prob)})
        if st.checkbox('Show input features after scaling'):
            st.write(pd.DataFrame(X_new_scaled, columns=features))