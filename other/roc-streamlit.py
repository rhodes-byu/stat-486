# app.py
"""
Streamlit interactive demo: How the threshold tau affects the ROC curve.

Run:
    streamlit run app.py
"""
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide", page_title="ROC & Threshold (τ) Explorer")

# ---------------------
# Sidebar controls
# ---------------------
st.sidebar.header("Data / model settings")

n_samples = st.sidebar.slider("Samples (total)", 200, 2000, 1000, step=100)
pos_frac = st.sidebar.slider("Positive class fraction", 0.01, 0.5, 0.2, step=0.01)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

st.sidebar.markdown("---")
st.sidebar.write("Score distribution controls (Gaussian mixture):")
mean_neg = st.sidebar.slider("Mean (negatives)", -3.0, 3.0, -1.0, step=0.1)
std_neg = st.sidebar.slider("Std (negatives)", 0.1, 3.0, 1.0, step=0.05)
mean_pos = st.sidebar.slider("Mean (positives)", -1.0, 5.0, 1.0, step=0.1)
std_pos = st.sidebar.slider("Std (positives)", 0.1, 3.0, 1.0, step=0.05)

st.sidebar.markdown("---")
st.sidebar.write("Threshold τ and display")
tau = st.sidebar.slider("Threshold τ", 0.0, 1.0, 0.5, step=0.01)
score_transform = st.sidebar.selectbox("Score transform (sigmoid adds realism)", ["none", "sigmoid"], index=1)
st.sidebar.write("Tip: lower τ → more positives (↑TPR and ↑FPR). Higher τ → fewer positives.")

# ---------------------
# Data generation
# ---------------------
rng = np.random.RandomState(seed)

n_pos = int(np.round(n_samples * pos_frac))
n_neg = n_samples - n_pos

# sample raw scores from Gaussian distributions
scores_neg_raw = rng.normal(loc=mean_neg, scale=std_neg, size=n_neg)
scores_pos_raw = rng.normal(loc=mean_pos, scale=std_pos, size=n_pos)

scores_raw = np.concatenate([scores_pos_raw, scores_neg_raw])
y = np.concatenate([np.ones_like(scores_pos_raw), np.zeros_like(scores_neg_raw)]).astype(int)

# Optionally squash to (0,1) with sigmoid for nicer probability-like scores
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if score_transform == "sigmoid":
    # scale raw scores to avoid saturation
    scores = sigmoid((scores_raw - scores_raw.mean()) / (scores_raw.std() + 1e-12))
else:
    # min-max to 0..1 for thresholding convenience
    minv, maxv = scores_raw.min(), scores_raw.max()
    if maxv - minv == 0:
        scores = np.zeros_like(scores_raw)
    else:
        scores = (scores_raw - minv) / (maxv - minv)

# build dataframe
df = pd.DataFrame({"y": y, "score": scores})

# ---------------------
# Compute ROC and metrics
# ---------------------
fpr, tpr, thresholds = roc_curve(df["y"], df["score"])
roc_auc = auc(fpr, tpr)

# Determine classification at tau
y_pred = (df["score"] >= tau).astype(int)

# Confusion matrix and derived metrics
tn, fp, fn, tp = confusion_matrix(df["y"], y_pred).ravel()
precision = precision_score(df["y"], y_pred, zero_division=0)
recall = recall_score(df["y"], y_pred, zero_division=0)
fpr_tau = fp / (fp + tn) if (fp + tn) > 0 else 0.0
tpr_tau = recall
support_pos = int(df["y"].sum())
support_neg = len(df) - support_pos

# ---------------------
# Layout
# ---------------------
st.title("ROC Curve & Threshold (τ) Explorer")
st.markdown(
    "Visualize how changing the decision threshold τ moves the classifier along the ROC curve "
    "and affects TPR, FPR, precision, and the confusion matrix."
)

col1, col2 = st.columns([1, 1])

# ROC plot (left)
with col1:
    st.subheader("ROC curve")
    fig_roc = go.Figure()
    fig_roc.add_trace(
        go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {roc_auc:.3f})", hoverinfo="x+y")
    )
    # add diagonal for random classifier
    fig_roc.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random classifier (diag)", line=dict(dash="dash"))
    )
    # mark the selected tau point: compute its FPR/TPR
    fig_roc.add_trace(
        go.Scatter(
            x=[fpr_tau],
            y=[tpr_tau],
            mode="markers+text",
            marker=dict(size=12, symbol="x"),
            text=[f"τ={tau:.2f}"],
            textposition="bottom right",
            name="Selected τ"
        )
    )
    fig_roc.update_layout(
        xaxis_title="False Positive Rate (FPR)",
        yaxis_title="True Positive Rate (TPR)",
        xaxis=dict(range=[-0.02, 1.02]),
        yaxis=dict(range=[-0.02, 1.02]),
        height=450,
        margin=dict(t=30, b=20, l=40, r=20)
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown(
        f"**Selected threshold τ = {tau:.2f}**  &nbsp;&nbsp; "
        f"TP={tp} &nbsp; FP={fp} &nbsp; FN={fn} &nbsp; TN={tn}"
    )
    st.write(f"TPR (recall) = {tpr_tau:.3f}    •    FPR = {fpr_tau:.3f}    •    Precision = {precision:.3f}")

# Score distributions + threshold (right)
with col2:
    st.subheader("Score distributions (positives vs negatives)")
    # Build histograms for positives and negatives
    pos_scores = df[df["y"] == 1]["score"]
    neg_scores = df[df["y"] == 0]["score"]

    # Use density histograms with Plotly
    fig_dist = go.Figure()
    fig_dist.add_trace(
        go.Histogram(
            x=pos_scores,
            histnorm="probability density",
            name=f"Pos (n={len(pos_scores)})",
            opacity=0.6,
            nbinsx=40
        )
    )
    fig_dist.add_trace(
        go.Histogram(
            x=neg_scores,
            histnorm="probability density",
            name=f"Neg (n={len(neg_scores)})",
            opacity=0.6,
            nbinsx=40
        )
    )
    # add vertical line at tau
    fig_dist.add_vline(x=tau, line=dict(color="black", dash="dash"), annotation_text=f"τ={tau:.2f}", annotation_position="top right")

    fig_dist.update_layout(barmode="overlay", xaxis_title="Score / probability", yaxis_title="Density", height=450)
    st.plotly_chart(fig_dist, use_container_width=True)

# ---------------------
# Detailed metrics and explanation (below)
# ---------------------
st.markdown("---")
st.subheader("Detailed numbers & interpretation")

colA, colB, colC = st.columns([1, 1, 1])

with colA:
    st.metric("True positive rate (TPR / recall)", f"{tpr_tau:.3f}")
    st.metric("False positive rate (FPR)", f"{fpr_tau:.3f}")

with colB:
    st.metric("Precision (Positive Predictive Value)", f"{precision:.3f}")
    st.metric("Support (positives)", f"{support_pos}")

with colC:
    st.metric("Threshold τ", f"{tau:.3f}")
    st.metric("AUC", f"{roc_auc:.3f}")

st.write(
    """
**Interpretation tips**
- Moving τ to the left (lower) increases predicted positives → **TPR ↑** but also **FPR ↑**.
- Moving τ to the right (higher) reduces predicted positives → **TPR ↓** and **FPR ↓**.
- The ROC curve plots TPR vs FPR as τ varies. The red/marked point above shows the FPR/TPR for the currently selected τ.
- AUC measures ranking quality (probability a random positive scores above a random negative). It is independent of τ.
"""
)

st.markdown("### Confusion matrix (for selected τ)")
cm_df = pd.DataFrame(
    [[tn, fp], [fn, tp]],
    index=["Actual 0 (neg)", "Actual 1 (pos)"],
    columns=["Pred 0 (neg)", "Pred 1 (pos)"],
)
st.dataframe(cm_df.style.format("{:.0f}"))

st.markdown("---")
st.markdown("**Extensions you can try**")
st.write(
    """
- Replace Gaussian scores with a small classifier (e.g., logistic regression) trained on features, then manipulate τ.
- Add a slider to show multiple τs at once and plot them on the ROC curve.
- Overlay iso-F1 or iso-Precision lines on the ROC plot (requires translating coordinates).
- Show Precision-Recall curve side-by-side (useful when classes are imbalanced).
"""
)

st.caption("App created for teaching/demo purposes. Adjust the distributions and τ to see different trade-offs.")
