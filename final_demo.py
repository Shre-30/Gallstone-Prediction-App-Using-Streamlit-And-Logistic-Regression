# =========================================================
# app.py — Gallstone Risk Prediction + PDF Diet Report + Insights
# =========================================================

import streamlit as st
import pandas as pd
import pickle
import os
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ===========================
# 1. Load Model and Scaler
# ===========================
model = pickle.load(open("logistic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Final fixed order (as per dataset)
feature_order = [
    "Age", "Gender", "Comorbidity", "Coronary Artery Disease (CAD)", "Hypothyroidism",
    "Hyperlipidemia", "Diabetes Mellitus (DM)", "Height", "Weight", "Body Mass Index (BMI)",
    "Total Body Water (TBW)", "Extracellular Water (ECW)", "Intracellular Water (ICW)",
    "Extracellular Fluid/Total Body Water (ECF/TBW)", "Total Body Fat Ratio (TBFR) (%)",
    "Lean Mass (LM) (%)", "Body Protein Content (Protein) (%)", "Visceral Fat Rating (VFR)",
    "Bone Mass (BM)", "Muscle Mass (MM)", "Obesity (%)", "Total Fat Content (TFC)",
    "Visceral Fat Area (VFA)", "Visceral Muscle Area (VMA) (Kg)", "Hepatic Fat Accumulation (HFA)",
    "Glucose", "Total Cholesterol (TC)", "Low Density Lipoprotein (LDL)",
    "High Density Lipoprotein (HDL)", "Triglyceride", "Aspartat Aminotransferaz (AST)",
    "Alanin Aminotransferaz (ALT)", "Alkaline Phosphatase (ALP)", "Creatinine",
    "Glomerular Filtration Rate (GFR)", "C-Reactive Protein (CRP)",
    "Hemoglobin (HGB)", "Vitamin D"
]

binary_cols = [
    "Gender", "Comorbidity", "Coronary Artery Disease (CAD)",
    "Hypothyroidism", "Hyperlipidemia", "Diabetes Mellitus (DM)",
    "Hepatic Fat Accumulation (HFA)"
]
continuous_cols = [c for c in feature_order if c not in binary_cols]

# ===========================
# 2. Diet Plan Data
# ===========================
pdf_plan = {
    "Vegetarian": [
        ("Oats / Whole grains", "Fiber, Magnesium", "1–2 servings/day"),
        ("Citrus fruits", "Vitamin C", "1 fruit/day"),
        ("Spinach, Broccoli", "Magnesium, Fiber", "1 cup/day"),
        ("Nuts & Seeds", "Omega-3", "Small handful/day"),
        ("Water", "Hydration", "8–10 glasses/day")
    ],
    "Eggetarian": [
        ("Boiled Eggs", "Protein", "3–4 per week"),
        ("Leafy Greens", "Fiber, Magnesium", "1 cup/day"),
        ("Fresh Fruits", "Vitamin C", "1 serving/day"),
        ("Low-fat Dairy", "Calcium, Protein", "1 glass/day"),
        ("Water", "Hydration", "8–10 glasses/day")
    ],
    "Non-Vegetarian": [
        ("Grilled Fish / Chicken", "Protein, Omega-3", "2–3 servings/week"),
        ("Lentils / Legumes", "Fiber", "1 cup/day"),
        ("Fresh Vegetables", "Magnesium, Fiber", "2 servings/day"),
        ("Avoid Red Meat", "High fat load", "Limit intake"),
        ("Water", "Hydration", "8–10 glasses/day")
    ]
}

# ===========================
# 3. PDF Report Generator
# ===========================
def create_diet_pdf_bytes(preference, patient_info=None):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40)
    styles = getSampleStyleSheet()
    elements = []

    logo_path = "logo final.png"
    if os.path.exists(logo_path):
        try:
            logo = Image(logo_path, width=100, height=100)
            logo.hAlign = 'CENTER'
            elements.append(logo)
            elements.append(Spacer(1, 10))
        except Exception:
            pass

    elements.append(Paragraph("Gallstone Diet Chart", styles['Title']))
    elements.append(Spacer(1, 12))

    if patient_info:
        elements.append(Paragraph("Patient Details:", styles['Heading2']))
        for k, v in patient_info.items():
            elements.append(Paragraph(f"<b>{k}:</b> {v}", styles['Normal']))
        elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Dietary Preference: <b>{preference}</b>", styles['Heading2']))
    elements.append(Spacer(1, 8))

    data = [["Food Item", "Nutrients", "Recommended Intake"]] + pdf_plan[preference]
    table = Table(data, colWidths=[150, 180, 150])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#4B9CD3")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 10)
    ]))
    elements.append(table)
    elements.append(Spacer(1, 18))
    elements.append(Paragraph(
        "<i>Note: This plan is generated automatically for educational purposes. "
        "Always consult a certified medical professional before dietary changes.</i>",
        styles['Normal']
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ===========================
# 4. Prediction Function
# ===========================
def predict_from_dataframe(df_samples):
    df_scaled = df_samples.copy()
    df_scaled_cont = scaler.transform(df_samples[continuous_cols])
    df_scaled[continuous_cols] = df_scaled_cont
    preds = model.predict(df_scaled)
    probs = model.predict_proba(df_scaled)[:, 1]
    return preds, probs

# ===========================
# 5. Example Patients
# ===========================
high_risk_df = pd.DataFrame([{
    "Age": 55, "Gender": 1, "Comorbidity": 1, "Coronary Artery Disease (CAD)": 1,
    "Hypothyroidism": 0, "Hyperlipidemia": 1, "Diabetes Mellitus (DM)": 1,
    "Height": 160, "Weight": 85, "Body Mass Index (BMI)": 33.2,
    "Total Body Water (TBW)": 47, "Extracellular Water (ECW)": 20, "Intracellular Water (ICW)": 27,
    "Extracellular Fluid/Total Body Water (ECF/TBW)": 43, "Total Body Fat Ratio (TBFR) (%)": 38,
    "Lean Mass (LM) (%)": 33, "Body Protein Content (Protein) (%)": 19, "Visceral Fat Rating (VFR)": 13,
    "Bone Mass (BM)": 2.6, "Muscle Mass (MM)": 28, "Obesity (%)": 35,
    "Total Fat Content (TFC)": 76, "Visceral Fat Area (VFA)": 98,
    "Visceral Muscle Area (VMA) (Kg)": 62, "Hepatic Fat Accumulation (HFA)": 1,
    "Glucose": 120, "Total Cholesterol (TC)": 210, "Low Density Lipoprotein (LDL)": 140,
    "High Density Lipoprotein (HDL)": 38, "Triglyceride": 190, "Aspartat Aminotransferaz (AST)": 32,
    "Alanin Aminotransferaz (ALT)": 36, "Alkaline Phosphatase (ALP)": 100, "Creatinine": 1.2,
    "Glomerular Filtration Rate (GFR)": 85, "C-Reactive Protein (CRP)": 0.5,
    "Hemoglobin (HGB)": 13, "Vitamin D": 22
}])

low_risk_df = pd.DataFrame([{
    "Age": 29, "Gender": 0, "Comorbidity": 0, "Coronary Artery Disease (CAD)": 0,
    "Hypothyroidism": 0, "Hyperlipidemia": 0, "Diabetes Mellitus (DM)": 0,
    "Height": 168, "Weight": 60, "Body Mass Index (BMI)": 21.3,
    "Total Body Water (TBW)": 56, "Extracellular Water (ECW)": 22, "Intracellular Water (ICW)": 34,
    "Extracellular Fluid/Total Body Water (ECF/TBW)": 46, "Total Body Fat Ratio (TBFR) (%)": 17,
    "Lean Mass (LM) (%)": 46, "Body Protein Content (Protein) (%)": 23, "Visceral Fat Rating (VFR)": 5,
    "Bone Mass (BM)": 3.3, "Muscle Mass (MM)": 36, "Obesity (%)": 17,
    "Total Fat Content (TFC)": 58, "Visceral Fat Area (VFA)": 48,
    "Visceral Muscle Area (VMA) (Kg)": 66, "Hepatic Fat Accumulation (HFA)": 0,
    "Glucose": 90, "Total Cholesterol (TC)": 160, "Low Density Lipoprotein (LDL)": 90,
    "High Density Lipoprotein (HDL)": 55, "Triglyceride": 80, "Aspartat Aminotransferaz (AST)": 22,
    "Alanin Aminotransferaz (ALT)": 25, "Alkaline Phosphatase (ALP)": 85, "Creatinine": 0.8,
    "Glomerular Filtration Rate (GFR)": 100, "C-Reactive Protein (CRP)": 0.1,
    "Hemoglobin (HGB)": 15, "Vitamin D": 40
}])

# ===========================
# 6. Streamlit Layout
# ===========================
st.set_page_config(page_title="Gallstone Risk Predictor", layout="wide")

col1, col2 = st.columns([1, 6])
with col1:
    st.image("logo final.png", width=100)
with col2:
    st.title("🩺 Gallstone Risk Prediction using Logistic Regression")
    st.caption("AI-based Clinical Decision Support for Gallstone Risk Assessment")

tabs = st.tabs(["Introduction", "Manual Prediction", "Predefined Examples", "Feature Insights"])

# ---------------------------
# TAB 1: Introduction
# ---------------------------
with tabs[0]:
    st.markdown("""
    ## Welcome to the Gallstone Risk Predictor..!!

    Gallstones are hardened deposits that can form in the gallbladder due to cholesterol imbalance or bile-related disorders.  
    Early detection of gallstone risk helps prevent severe complications like cholecystitis or bile duct obstruction.

    This AI-driven application uses a **Logistic Regression model** trained on 38 key biochemical and physiological features 
    (such as cholesterol, triglycerides, BMI, and liver enzymes) to predict whether a patient is at **high or low risk** 
    of developing gallstones.

    ### Application Highlights
    - Uses **StandardScaler normalization** and a trained **Logistic Regression model**
    - Computes actual probability of gallstone risk
    - Provides risk classification (Low / High)
    - Generates **personalized diet report (PDF)** with nutritional recommendations
    - Offers feature-level educational insights

    ### Disclaimer !!
    This tool is designed for **research and educational purposes**.  
    It should **not** replace professional medical consultation.
    """)

# ---------------------------
# TAB 2: Manual Prediction
# ---------------------------
with tabs[1]:
    st.subheader("Enter Patient Details in Correct Feature Order")
    user_data = {}
    for feature in feature_order:
        if feature == "Gender":
            gender_choice = st.selectbox("Gender", ["Male", "Female"])
            user_data["Gender"] = 1 if gender_choice == "Male" else 0
        elif feature in binary_cols:
            user_data[feature] = st.selectbox(feature, [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        else:
            user_data[feature] = st.number_input(feature, min_value=0.0, max_value=300.0, step=0.1)

    diet_pref = st.selectbox("Select Dietary Preference", ["Vegetarian", "Eggetarian", "Non-Vegetarian"])

    if st.button("🔍 Predict Risk"):
        user_df = pd.DataFrame([user_data])
        preds, probs = predict_from_dataframe(user_df)
        pred_class, prob = int(preds[0]), probs[0]

        if pred_class == 1:
            st.error(f"**High Risk Detected!** (Probability: {prob:.2f})")
            pdf_bytes = create_diet_pdf_bytes(diet_pref, user_data)
            st.download_button("📄 Download Diet Plan (PDF)", pdf_bytes, "gallstone_diet_plan.pdf", mime="application/pdf")
        else:
            st.success(f"Low Risk! Stay healthy (Probability: {prob:.2f})")

# ---------------------------
# TAB 3: Predefined Examples
# ---------------------------
with tabs[2]:
    st.subheader("Run Prediction for Example Patients")
    choice = st.radio("Select Example:", ["High Risk Example", "Low Risk Example"])
    diet_pref = st.selectbox("Dietary Preference", ["Vegetarian", "Eggetarian", "Non-Vegetarian"])

    if st.button("Run Example Prediction"):
        df_input = high_risk_df if choice == "High Risk Example" else low_risk_df
        preds, probs = predict_from_dataframe(df_input)
        pred_class, prob = preds[0], probs[0]

        st.write(df_input)
        if pred_class == 1:
            st.error(f"High Risk Detected (Probability: {prob:.2f})")
            pdf_bytes = create_diet_pdf_bytes(diet_pref, df_input.iloc[0].to_dict())
            st.download_button("Download Diet Plan (PDF)", pdf_bytes, "gallstone_diet_plan.pdf")
        else:
            st.success(f"Low Risk! You're good to go! (Probability: {prob:.2f})")

# ---------------------------
# TAB 4: Feature Insights
# ---------------------------
with tabs[3]:
    st.subheader("Feature Descriptions and Effects")
    feature_data = [
        ["Age", "Older age increases bile cholesterol saturation", "Dependent ↑", "<35 Low / 35–50 Medium / >50 High"],
        ["BMI", "High BMI indicates obesity, linked to gallstones", "Dependent ↑", "18–24 Good / 25–29 Moderate / ≥30 High"],
        ["Glucose", "Elevated glucose signals metabolic issues", "Dependent ↑", "<100 Normal / ≥126 High"],
        ["Cholesterol", "High cholesterol causes bile imbalance", "Dependent ↑", "<200 Good / ≥240 High"],
        ["Triglycerides", "High levels increase cholesterol in bile", "Dependent ↑", "<150 Normal / ≥200 High"],
        ["CRP", "Inflammation marker linked to gallstones", "Dependent ↑", "<0.3 Good / >1 High"],
        ["Vitamin D", "Low levels correlate with gallstone risk", "Dependent ↓", ">30 Good / <20 Low"],
    ]
    df_info = pd.DataFrame(feature_data, columns=["Feature", "Description", "Effect on Risk", "Healthy Range"])
    st.dataframe(df_info, use_container_width=True)

