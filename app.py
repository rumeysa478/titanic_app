import streamlit as st
import pandas as pd
import altair as alt

# 1. Sayfa Yapılandırması ve Renk Paleti (Navy & Teal)
st.set_page_config(page_title="Titanic Passenger Insights", layout="wide")

NAVY = "#1B365D"
TEAL = "#008080"
LIGHT_TEAL = "#4DB6AC"
PALETTE = [NAVY, TEAL, LIGHT_TEAL]

st.title("🚢 Titanic Passenger Insights")
st.caption("Explore survival patterns across passenger groups")

# 2. Veri Yükleme ve Ön İşleme
@st.cache_data
def load_and_prepare_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    cols = ["survived", "pclass", "sex", "age", "fare", "embarked"]
    raw = pd.read_csv(url).loc[:, cols]

    # Cleaned kopya (Eksik değer doldurma)
    cleaned = raw.fillna({
        "age": raw["age"].median(),
        "embarked": raw["embarked"].mode()[0]
    })

    # Transformed kopya (Normalizasyon, Gruplama, Kategorik Kodlama)
    transformed = cleaned.copy()
    for col in ["age", "fare"]:
        transformed[f"{col}_norm"] = (transformed[col] - transformed[col].min()) / (transformed[col].max() - transformed[col].min())

    age_labels = ["Child", "Teen", "Young Adult", "Adult", "Older Adult"]
    transformed["age_group"] = pd.cut(transformed["age"], bins=[-1, 12, 18, 35, 60, 120], labels=age_labels)
    
    # Tek sütunda F / M ve mevcut Embarked kodu
    transformed["sex_code"] = transformed["sex"].map({"female": "F", "male": "M"})
    transformed["embarked_code"] = transformed["embarked"]  # S, C, Q statüleri

    return raw, cleaned, transformed

raw_df, cleaned_df, df = load_and_prepare_data()
age_labels = ["Child", "Teen", "Young Adult", "Adult", "Older Adult"]

# 3. Kenar Çubuğu (Sidebar) Filtreleri
st.sidebar.header("Filters")
pclass_opts = ["All"] + sorted(df["pclass"].unique().tolist())
sex_opts = ["All", "F", "M"]
age_opts = ["All"] + age_labels

sel_pclass = st.sidebar.selectbox("Passenger class", pclass_opts)
sel_sex = st.sidebar.selectbox("Sex", sex_opts)
sel_age = st.sidebar.selectbox("Age group", age_opts)

filtered_df = df.copy()
if sel_pclass != "All":
    filtered_df = filtered_df[filtered_df["pclass"] == sel_pclass]
if sel_sex != "All":
    filtered_df = filtered_df[filtered_df["sex_code"] == sel_sex]
if sel_age != "All":
    filtered_df = filtered_df[filtered_df["age_group"] == sel_age]

# 4. Dört Sekmeli Ana Arayüz
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Explore", "Preprocessing", "Data Explorer"])

# --- TAB 1: OVERVIEW ---
with tab1:
    k1, k2, k3, k4 = st.columns(4)
    total_p = len(filtered_df)
    survivors = int(filtered_df["survived"].sum()) if total_p else 0
    surv_rate = (filtered_df["survived"].mean() * 100) if total_p else 0.0
    avg_fare = filtered_df["fare"].mean() if total_p else 0.0

    k1.metric("Passengers in selection", total_p)
    k2.metric("Survivors", survivors)
    k3.metric("Survival rate", f"{surv_rate:.1f}%")
    k4.metric("Average fare", f"${avg_fare:.2f}")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Survival Counts")
        surv_counts = filtered_df["survived"].map({0: "Did Not Survive", 1: "Survived"}).value_counts().reset_index()
        surv_counts.columns = ["Status", "Count"]
        chart_counts = alt.Chart(surv_counts).mark_bar().encode(
            x=alt.X("Status:N", title="Survival Status", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Count:Q", title="Number of Passengers"),
            color=alt.Color("Status:N", scale=alt.Scale(domain=["Did Not Survive", "Survived"], range=[NAVY, TEAL]), legend=None)
        ).properties(height=320)
        st.altair_chart(chart_counts, use_container_width=True)

    with c2:
        st.subheader("Survival Rate by Passenger Class")
        class_rate = filtered_df.groupby("pclass", as_index=False)["survived"].mean()
        class_rate["survived"] *= 100
        chart_class = alt
