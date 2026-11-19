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
        chart_class = alt.Chart(class_rate).mark_bar(color=TEAL).encode(
            x=alt.X("pclass:O", title="Passenger Class", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("survived:Q", title="Survival Rate (%)", scale=alt.Scale(domain=[0, 100]))
        ).properties(height=320)
        st.altair_chart(chart_class, use_container_width=True)

    st.caption("Note: Rates describe this dataset only.")

# --- TAB 2: EXPLORE ---
with tab2:
    st.subheader("Compare Passenger Groups")
    g1, g2 = st.columns(2)

    with g1:
        st.markdown("**Survival Rate by Class**")
        c_rate = filtered_df.groupby("pclass", as_index=False)["survived"].mean()
        c_rate["survived"] *= 100
        ch1 = alt.Chart(c_rate).mark_bar().encode(
            x=alt.X("pclass:O", title="Passenger Class", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("survived:Q", title="Survival Rate (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("pclass:O", scale=alt.Scale(range=PALETTE), legend=None)
        ).properties(height=280)
        st.altair_chart(ch1, use_container_width=True)

    with g2:
        st.markdown("**Survival Rate by Sex**")
        s_rate = filtered_df.groupby("sex_code", as_index=False)["survived"].mean()
        s_rate["survived"] *= 100
        ch2 = alt.Chart(s_rate).mark_bar().encode(
            x=alt.X("sex_code:N", title="Sex (F = Female, M = Male)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("survived:Q", title="Survival Rate (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("sex_code:N", scale=alt.Scale(domain=["F", "M"], range=[TEAL, NAVY]), legend=None)
        ).properties(height=280)
        st.altair_chart(ch2, use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        st.markdown("**Age Distribution**")
        ch3 = alt.Chart(filtered_df).mark_bar(color=NAVY).encode(
            x=alt.X("age:Q", bin=alt.Bin(maxbins=20), title="Age (Years)"),
            y=alt.Y("count():Q", title="Number of Passengers")
        ).properties(height=280)
        st.altair_chart(ch3, use_container_width=True)

    with g4:
        st.markdown("**Average Fare by Class**")
        f_class = filtered_df.groupby("pclass", as_index=False)["fare"].mean()
        ch4 = alt.Chart(f_class).mark_bar().encode(
            x=alt.X("pclass:O", title="Passenger Class", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("fare:Q", title="Average Fare ($)"),
            color=alt.Color("pclass:O", scale=alt.Scale(range=PALETTE), legend=None)
        ).properties(height=280)
        st.altair_chart(ch4, use_container_width=True)

    st.info("**Finding:** Within this dataset, first-class passengers and female passengers (`F`) exhibit noticeably higher survival rates compared to other passenger groups.")

# --- TAB 3: PREPROCESSING ---
with tab3:
    with st.expander("Section 1: Missing Values", expanded=True):
        st.caption("Missing values in `age` were imputed using the median, while missing `embarked` entries were filled with the most frequent port.")
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.write("**Before Imputation**")
            st.dataframe(raw_df[["age", "embarked"]].isnull().sum().rename("Missing Count"))
        with m_col2:
            st.write("**After Imputation**")
            st.dataframe(cleaned_df[["age", "embarked"]].isnull().sum().rename("Missing Count"))

    with st.expander("Section 2: Normalization", expanded=False):
        st.caption("Min-Max normalization rescales `age` and `fare` to a [0, 1] range while preserving their original columns.")
        st.dataframe(df[["age", "age_norm", "fare", "fare_norm"]].head(6), use_container_width=True)

    with st.expander("Section 3: Age Discretization", expanded=False):
        st.caption("Continuous `age` values are grouped into five ordered life-stage categories (`age_group`).")
        st.dataframe(df[["age", "age_group"]].head(6), use_container_width=True)

    with st.expander("Section 4: Categorical Coding (Sex & Embarked)", expanded=False):
        st.caption("Categorical values are simplified into single compact columns (`sex_code` as F/M and `embarked_code` as port status) without normalizing target or indicator columns.")
        c_b, c_a = st.columns(2)
        with c_b:
            st.write("**Before**")
            st.dataframe(cleaned_df[["survived", "sex", "embarked"]].head(5))
        with c_a:
            st.write("**After**")
            st.dataframe(df[["survived", "sex_code", "embarked_code"]].head(5))

# --- TAB 4: DATA EXPLORER ---
with tab4:
    st.subheader("Dataset Records")
    view_choice = st.radio("Select view:", ["Raw", "Cleaned", "Transformed"], horizontal=True)

    if view_choice == "Raw":
        display_df = raw_df.copy()
        default_cols = ["survived", "pclass", "sex", "age", "fare", "embarked"]
    elif view_choice == "Cleaned":
        display_df = cleaned_df.copy()
        default_cols = ["survived", "pclass", "sex", "age", "fare", "embarked"]
    else:
        display_df = filtered_df.copy()
        default_cols = ["survived", "pclass", "sex_code", "age", "age_group", "fare", "embarked_code"]

    selected_cols = st.multiselect("Choose columns to display:", options=display_df.columns.tolist(), default=default_cols)
    st.write(f"**Row count:** {len(display_df)} records")

    if selected_cols:
        st.dataframe(display_df[selected_cols], use_container_width=True, height=400)
    else:
        st.warning("Please select at least one column to display.")
