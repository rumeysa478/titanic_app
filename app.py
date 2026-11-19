import streamlit as st
import pandas as pd

st.set_page_config(page_title="Titanic Passenger Insights", layout="wide")
st.title("🚢 Titanic Passenger Insights")
st.write("Titanic veri seti üzerinde veri hazırlama ve hayatta kalma analizi.")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    df = pd.read_csv(url)
    cols = ["survived", "pclass", "sex", "age", "fare", "embarked"]
    return df.loc[:, cols]

raw_df = load_data()

st.header("1. Veri Seti İncelemesi (Ham Veri)")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Satır ve Sütun Sayısı:** {raw_df.shape[0]} satır, {raw_df.shape[1]} sütun")
    st.write("**Sütun İsimleri ve Veri Tipleri:**")
    st.dataframe(raw_df.dtypes.astype(str).rename("Veri Tipi"))
with col2:
    st.write("**Eksik Değer Sayıları (İşlem Öncesi):**")
    st.dataframe(raw_df.isnull().sum().rename("Eksik Sayısı"))

st.write("**İlk 5 Satır (`iloc` ile seçim):**")
st.dataframe(raw_df.iloc[:5, :])

st.header("2. Eksik Değerlerin İşlenmesi")
df = raw_df.copy()
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

st.write("**Eksik Değer Sayıları (İşlem Sonrası):**")
st.dataframe(df.isnull().sum().rename("Eksik Sayısı"))

# 4. Özellik Dönüşümleri (Transformed Features: Min-Max & Discretization)
st.header("3. Özellik Dönüşümleri (Normalizasyon ve Gruplama)")
df["age_norm"] = (df["age"] - df["age"].min()) / (df["age"].max() - df["age"].min())
df["fare_norm"] = (df["fare"] - df["fare"].min()) / (df["fare"].max() - df["fare"].min())

bins = [-1, 12, 18, 35, 60, 120]
labels = ["Child", "Teen", "Young Adult", "Adult", "Older Adult"]
df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)

st.dataframe(df[["age", "age_norm", "fare", "fare_norm", "age_group"]].head())

st.header("4. Dummy Coding (Kategorik Değişkenler)")
st.write("**Öncesi (`sex` ve `embarked`):**")
st.dataframe(df[["survived", "sex", "embarked"]].head(3))

df_encoded = pd.get_dummies(df, columns=["sex", "embarked"], drop_first=False)
st.write("**Sonrası (Dummy Değişkenler Eklenmiş Hali):**")
st.dataframe(df_encoded.head(3))

st.header("5. İnteraktif Hayatta Kalma Analizi")
st.sidebar.header("Filtreler")

selected_pclass = st.sidebar.multiselect("Yolcu Sınıfı (Pclass)", options=sorted(df["pclass"].unique()), default=sorted(df["pclass"].unique()))
selected_sex = st.sidebar.multiselect("Cinsiyet (Sex)", options=df["sex"].unique(), default=df["sex"].unique())
selected_age_group = st.sidebar.multiselect("Yaş Grubu (Age Group)", options=labels, default=labels)

filtered_df = df[
    (df["pclass"].isin(selected_pclass)) &
    (df["sex"].isin(selected_sex)) &
    (df["age_group"].isin(selected_age_group))
]

m1, m2, m3 = st.columns(3)
m1.metric("Seçili Yolcu Sayısı", len(filtered_df))
survival_rate = (filtered_df["survived"].mean() * 100) if len(filtered_df) > 0 else 0
m2.metric("Hayatta Kalma Oranı", f"%{survival_rate:.1f}")
avg_fare = filtered_df["fare"].mean() if len(filtered_df) > 0 else 0
m3.metric("Ortalama Bilet Ücreti", f"${avg_fare:.2f}")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Sınıfa Göre Hayatta Kalma Oranı")
    st.bar_chart(filtered_df.groupby("pclass")["survived"].mean())
with c2:
    st.subheader("Yaş Grubuna Göre Hayatta Kalma Oranı")
    st.bar_chart(filtered_df.groupby("age_group", observed=False)["survived"].mean())

st.header("6. Temel Bulgu (Observed Association)")
st.info("📌 **Bulgu:** Veri setindeki grafikler incelendiğinde, 1. sınıf (Pclass 1) yolcuların ve kadın yolcuların hayatta kalma oranlarının diğer gruplara kıyasla belirgin şekilde daha yüksek olduğu gözlemlenmektedir. *(Not: Bu durum değişkenler arasında istatistiksel bir ilişki olduğunu gösterir, doğrudan nedensellik iddiası taşımaz.)*")