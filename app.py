import streamlit as st
import pandas as pd

st.set_page_config(page_title="Titanic Passenger Insights", layout="wide")
st.title("🚢 Titanic Passenger Insights")

# 1. Veri Yükleme
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
    cols = ["survived", "pclass", "sex", "age", "fare", "embarked"]
    return pd.read_csv(url).loc[:, cols]

raw_df = load_data()

# 2. Yükleme ve İnceleme (Load & Inspect)
st.header("1. Veri Seti İncelemesi")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Boyut:** {raw_df.shape[0]} satır, {raw_df.shape[1]} sütun")
    st.dataframe(raw_df.dtypes.astype(str).rename("Veri Tipi"))
with col2:
    st.write("**Eksik Değerler (Öncesi):**")
    st.dataframe(raw_df.isnull().sum().rename("Eksik Sayısı"))

st.write("**İlk 5 Satır (`iloc`):**")
st.dataframe(raw_df.iloc[:5, :])

# 3. Eksik Değerleri Doldurma
st.header("2. Eksik Değerlerin İşlenmesi")
df = raw_df.fillna({
    "age": raw_df["age"].median(),
    "embarked": raw_df["embarked"].mode()[0]
})
st.dataframe(df.isnull().sum().rename("İşlem Sonrası Eksik"))

# 4. Özellik Dönüşümleri (Min-Max & Discretization)
st.header("3. Özellik Dönüşümleri")
for col in ["age", "fare"]:
    df[f"{col}_norm"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

labels = ["Child", "Teen", "Young Adult", "Adult", "Older Adult"]
df["age_group"] = pd.cut(df["age"], bins=[-1, 12, 18, 35, 60, 120], labels=labels)

st.dataframe(df[["age", "age_norm", "fare", "fare_norm", "age_group"]].head())

# 5. Sadeleştirilmiş Kategorik Kodlama (Tek Sütunda F/M ve Embarked)
st.header("4. Kategorik Kodlama (Sadeleştirilmiş)")
df_encoded = df.copy()
df_encoded["sex"] = df_encoded["sex"].map({"female": "F", "male": "M"})
# embarked zaten tek sütunda 'S', 'C', 'Q' olarak liman statüsünü tutuyor

c_before, c_after = st.columns(2)
with c_before:
    st.write("**Öncesi:**")
    st.dataframe(df[["survived", "sex", "embarked"]].head(5))
with c_after:
    st.write("**Sonrası (Tek Sütun F/M ve Embarked):**")
    st.dataframe(df_encoded[["survived", "sex", "embarked"]].head(5))

# 6. İnteraktif Dashboard
st.header("5. İnteraktif Hayatta Kalma Analizi")
st.sidebar.header("Filtreler")

pclass_sel = st.sidebar.multiselect("Sınıf (Pclass)", sorted(df_encoded["pclass"].unique()), default=sorted(df_encoded["pclass"].unique()))
sex_sel = st.sidebar.multiselect("Cinsiyet (Sex)", ["F", "M"], default=["F", "M"])
age_sel = st.sidebar.multiselect("Yaş Grubu", labels, default=labels)

filtered = df_encoded[
    df_encoded["pclass"].isin(pclass_sel) &
    df_encoded["sex"].isin(sex_sel) &
    df_encoded["age_group"].isin(age_sel)
]

m1, m2, m3 = st.columns(3)
m1.metric("Yolcu Sayısı", len(filtered))
m2.metric("Hayatta Kalma Oranı", f"%{(filtered['survived'].mean() * 100):.1f}" if len(filtered) else "%0.0")
m3.metric("Ortalama Ücret", f"${filtered['fare'].mean():.2f}" if len(filtered) else "$0.00")

g1, g2 = st.columns(2)
with g1:
    st.subheader("Sınıfa Göre Hayatta Kalma")
    st.bar_chart(filtered.groupby("pclass")["survived"].mean())
with g2:
    st.subheader("Yaş Grubuna Göre Hayatta Kalma")
    st.bar_chart(filtered.groupby("age_group", observed=False)["survived"].mean())

# 7. Bulgu
st.header("6. Temel Bulgu")
st.info("📌 **Bulgu:** 1. sınıf (Pclass 1) yolcuların ve kadın (`F`) yolcuların hayatta kalma oranları diğer gruplara göre belirgin şekilde daha yüksektir. *(Gözlemlenen istatistiksel bir ilişkidir; doğrudan nedensellik iddiası içermez.)*")
