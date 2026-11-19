# 🚢 Titanic Passenger Insights

An interactive data exploration and preprocessing dashboard built with **Python**, **Pandas**, and **Streamlit** for the *Programming for Data Science* course activity.

## 📋 Project Overview

This application demonstrates core data preparation techniques on the classic Titanic dataset and provides an interactive dashboard to explore passenger survival patterns without building a predictive model.

### Key Features & Implemented Tasks

1. **Load and Inspect Data:**
   - Loads the Titanic dataset directly via URL (or local CSV) and selects key fields (`survived`, `pclass`, `sex`, `age`, `fare`, `embarked`).
   - Displays dataset dimensions (rows & columns), data types, missing value counts, and a preview of the first 5 rows using `.loc` / `.iloc`.

2. **Handle Missing Values:**
   - Imputes missing `age` values using the **median**.
   - Imputes missing `embarked` values using the **mode** (most frequent category).
   - Displays missing value counts before and after imputation.

3. **Feature Transformation:**
   - **Min-Max Normalization:** Scales `age` and `fare` into `[0, 1]` ranges (`age_norm`, `fare_norm`) while preserving the original columns.
   - **Discretization:** Categorizes original `age` values into five age groups: `Child` (0–12), `Teen` (13–18), `Young Adult` (19–35), `Adult` (36–60), and `Older Adult` (61+).

4. **Dummy Coding:**
   - Applies one-hot encoding (`pd.get_dummies`) to categorical variables (`sex` and `embarked`) while keeping `survived` unencoded and leaving dummy columns unnormalized.
   - Shows a before-and-after comparison table.

5. **Interactive Dashboard:**
   - Sidebar filters for **Passenger Class (`pclass`)**, **Sex (`sex`)**, and **Age Group (`age_group`)**.
   - Dynamically updates key metrics (total selected passengers, survival rate, average fare) and survival rate bar charts.

6. **Observed Finding:**
   - Summarizes an observed statistical association between passenger class/sex and survival rate without claiming causality.

---

## 🛠️ Prerequisites & Installation

Make sure you have **Python 3.8+** installed on your Windows machine.

1. **Clone or create your project folder:**
   ```bash
   mkdir titanic_app
   cd titanic_app
   ```

2. **Place `app.py` inside the project folder.** *(Optional: Download `titanic.csv` into the same folder if you prefer loading it locally instead of via URL.)*

3. **Install the required Python libraries:**
   ```bash
   pip install streamlit pandas
   ```

---

## ▶️ How to Run the Application

1. Open a terminal (Command Prompt, PowerShell, or VS Code Terminal) inside your project directory.
2. Run the following command:
   ```bash
   streamlit run app.py
   ```
3. Your default web browser will automatically open the dashboard at `http://localhost:8501`.

---

## 📁 Project Structure

```text
titanic_app/
├── app.py          # Main Streamlit application script
└── README.md       # Project documentation
```# titanic_app
