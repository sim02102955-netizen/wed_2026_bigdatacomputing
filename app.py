
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

results = joblib.load("results.pkl")

models = {
    "Linear": joblib.load("Linear.pkl"),
    "Poly": joblib.load("Poly.pkl"),
    "Ridge": joblib.load("Ridge.pkl")
}
st.header("모델 성능 비교")
st.dataframe(results) # R2, MSE, 특성개수 표시

# 막대그래프 시각화
fig, ax = plt.subplots()
results.plot(kind='bar', x='Model', y='Test R2', ax=ax, color='skyblue')
ax.set_title("Model Test R^2 Comparison")
st.pyplot(fig)

# 3. [조건 4] 실시간 예측 UI 구성 (사이드바)
st.sidebar.header("입력 특성 조절")
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
gdp = st.sidebar.slider("GDP", 0.0, 100000.0, 5000.0)
alcohol = st.sidebar.slider("Alcohol", 0.0, 20.0, 5.0)

# 선택된 특성을 데이터프레임으로 변환
input_data = pd.DataFrame([[bmi, gdp, alcohol]], columns=['BMI', 'GDP', 'Alcohol'])

# 모델 선택
selected_model_name = st.selectbox("사용할 모델 선택", ["Linear", "Poly", "Ridge"])
selected_model = models[selected_model_name]

# 예측
if st.button("예측하기"):
    prediction = selected_model.predict(input_data)
    st.subheader(f"예측된 기대수명: {prediction[0]:.2f} 세")
