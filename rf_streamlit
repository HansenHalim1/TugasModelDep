import streamlit as st
import joblib
import numpy as np

model = joblib.load('model.pkl')

def main():
    st.title('Machine learning model deployment')
    
    sepal_length = st.slider('sepal_length', min_value = 0.0, max_value = 10.0, value = 0.1)
    sepal_width = st.slider('sepal_length', min_value = 0.0, max_value = 10.0, value = 0.1)
    petal_length = st.slider('sepal_length', min_value = 0.0, max_value = 10.0, value = 0.1)
    petal_width = st.slider('sepal_length', min_value = 0.0, max_value = 10.0, value = 0.1)
    
    if st.button('make prediction'):
        features = [sepal_length,sepal_width,petal_length,petal_width]
        result = make_prediction(features)
        st.success(f'Prediction result is: {result}')
        
def make_prediction(features):
    input_array = np.array(features).reshape(1,-1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()
