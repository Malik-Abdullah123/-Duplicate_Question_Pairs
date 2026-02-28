import streamlit as st
import helper

import pickle
import os

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    import traceback
    traceback.print_exc()

st.header("Duplicate Question Pairs")

q1= st.text_input("Enter the first question")
q2= st.text_input("Enter the second question")

if st.button("Find"):
    query=(helper.query_point_creator(q1,q2))
    result=model.predict(query)[0]


    if result:
        st.header("Duplicate")
    else:
        st.header("Not Duplicate")