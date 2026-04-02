import streamlit as st
LABEL ={0: "democrate", 1:"republican"}
def get_prediction(input_text):
    # TODO - task 3
    # -----------------------------------
    # Goal: our goal is to complete the implementation of this function, 
    #       which takes input text and returns a prediction result from a pre-trained model.
    import requests
    response = requests.post(
        "http://model_inference_endpoint:8000/get-prediction/",
        json={"input_texts": input_text}
        )
    result = response.json().get("prediction")
    if result is not None:
        return LABEL.get(result[0], result[0])
    return None

# Streamlit page configuration
st.set_page_config(page_title="Tweet Classifier", layout="wide")

# Streamlit UI components
st.title("Classify your tweet")

# User inputs the tweet
tweet_input = st.text_input("Enter your tweet", "")

# Button to trigger prediction
if st.button("Classify Tweet"):
    # Get prediction
    prediction = get_prediction(tweet_input)
    
    # Display the prediction
    st.write("Prediction:", prediction)

