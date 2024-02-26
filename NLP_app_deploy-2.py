import uvicorn
from fastapi import FastAPI, Query
import pickle
import pandas as pd
import warnings
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Create an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['Name', 'Surname', 'Opinion', 'Sentiment Prediction'])

# Create a FastAPI app instance
app = FastAPI()

# Load the pickled model using a relative file path
with open('nlp.pkl', "rb") as model_file:
    model = pickle.load(model_file)

# Load TF-IDF vectorizer
with open('tfidfVectorizer.pkl', 'rb') as tfidf_file:
    tfidf_vectorizer = pickle.load(tfidf_file)

html_content = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Opinion Sentiment Prediction</title>
    <style>
        body {
            background-image: url('https://wallpapers.com/images/hd/high-quality-black-wood-texture-po65aj0bvhhkppwa.jpg');
            background-size: cover;
            text-align: center;
            height: 100vh;
            margin: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        .form-container {
            background-color: #fafafa; /* Dark white color */
            padding: 20px;
            border-radius: 10px;
            width: 80%;
            max-width: 400px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            animation: fadeIn 0.5s ease-out;
            color: #333; /* Set text color to a darker color */
        }

        @keyframes fadeIn {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }

            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        label {
            font-size: 18px;
            color: #333;
            display: block;
            margin-bottom: 5px;
            text-align: left;
        }

        input {
            font-size: 16px;
            padding: 8px;
            margin-bottom: 15px;
            width: 100%;
            box-sizing: border-box;
        }

        input[type="submit"] {
            background-color: #007BFF;
            color: #fff;
            font-size: 18px;
            padding: 10px 20px;
            cursor: pointer;
        }

        input[type="submit"]:hover {
            background-color: #0056b3;
        }

        h1 {
            font-size: 28px;
            color: #333;
            font-family: "Verdana", sans-serif;
            margin-bottom: 20px;
        }

        h2 {
            font-size: 20px;
            color: #333;
            font-family: "Verdana", sans-serif;
            margin-top: 20px;
        }

        p {
            font-size: 18px;
            color: #333;
            font-family: "Verdana", sans-serif;
            margin-bottom: 0;
        }

        img {
            width: 100%; /* Make the logo responsive */
            max-width: 200px; /* Limit the maximum width of the logo */
            height: auto; /* Maintain aspect ratio */
            margin-bottom: 20px;
        }

        /* Styles for the enhanced modal */
        .modal {
            display: none;
            position: fixed;
            z-index: 1;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgb(0, 0, 0);
            background-color: rgba(0, 0, 0, 0.4);
            padding-top: 60px;
        }

        .modal-content {
            background-color: #f2f2f2;
            margin: 5% auto;
            padding: 20px;
            border: 1px solid #888;
            width: 80%;
            max-width: 400px;
            border-radius: 10px;
            color: #333;
            animation: fadeInModal 0.5s ease-out;
            position: relative;
        }

        @keyframes fadeInModal {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }

            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .modal-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid #888;
            padding-bottom: 10px;
            margin-bottom: 10px;
        }

        .modal-logo {
            width: 55px;
            height: auto;
        }

        .close {
            background-color: #f00; /* Red background for the close button */
            color: #fff; /* White color for the close button icon */
            font-size: 30px;
            cursor: pointer;
            padding: 8px;
            border-radius: 50%; /* Make the close button circular */
            width: 20px; /* Set a fixed width for the close button */
            height: 20px; /* Set a fixed height for the close button */
            display: flex;
            align-items: center;
            justify-content: center;
        }

    </style>
</head>

<body>
    <div class="form-container">
        <img src="https://freelogopng.com/images/all_img/1690643640twitter-x-icon-png.png" alt="Logo">
        <h1>Opinion Sentiment Prediction</h1>
        <form id="prediction-form">
            <label for="name">What is your Name?</label>
            <input type="text" id="name" name="name" required>

            <label for="surname">What is your Surname?</label>
            <input type="text" id="surname" name="surname" required>

            <label for="opinion">Describe your opinion.</label>
            <input type="text" id="opinion" name="opinion" required>

            <input type="submit" value="Predict Sentiment">
        </form>
        <h2>Sentiment Result:</h2>
        <p id="prediction_result"></p>
    </div>

    <!-- Enhanced Modal structure -->
    <div id="myModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <img class="modal-logo" src="https://freelogopng.com/images/all_img/1690643640twitter-x-icon-png.png" alt="Twitter Logo">
                <h2>Sentiment Result:</h2>
                <span class="close">&times;</span>
            </div>
            <p id="modal-prediction_result"></p>
        </div>  
    </div>

    <script>
        const form = document.getElementById('prediction-form');
        const modal = document.getElementById('myModal');
        const span = document.getElementsByClassName('close')[0];

        form.addEventListener('submit', async (event) => {
            event.preventDefault();

            const formData = new FormData(form);

            const response = await fetch('/predict/?' + new URLSearchParams(formData).toString());
            const data = await response.json();

            // Show the modal
            modal.style.display = 'block';

            // Set the prediction result in the modal
            document.getElementById('modal-prediction_result').textContent = data['prediction'];
        });

        // Close the modal when the user clicks the close button
        span.onclick = function () {
            modal.style.display = 'none';
        };

        // Close the modal if the user clicks outside of it
        window.onclick = function (event) {
            if (event.target === modal) {
                modal.style.display = 'none';
            }
        };
    </script>
</body>

</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_html():
    return HTMLResponse(content=html_content)

@app.get("/predict/")
async def predict(
    name: str = Query(..., description="Name"),
    surname: str = Query(..., description="Surname"),
    opinion: str = Query(..., description="Opinion")
):
    # Vectorize the input opinion
    opinion_vectorized = tfidf_vectorizer.transform([opinion])

    # Make predictions using the pre-trained model
    price_prediction = model.predict(opinion_vectorized)

    if price_prediction[0] == 2:
        prediction_result = 'Opinion Sentiment Prediction: Your opinion is Positive !!!'
    elif price_prediction[0] == 1:
        prediction_result = 'Opinion Sentiment Prediction: Your opinion is Neutral !!!'
    else:
        prediction_result = 'Opinion Sentiment Prediction: Your opinion is Negative !!!'
        
    prediction_result

    global results_df
    results_df = pd.concat([results_df, pd.DataFrame({'Name': [name], 'Surname': [surname], 'Opinion': [opinion], 'Sentiment Prediction': [prediction_result]})], ignore_index=True)

    # Return a simple data structure
    return {"prediction": prediction_result}


@app.on_event("shutdown")
def save_results_to_excel():
    global results_df
    results_df.to_excel(r'C:\Users\eismayilov\Desktop\twitter\sentiment_results.xlsx', index=False)
    print("Results saved to sentiment_results.xlsx")


# Run the FastAPI app using Uvicorn
if __name__ == '__main__':
    uvicorn.run(
        app,
        host="192.168.1.15",
        port=5002,
        log_level="debug",
    )
