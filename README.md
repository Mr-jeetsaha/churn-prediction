# Customer Churn Prediction

This is a small end-to-end project where I predict whether a bank customer is likely to leave the bank (churn) or stay. I trained a neural network on customer data, saved it, and then built a simple web app with Streamlit where anyone can type in a customer's details and get a churn probability.

I built this to practise the full flow of a machine learning project: cleaning data, preparing it, training a model, and then actually using that model in an app instead of leaving it in a notebook.

## About the data

The dataset is `Churn_Modelling.csv`. It has about 10,000 bank customers with details like credit score, country, gender, age, tenure, balance, number of products, whether they have a credit card, whether they are an active member, and estimated salary.

The target column is `Exited`. It is 1 if the customer left the bank and 0 if they stayed. Around 20% of the customers in this data left, so the data is a bit imbalanced.

## What I did, step by step

**1. Looked at the data**
Loaded the CSV with pandas and checked the columns and data types with `head()` and `info()`. This is where I found a few missing values.

**2. Removed columns that don't help**
`RowNumber`, `CustomerId` and `Surname` are just identifiers. They say nothing about whether a person will leave, so I dropped them.

**3. Converted Gender to numbers**
Used `LabelEncoder` so Female and Male become 0 and 1. I saved this encoder as `label_encoder_gender.pkl` so the app can use the exact same mapping later.

**4. Handled missing values**
Four columns had one missing value each:
- `Geography`: filled with "France" (the most common country)
- `Age`: filled with the mean age
- `HasCrCard`: filled with 1
- `IsActiveMember`: filled with 1

**5. One-hot encoded Geography**
Geography has three countries (France, Germany, Spain) with no natural order, so I used `OneHotEncoder` to turn it into three separate 0/1 columns. Saved as `onehot_encoder_geo.pkl`.

**6. Split the data**
80% for training and 20% for testing, with `random_state=42` so the split is repeatable.

**7. Scaled the features**
Used `StandardScaler` so that columns like Balance (big numbers) and Tenure (small numbers) are on a similar scale. The neural network trains much better this way. Saved as `scaler.pkl`.

**8. Built the neural network**
A simple ANN using Keras:
- Dense layer with 64 neurons (ReLU)
- Dense layer with 32 neurons (ReLU)
- Output layer with 1 neuron (sigmoid), which gives a probability between 0 and 1


**9. Trained the model**
- Optimizer: Adam
- Loss: binary cross-entropy
- Up to 100 epochs, but with Early Stopping (patience of 5, monitoring validation loss) so it stops when it is no longer improving and keeps the best weights. Training stopped after 18 epochs.
- Logged the training with TensorBoard so I could look at the loss and accuracy curves.

**10. Saved the model**
Saved as `model.h5`.

**11. Tested predictions in a notebook**
In `prediction.ipynb` I load the model and the three saved files, pass in one sample customer, apply the same encoding and scaling as in training, and print whether the customer is likely to churn.

**12. Built the Streamlit app**
`app.py` loads the model and the saved encoders and scaler, takes the customer details from the user through dropdowns, sliders and input boxes, and shows the churn probability when you click Predict. If the probability is above 0.5 it says the customer is likely to churn, otherwise not likely.

## Project files

```
.
├── app.py                      # Streamlit web app
├── Churn_Modelling.csv         # dataset
├── experiments.ipynb           # data cleaning, preprocessing and model training
├── prediction.ipynb            # testing a single prediction
├── model.h5                    # trained neural network
├── scaler.pkl                  # saved StandardScaler
├── label_encoder_gender.pkl    # saved gender encoder
├── onehot_encoder_geo.pkl      # saved geography encoder
└── requirements.txt            # packages needed
```

## How to run it

1. Clone the repo and go into the folder.

2. (Optional but recommended) create a virtual environment.

3. Install the packages:
   ```
   pip install -r requirements.txt
   ```

4. Start the app:
   ```
   streamlit run app.py
   ```

5. Your browser will open the app. Fill in the customer details and click **Predict**.

If you want to retrain the model, open `experiments.ipynb` and run it from top to bottom. It will overwrite the model and the `.pkl` files. To see the TensorBoard graphs after training:
```
tensorboard --logdir logs/fit
```

## Tech used

Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, TensorBoard, Streamlit



