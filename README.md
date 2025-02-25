# MLProject
Fake News Data
This notebook provides a step-by-step guide to building a fake news classifier using a Naive Bayes model and PySpark. The pipeline involves data preprocessing, feature extraction, model training, and evaluation. We also include visualizations for better understanding and insight into the dataset and model performance.
1. Load the Dataset
Objective:
Load the dataset from a CSV file stored in Databricks FileStore.
Explanation:
Schema: The dataset has columns id, title, author, text, and label. The schema is explicitly defined for better control over the data types.
File Location: The file is loaded from Databricks FileStore (dbfs:/).
2. Data Preprocessing
Objective:
Remove missing values.
Select relevant columns (title, text, label).
Explanation:
Drop Missing Values: We remove rows that contain any missing (NaN) values in the title, text, or label columns.
Column Selection: Only the title, text, and label columns are selected for training.
3. Define the NLP Pipeline
Objective:
Tokenize the text.
Remove stop words.
Vectorize the text using CountVectorizer and TF-IDF.
Explanation:
Tokenizer: Splits the text into individual words (tokens).
StopWordsRemover: Removes common stop words (e.g., "the", "and").
CountVectorizer: Converts the filtered words into a bag-of-words model.
TF-IDF (IDF): Assigns weights to the word features based on their importance across the entire dataset.
NaiveBayes: Uses Naive Bayes classification to predict the label (fake or real news).
4. Train the Model
Objective:
Train the model using the defined pipeline.
Explanation:
Pipeline Training: The pipeline is fit on the preprocessed data (data), which trains the model by performing all transformations and training steps in one go.
5. Make Predictions
Objective:
Use the trained model to make predictions on the data.
Explanation:
Model Prediction: The trained model is used to predict the labels (real or fake) for the dataset.
Display: Displays the original title, text, and predicted labels (prediction).
6. Evaluate the Model
Objective:
Evaluate the model's performance by calculating its accuracy
Explanation:
MulticlassClassificationEvaluator: This evaluator is used to calculate the accuracy of the model by comparing the predicted labels with the actual labels.
Accuracy: A metric that shows how well the model is performing.
7. Save the Model
Objective:
Save the trained model for future use.
Explanation:
Model Saving: The trained model is saved to Databricks FileStore (dbfs:/) for later use or deployment.
8. Visualizations
Objective:
Generate insightful visualizations for the dataset and model predictions.

Class Distribution Visualization:
Shows the balance between fake (1) and real (0) news in the dataset
Word Cloud Visualization:
Visualizes the most frequent words in the fake news articles.
Model Prediction Accuracy (Bar Chart):
Shows the number of correct vs. incorrect predictions made by the model.
Confusion Matrix Visualization:
Displays the confusion matrix to analyze the true positives, false positives, true negatives, and false negatives.
Model: A Naive Bayes model was trained and evaluated on the WELFake dataset.
Visualizations: Insights into class distribution, frequent words in fake news, and model performance were generated using visualizations.
Model Saved: The trained model is saved for future use or deployment.
