import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from preprocessing import clean_email
from features import vectorize
from load import load_emails
#create directory for trained model and vectorizer
model_directory = 'models'
model_path = os.path.join(model_directory, 'model.joblib')
vectorizer_path = os.path.join(model_directory, 'vectorizer.joblib')
def train_model():
    emails = load_emails()
    emails['clean_text'] = emails['text'].apply(clean_email)
    """
    x_train - input data for training the model
    y_train - right answers for the training data
    x_test - input data for testing the model
    y_test - right answers for the testing data
    test_size - the percentage of the data that will be used for testing the model, 80% for training and 20% for testing
    random_state - random number to have same split of data on testing and training
    """
    X_train, X_test, y_train, y_test = train_test_split(emails['clean_text'], emails['label'], test_size=0.2, random_state=67)
    vectorizer = vectorize()
    x_train_vectorized = vectorizer.fit_transform(X_train) # fit - create the vocabulary of the training data and transform - create numerical matrix tf idf
    x_test_vectorized = vectorizer.transform(X_test) # we only transform test data because the data should be new for the model so we use same vocabulary as for training data
    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train_vectorized, y_train)
    train_score = model.score(x_train_vectorized, y_train)
    test_score = model.score(x_test_vectorized, y_test)
    print(f'Training Accuracy: {train_score:.4f}')
    print(f'Testing Accuracy: {test_score:.4f}')
    # Save the model and vectorizer
    os.makedirs(model_directory, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

train_model()