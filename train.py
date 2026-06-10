import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing import clean_email
from features import vectorize
from load import load_emails
#create directory for trained model and vectorizer
model_directory = 'models'
model_path = os.path.join(model_directory, 'model.joblib')
vectorizer_path = os.path.join(model_directory, 'vectorizer.joblib')
def evaluate_model(model, X_test_features, y_test): # input trained model
    y_pred = model.predict(X_test_features) # predict labels for test data
    """
    classification report - precision, recall, f1-score for each class
    precision - percentage of correctly predicted, how many of the predicted phishing emails are actually phishing
    recall - how many phishing emails were found
    f1-score balanced measure of precision and recall
    
    support - actual number of phising and legitimate emails in test data
    confusion matrix - shows true positives, true negatives, false positives and false negatives
                   predicted legitimate  predicted phishing
    actual legitimate      tn                 fp
    actual phishing        fn                 tp
    fn is most dangerous because it means that phishing email was not detected and can cause harm to user
    and fp is also dangerous - it means that legitimate email was detected as phishing
    """
    print(classification_report(y_test, y_pred))
    labels = ['legitimate', 'phishing']
    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    print(matrix)
    tn, fp, fn, tp = matrix.ravel()
    print(f'True Positives(phishing caught): {tp}')
    print(f'False Positives(legitimate emails flagged as phishing): {fp}')
    print(f'False Negatives(phishing emails not detected): {fn}')
    print(f'True Negatives(legitimate emails correctly identified): {tn}')

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
    X_train, X_test, y_train, y_test = train_test_split(emails['clean_text'], emails['label'], test_size=0.2, random_state=67, stratify=emails['label'])
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
    evaluate_model(model, x_test_vectorized, y_test)
    # Save the model and vectorizer
    os.makedirs(model_directory, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

if __name__ == "__main__":
    train_model()