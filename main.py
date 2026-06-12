from load import load_emails
from preprocessing import clean_email
from features import vectorize
import os
import argparse
from analysis import analysis, load_model_and_vectorizer
def main():
    """ parser for command line arguments in console
    be able to manage
    python main.py --text "text of email to check"
    python main.py --file "path to file with email to check"
    """
    parser = argparse.ArgumentParser(description="check if email is phishing.")
    parser.add_argument("--text",  help="text of email to check")
    parser.add_argument(("--file"), help="path to file with email to check")
    args = parser.parse_args()
    if args.text: #if text just return
        email_text = args.text
    elif args.file: # if file open file and read text
        if os.path.exists(args.file):
            with open(args.file, "r", encoding="utf-8") as f:
                email_text = f.read()
        else:
            print("File not found.")
            return
    try:
        model, vectorizer = load_model_and_vectorizer()
    except FileNotFoundError as e:
        print(e)
        return
    label, confidence = analysis(email_text, model, vectorizer) # works for 1 email text
    print(f"Prediction: {label}, Confidence: {confidence:.4f}")
    
    
    

if __name__ == "__main__":
    main()