import os
import pandas as pd

DATA_PATH = os.path.join("CEAS_08.csv")
LABEL_NAMES = {1: "phishing", 0: "legitimate"}

def load_emails(path=DATA_PATH):    
    #read dataset (csv file from Kaggle) 
    data = pd.read_csv(path) #sender,receiver,date,subject,body,label,urls
    data['subject'] = data['subject'].fillna('')
    data['body'] = data['body'].fillna('')
    data['text'] = data['subject'] + ' ' + data['body']
    data['label'] = data['label'].map(LABEL_NAMES) # change 0 to legitimate and 1 to phishing
    emails = data[['text', 'label']]
    return emails
print(load_emails())