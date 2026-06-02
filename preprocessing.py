import re
def clean_email(text):
    text = str(text).lower() # lowercase
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text) # replace urls with "urltoken" for model
    text = re.sub(r"\S+@\S+", " emailtoken ", text) # replace emails with "emailtoken" for model
    text = re.sub(r"<.*?>", " ", text) # remove unnecessary html tahgs
    text = re.sub(r"[^a-z\s]", " ", text) # stay only letters and spaces
    text = re.sub(r"\s+", " ", text).strip() # remove extra spaces
    return text
