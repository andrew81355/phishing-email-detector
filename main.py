from load import load_emails
from preprocessing import clean_email
def main():
    emails = load_emails()
    print(len(emails))
    print(emails["label"].value_counts())
    # Preprocess(after cleaning indexes of emails may changeso we use iloc to to get strs by position in table 0,1,2,3 not indexes)
    emails["clean_text"] = emails["text"].apply(clean_email)
    print(emails["text"].iloc[0][:200])
    print(emails["clean_text"].iloc[0][:200])
main()