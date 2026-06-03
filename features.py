from sklearn.feature_extraction.text import TfidfVectorizer
def vectorize():
    """
    we use TfidVectorizer to convert words to numbers for classifier 
    these numbers show how important is word for email
    basically if word is often in specific email and not often in all others it will have high number ->
    ->(this word is important for this email)
    so when classifier will use this words and their numbers with labels it will learn to identify which words
    are identificators of phising emails or legitimate emails

    """
    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
    return vectorizer