import spacy
import nltk
from nltk.corpus import stopwords

def main():
    # spaCy model
    spacy.cli.download("en_core_web_sm")  # remove quiet=True

    # NLTK resources
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

    print("Models downloaded successfully.")

if __name__ == "__main__":
    main()
