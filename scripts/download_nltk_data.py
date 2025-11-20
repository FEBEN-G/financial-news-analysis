"""
Script to download required NLTK data for sentiment analysis.
"""
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_data():
    """Download required NLTK datasets."""
    print("Downloading NLTK data for sentiment analysis...")
    
    datasets = [
        'punkt',           # Tokenizer
        'vader_lexicon',   # Sentiment analysis
        'stopwords',       # Common stop words
        'averaged_perceptron_tagger'  # Part-of-speech tagging
    ]
    
    for dataset in datasets:
        print(f"Downloading {dataset}...")
        try:
            nltk.download(dataset, quiet=True)
            print(f"✅ {dataset}")
        except Exception as e:
            print(f"❌ {dataset}: {e}")
    
    print("\nNLTK data download complete!")

if __name__ == "__main__":
    download_nltk_data()