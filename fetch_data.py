import os
import zipfile
import gdown

PRODUCT_CLASSIFICATION_TRAIN_URL = "https://drive.google.com/u/0/uc?id=1O4YR4UBatOLnaP4gMHbmFw7UJvhhxFwq&export=download"
PRODUCT_CLASSIFICATION_TEST_URL = "https://drive.google.com/u/0/uc?id=1-7aMdKW4KcCKLwoUKC3XxdIwfIKkzwx6&export=download"
SENTIMENT_ANAYLSIS_TRAIN_URL = "https://drive.google.com/u/0/uc?id=1-AlW7oNJHaqi3xk_9dWHUS52Dzl_FmFW&export=download"
SENTIMENT_ANAYLSIS_TEST_URL = "https://drive.google.com/u/0/uc?id=1-8TsrqTRFP-q9TM-6HinhO0ZVXFHq9TB&export=download"

PRODUCT_CLASSIFICATION_DATA_PATH = "data/product_classification"
SENTIMENT_ANALYSIS_DATA_PATH = "data/sentiment_analysis"

def fetch_data(url, path, output):
    """
    Download data from url to path
    Unzip them if required
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
    file_path = os.path.join(path, output)
    
    # check if data already exists
    if os.path.exists(file_path):
        return

    # download data
    gdown.download(url, file_path, quiet=False)    
    
    # unzip data
    if output.endswith(".zip"):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(path)
    
        # remove zip file
        os.remove(file_path)   
    
def fetch_product_classification_data():
    """
    Fetch product classification data
    """
    fetch_data(PRODUCT_CLASSIFICATION_TRAIN_URL, PRODUCT_CLASSIFICATION_DATA_PATH, "train_data.zip")
    fetch_data(PRODUCT_CLASSIFICATION_TEST_URL, PRODUCT_CLASSIFICATION_DATA_PATH, "test_data.zip")
    
def fetch_sentiment_analysis_data():
    """
    Fetch sentiment analysis data
    """
    fetch_data(SENTIMENT_ANAYLSIS_TRAIN_URL, SENTIMENT_ANALYSIS_DATA_PATH, "train_data.csv")
    fetch_data(SENTIMENT_ANAYLSIS_TEST_URL, SENTIMENT_ANALYSIS_DATA_PATH, "test_data.csv")
    
def fetch_all_data():
    """
    Fetch all data
    """
    fetch_product_classification_data()
    fetch_sentiment_analysis_data()
    
if __name__ == "__main__":
    fetch_all_data()