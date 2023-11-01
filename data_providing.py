import os
import zipfile
import gdown
import cv2
import numpy as np
import pandas as pd 

PRODUCT_CLASSIFICATION_TRAIN_URL = "https://drive.google.com/u/0/uc?id=1O4YR4UBatOLnaP4gMHbmFw7UJvhhxFwq&export=download"
PRODUCT_CLASSIFICATION_TEST_URL = "https://drive.google.com/u/0/uc?id=1-7aMdKW4KcCKLwoUKC3XxdIwfIKkzwx6&export=download"
SENTIMENT_ANAYLSIS_TRAIN_URL = "https://drive.google.com/u/0/uc?id=1-AlW7oNJHaqi3xk_9dWHUS52Dzl_FmFW&export=download"
SENTIMENT_ANAYLSIS_TEST_URL = "https://drive.google.com/u/0/uc?id=1-8TsrqTRFP-q9TM-6HinhO0ZVXFHq9TB&export=download"
SENTIMENT_ANALYSIS_TITLE_BRAND_URL = "https://drive.google.com/u/0/uc?id=1I9aPAvvYgQWdHGKtnd7IeTGXpx8vOm4h&export=download"

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
    if os.path.exists(file_path) or os.path.exists(file_path.replace('.zip', '')):
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
    fetch_data(SENTIMENT_ANALYSIS_TITLE_BRAND_URL, SENTIMENT_ANALYSIS_DATA_PATH, "title_brand.csv")
    
def fetch_all_data():
    """
    Fetch all data
    """
    fetch_product_classification_data()
    fetch_sentiment_analysis_data()
    
def load_sentiment_analysis_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load sentiment analysis data
    
    return: train_data, test_data, title_brand_data as pandas dataframe
    """
    fetch_sentiment_analysis_data()
    train_data_path = os.path.join(SENTIMENT_ANALYSIS_DATA_PATH, "train_data.csv")
    test_data_path = os.path.join(SENTIMENT_ANALYSIS_DATA_PATH, "test_data.csv")
    title_brand_data_path = os.path.join(SENTIMENT_ANALYSIS_DATA_PATH, "title_brand.csv")
    
    # Load data
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    title_brand_data = pd.read_csv(title_brand_data_path)
    
    return train_data, test_data, title_brand_data


def load_product_classification_data() -> tuple[dict[str, list], dict[str, list]]:
    """
    Load product classification data
    
    return: train_data, test_data as dictionary[images, labels], dictionary[images, file_names]
    """
    fetch_product_classification_data()
    train_data_path = os.path.join(PRODUCT_CLASSIFICATION_DATA_PATH, "train_data")
    test_data_path = os.path.join(PRODUCT_CLASSIFICATION_DATA_PATH, "test_data")
    
    train_data = {'images': [], 'labels': []}
    test_data = {'images': [], 'file_names': []}
    
    # Load train data
    print("Loading train data...")
    for dir_name in os.listdir(train_data_path):
        for file_name in os.listdir(os.path.join(train_data_path, dir_name)):
            train_data['images'].append(cv2.imread(os.path.join(train_data_path, dir_name, file_name)))
            train_data['labels'].append(dir_name)
    
    # Load test data
    print("Loading test data...")
    for file_name in os.listdir(test_data_path):
        test_data['images'].append(cv2.imread(os.path.join(test_data_path, file_name)))
        test_data['file_names'].append(file_name)
            
    return train_data, test_data

if __name__ == "__main__":
    fetch_all_data()