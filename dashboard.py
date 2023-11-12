import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import plotly.graph_objects as go
import io
import os
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk

#################################################

##### To run the file:
## 1- install and import necessary libraries
## 2- cd into project directory
## 3- run command : 'streamlit run dashboard.py'

#################################################

st.set_page_config(page_title="Final-Project", page_icon="random", layout="centered")

### Set Title
st.markdown(
    '<div style="text-align: center; font-size: 45px; margin-bottom:30px">Quera Final-Project</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="text-align: center; font-size: 30px; margin-bottom:20px">Welcome to Dayche Final Representation.</div>',
    unsafe_allow_html=True,
)

############################################################  Q2 Start ############################################################

st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align: center; font-size: 25px; margin-bottom:30px">Question 2</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="text-align: center; font-size: 15px; margin-bottom:30px">We are represented with three csv files. Lets explore them and make any necessary changes.</div>',
    unsafe_allow_html=True,
)

############################################################  read all csv files ############################################################


### Seatch and find all csv files
def find_csv_files(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(file)
    return csv_files


if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    csv_files = find_csv_files(current_directory)
for csv in csv_files:
    st.code(csv)

df_train = pd.read_csv("./train_data.csv")
df_test = pd.read_csv("./test_data.csv")
df_title = pd.read_csv("./title_brand.csv")

st.markdown("<hr/>", unsafe_allow_html=True)

############################################################  Preprocessing ############################################################


############################################################  train_data.csv ############################################################

st.write(f"First let's take a look at train_data.csv :")
st.dataframe(df_train.head(10))
st.write(f"Its shape :")
st.write(df_train.shape)


st.write(f"Let's get some information from train_data.csv using info method :")
buffer = io.StringIO()
df_train.info(buf=buffer)
dfTrainInfo = buffer.getvalue()
st.code(dfTrainInfo)

st.write(f"And overall column unique values :")
st.code(df_train["overall"].unique())
st.markdown(
    """
- overall : There are only 5 unique values in overall column so it can be converted to int32 to save memory:
- reviewTime : Needs to be converted to datetime64
- reviewerName, summary and style columns have many NaN values in them. In order to lessen the needed computational power, we can drop these from our dataframe.
- vote : This column is in string format which needs to be converted to numeric. 
"""
)

### Convert to int16 to save memory
df_train["overall"] = df_train["overall"].astype("int16")

### Convert to datetime type
df_train["reviewTime"] = pd.to_datetime(df_train["reviewTime"])
df_test["reviewTime"] = pd.to_datetime(df_test["reviewTime"])
### Drop rows so that data becomes cleaner (This is a minor change.)
df_train = df_train.dropna(subset=["summary", "reviewerName", "style"])

### Convert vote column
### For train
df_train["vote"] = pd.to_numeric(df_train["vote"], errors="coerce", downcast="integer")
df_train["vote"] = df_train["vote"].fillna(pd.NA).astype("Int32")

st.write(f"Now after above changes, let's take another look at train_data.csv :")
buffer = io.StringIO()
df_train.info(buf=buffer)
dfTrainInfo = buffer.getvalue()
st.code(dfTrainInfo)

st.markdown(
    """
- From above code snippet we can clearly see that memory usage is much less now.
- dtype of columns : 'overall, vote, reviewTime' have changed. 
"""
)
st.write(f"Now, let's see the time period of our dataframe :")
st.write(f"From :")
st.code(df_train["reviewTime"].min())
st.write(f"To :")
st.code(df_train["reviewTime"].max())
st.markdown("<hr/>", unsafe_allow_html=True)

############################################################  test_data.csv ############################################################


st.write(f"Now, let's take a look at test_data.csv :")
st.dataframe(df_test.head(10))
st.write(f"Its shape :")
st.write(df_test.shape)

st.write(f"And a deeper look at test_data.csv :")
buffer = io.StringIO()
df_test.info(buf=buffer)
dfTestInfo = buffer.getvalue()
st.code(dfTestInfo)

st.write(f"So, it's basically the same. We just need to change vote, reviewTime dtype.")

df_test["reviewTime"] = pd.to_datetime(df_test["reviewTime"])
### For test
df_test["vote"] = pd.to_numeric(df_test["vote"], errors="coerce", downcast="integer")
df_test["vote"] = df_test["vote"].fillna(pd.NA).astype("Int32")

### Sort by datetime (reviewTime column)
st.write(
    f"If we look closely we can see that both dataframes are not in order of reviewTime.Would be nice if we could sort them in ascending order."
)
df_train.sort_values(by="reviewTime", axis=0, inplace=True, ascending=True)
df_test.sort_values(by="reviewTime", axis=0, inplace=True, ascending=True)

st.write(f"We dropped some NaN rows so let's do a reset_index too.")
### reset index
df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)
st.markdown("<hr/>", unsafe_allow_html=True)

############################################################  title_brand.csv ############################################################
st.write(f"title_brand.csv :")
st.dataframe(df_title.head(10))
st.write(f"Its shape :")
st.write(df_title.shape)

st.write(f"title_brand.csv doesnt need any preprocessing as seen above.")
st.markdown("<hr/>", unsafe_allow_html=True)


############################################################  Q2 Section 1 ############################################################
st.markdown(
    '<div style="text-align: center; font-size: 25px; margin-bottom:30px">Q2 Section 1</div>',
    unsafe_allow_html=True,
)

Q2S1 = """
<div style="direction:rtl; font-size: 25px; background-color: green; color:black; border-radius:20px; padding:10px 15px; margin:10px"> قسمت اول
    <br>
    <p>
        توزیع ستون overall را رسم کنید. آیا مجموعه‌داده متوازن است؟ اگر خیر، آیا نیاز است برای مدل‌سازی خود آن را متوازن کنید؟ چه راه‌حلی برای این کار پیشنهاد می‌کنید؟
    </p>
</div>
"""
st.markdown(Q2S1, unsafe_allow_html=True)

### Plot
overall_values = df_train["overall"].value_counts().sort_index(ascending=False)
fig, ax = plt.subplots()
ax.bar(overall_values.index, overall_values.values)
ax.set_xlabel("Values in column Overall")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Overall")
st.pyplot(fig)

st.markdown(
    """
    - Clearly, value 5 has most frequency over 250,000.
    - Value 4 comes in second with 100,000 frequency.
    - And the rest are around 50,000 frequency. 
    """
)
st.markdown("<hr/>", unsafe_allow_html=True)


############################################################  Q2 Section 2 ############################################################
st.markdown(
    '<div style="text-align: center; font-size: 25px; margin-bottom:30px">Q2 Section 2</div>',
    unsafe_allow_html=True,
)

Q2S2 = """
<div style="direction:rtl; font-size: 25px; background-color: green; color:black; border-radius:20px; padding:10px 15px; margin:10px"> قسمت دوم
    <br>
    <p>
فرض کنید نظراتی که مقدار ستون overall آن‌ها ۴ یا ۵ است را همراه با حس مثبت، نظراتی که مقدارشان ۳ است را خنثی و نظراتی که مقدارشان ۱ یا ۲ است را حس منفی بدانیم. به‌ازای هر کدام از این سه دسته یک ابر کلمات (Word Cloud) رسم کنید تا بتوان کلمات پرتکرار هر دسته را مشاهده کرد. تا حد ممکن سعی کنید ابر کلمات به‌دست‌آمده شامل اطلاعات مفیدی باشد و کلمات زائد (Stop words) بین آن‌ها وجود نداشته باشد. آیا اشتراکی بین کلمات دسته‌ی مثبت و منفی وجود داشته است؟ چگونه آن‌ها را تفسیر می‌کنید؟
 </p>
</div>
"""
st.markdown(Q2S2, unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

st.markdown(
    """
    - Since it takes a while to generate Word Clouds, we only used their images[PNGs] in dashboard.
    - Actual code for generating word clouds in available in jupyter notebook of Question 2.
    """
)

st.image("./dashboard-data/WCloud-noSense.png")
st.image("./dashboard-data/WCloud-negSense.png")
st.image("./dashboard-data/WCloud-posSense.png")

st.markdown(
    """
    - We can clearly see that some words do exist in all three plots.
    - Words like : Even , camera , use[d] , one , device
    - For example from word 'camera' we can see that it might be a common product for our users.
    - By seeing 'device' a lot here, we can infere that many of products can be categorized as electronic devides such as camera itself, phones, laptops etc.
    - Word 'even' also can mean that something was unexpected for the user since it usually indicates a less common point.
    - Word 'use' is used alot since users submit a review usually after using their purchased product after a while. So to metnion time period after its being used.
    
    """
)
st.markdown("<hr/>", unsafe_allow_html=True)

############################################################  Q2 Section 3 ############################################################
st.markdown(
    '<div style="text-align: center; font-size: 25px; margin-bottom:30px">Q2 Section 3</div>',
    unsafe_allow_html=True,
)

Q2S3 = """
<div style="direction:rtl; font-size: 25px; background-color: green; color:black; border-radius:20px; padding:10px 15px; margin:10px"> قسمت سوم
    <br>
    <p>
از بین نظردهندگان، ۱۰ نفری که در مجموع نظرات‌شان بیشتر مفید واقع شده (مجموع vote بیشتری داشته‌اند) را پیدا کنید. نام هر فرد و مجموع vote آن را به‌ترتیب نمایش دهید.
</p>
</div>
"""
st.markdown(Q2S3, unsafe_allow_html=True)

top_10_votes = df_train.sort_values(by="vote", ascending=False)
st.dataframe(top_10_votes[["reviewerName", "vote"]].head(10))
st.markdown(
    """
        - We kept original index in case for later reference.
    """
)

st.markdown("<hr/>", unsafe_allow_html=True)

############################################################  Q2 Section 4 ############################################################
st.markdown(
    '<div style="text-align: center; font-size: 25px; margin-bottom:30px">Q2 Section 4</div>',
    unsafe_allow_html=True,
)

Q2S4 = """
<div style="direction:rtl; font-size: 25px; background-color: green; color:black; border-radius:20px; padding:10px 15px; margin:10px"> قسمت چهارم
    <br>
    <p>
هیستوگرام طول متن (تعداد کاراکتر) ستون reviewText را رسم کنید. یک‌بار با حالت اصلی رسم کنید و یک‌بار به‌صورت فیلترشده (آن دسته‌هایی که تعداد نمونه‌های کم و پرتی دارند را در نظر نگیرید) ترسیم کنید. انتخاب تعداد دسته‌ها (bins) برعهده‌ی خودتان است و نمودار خروجی شما باید مناسب و خوانا باشد. آیا نیاز است در هنگام مدل‌سازی محدودیتی روی تعداد کاراکترها بگذاریم؟ اگر بله، بازه‌ی پیشنهادی شما چه عددهایی است؟
</p>
</div>
"""
st.markdown(Q2S4, unsafe_allow_html=True)

st.markdown(
    '<div style="text-align: center; font-size: 20px; margin-bottom:30px">No Filter</div>',
    unsafe_allow_html=True,
)
st.image('./dashboard-data/hist-noFilter.png')
st.markdown('''
            - After ploting above histogram, we can infer that a limit of 1500 character would be a good choice here.
            - So, we apply 1500 to next plot.
            ''')

st.markdown(
    '<div style="text-align: center; font-size: 20px; margin-bottom:30px">Applied Filter</div>',
    unsafe_allow_html=True,
)
st.image('./dashboard-data/hist-filtered.png')

st.markdown("<hr/>", unsafe_allow_html=True)

############################################################  Q2 Section 5 ############################################################
st.markdown(
    '<div style="text-align: center; font-size: 25px; margin-bottom:30px">Q2 Section 5</div>',
    unsafe_allow_html=True,
)

Q2S5 = """
<div style="direction:rtl; font-size: 25px; background-color: green; color:black; border-radius:20px; padding:10px 15px; margin:10px"> قسمت پنجم
    <br>
    <p>
کدام محصولات بیشترین امتیاز ۵ را کسب کرده‌اند؟ ۱۰ مورد برتر را به‌ترتیب به‌صورت یک جدول شامل نام برند، عنوان محصول و تعداد نظرات با امتیاز ۵ نمایش دهید
</p>
</div>
"""
st.markdown(Q2S5, unsafe_allow_html=True)

st.markdown('''
            - Our team approached this section in two ways : \n
             1- Using title_brand.csv \n
             2- Scraping from Amazon website directly (using selenium and bs4).
            - Results from both methods were the same.
            ''')

### Groupby product string ID
top_10_products = df_train[df_train['overall'] == 5].groupby('asin').size().reset_index(name='count')
### Sort them by their count
top_10_products = top_10_products.sort_values(by='count', ascending=False)
top_10_products = top_10_products[:10].reset_index(drop=True)
### Find both brand and title for each product
### If it exists in df_title.csv
for feature in ['brand', 'title']:
    found_features = []
    for pid in top_10_products['asin']:
        ### If it was found, add its feature
        if pid in df_title['asin'].values:
            found_feature = df_title[df_title['asin'] == pid ][feature].values[0]
            found_features.append(found_feature)
        ### If not found, add NaN instead
        else:
            found_features.append(pd.NA)
    ### Add brands to df
    top_10_products[feature.capitalize()] = found_features
st.dataframe(top_10_products)

st.markdown('''
            - We kept 'asin(product_id)' column for later reference. 
            ''')
st.markdown("<hr/>", unsafe_allow_html=True)

############################################################  Q2 Section 6 ############################################################
st.markdown(
    '<div style="text-align: center; font-size: 25px; margin-bottom:30px">Q2 Section 6</div>',
    unsafe_allow_html=True,
)

Q2S6 = """
<div style="direction:rtl; font-size: 25px; background-color: green; color:black; border-radius:20px; padding:10px 15px; margin:10px"> قسمت ششم
    <br>
    <p>
ابتدا ۱۰ برندی که بیشترین تعداد نظر را داشته‌اند پیدا کنید. سپس میانگین امتیاز هرکدام را محاسبه کرده و یک جدول شامل نام برند و میانگین امتیاز آن به‌ترتیب میانگین امتیاز نمایش دهید.
</p>
</div>
"""
st.markdown(Q2S6, unsafe_allow_html=True)

st.markdown('''
            - Same as previous section, our team approached this section in two ways : \n
             1- Using title_brand.csv \n
             2- Scraping from Amazon website directly (using selenium and bs4).
            - Results from both methods were the same.
            ''')

### First get top 10 products with most reviews
top_10_reviews = df_train.groupby(by='asin').agg({'asin': 'count'}).rename_axis('Product_id').sort_values(by='asin', ascending=False)[:10].reset_index()
top_10_reviews.rename(columns={'asin': 'Total_Reviews'}, inplace=True)
### Then calculate average overall rating for top 10 products with most reviews
means = []
for pid in top_10_reviews['Product_id']:
    average = df_train[df_train['asin'] == pid]['overall'].mean()
    means.append(average)
### Add averages to top_10_reviews
top_10_reviews['Mean_Overall'] = means

### Get only brand this time. (No title needed)
for feature in ['brand']:
    found_features = []
    for pid in top_10_reviews['Product_id']:
        ### If it was found, add its feature
        if pid in df_title['asin'].values:
            found_feature = df_title[df_title['asin'] == pid ][feature].values[0]
            found_features.append(found_feature)
        ### If not found, add NaN instead
        else:
            found_features.append(pd.NA)
    ### Add brands to df
    top_10_reviews[feature.capitalize()] = found_features
### Sort by Mean_Overall
top_10_reviews.sort_values(by='Mean_Overall', ascending=False,inplace=True)
top_10_reviews.reset_index(drop=True, inplace=True)

st.dataframe(top_10_reviews)

st.markdown("<hr/>", unsafe_allow_html=True)

############################################################  Q1 End ############################################################
