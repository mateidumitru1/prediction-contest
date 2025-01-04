import spacy
import pandas as pd
from nltk.corpus import stopwords
from tqdm import tqdm

nlp = spacy.load('fr_core_news_sm')
stop_words = set(stopwords.words('french'))

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in stop_words]
    return ' '.join(tokens)

def preprocess_dataframe(df, text_column):
    tqdm.pandas(desc="Preprocessing Text")
    df['cleaned_text'] = df[text_column].progress_apply(preprocess_text)
    return df

def load_and_preprocess_data(train_path, test_path, text_column):
    print("Loading datasets...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print("Preprocessing train data...")
    df_train = preprocess_dataframe(df_train, text_column)

    print("Preprocessing test data...")
    df_test = preprocess_dataframe(df_test, text_column)

    return df_train, df_test
