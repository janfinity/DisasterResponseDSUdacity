#import all relevant libraries
import sys
import pandas as pd
import numpy as np
import os
import re
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



def load_data(database_filepath):
    """
    Function to load data from database

    Arguments:
        database_filepath : path leading to the database

    Output:
        X : dataframe features
        y : dataframe labels
        category_names : category labels
    """
    # load data from database
    engine = create_engine('sqlite:///data/' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","")
    df = pd.read_sql_table(table_name,engine)

    #extract all text/messages values in X
    X = df['message']
    #extract all category values in y
    y = df.iloc[:,3:]
    #save category_names
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    """
    Function to process text data
    - normalize
    - tokenize
    - remove stopwords
    - stem/ lemmatize

    Arguments:
        text : text to be processed

    Output:
        tokens : processed tokens
    """
    #Convert to lowercase
    text = text.lower()
    # Remove punctuation characters
    text = re.sub(r'[^a-zA-Z0-9]',' ',text)
    #Define Term for regular expressions URL
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    #Find all URLs in text and replace them with 'URL'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'URL')
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Reduce words to their stems
    tokens = [PorterStemmer().stem(token) for token in tokens]
    # Reduce words to their root form
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens]
    # Lemmatize verbs by specifying pos
    tokens = [WordNetLemmatizer().lemmatize(token, pos='v') for token in tokens]

    return tokens


def build_pipeline():
    """
    Defining a ML Pipeline
    Defining parameters for optimization
    Use GridSearch to find best parameter setup

    Arguments:
        none

    Output:
        cv : return grisearch model
    """
    # Pipeline: Random Forest Classifier
    pipeline_rfc = Pipeline([
            ('vect', CountVectorizer(tokenizer = tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
    'tfidf__use_idf' : (True, False),
    'clf__estimator__n_estimators': [50, 100, 200]
    }

    cv = GridSearchCV(pipeline_rfc, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluating the ML Model

    Arguments:
        model : ML Model
        X_test : Test features
        y_test : Test labels
        category_names : label names

    Output:
        none
    """
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred, target_names = category_names))
    accuracy = (y_pred == y_test).mean().mean()
    print('Average overall accuracy: {0:.2f}%'.format(accuracy*100))



def save_model(model, model_filepath):
    """
    save model in pickle-file

    Arguments:
        model : ML model
        model_filepath : path to model

    Output:
        none
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_pipeline()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
