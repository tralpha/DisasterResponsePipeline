import sys
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization

# import libraries
import numpy as np
import pandas as pd
import warnings
from sqlalchemy import create_engine

# import statements
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
import os
import warnings

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    filenamedb = database_filepath.split('.')[0]
    df = pd.read_sql_table(filenamedb, engine)
    X = df.message.values
    Y = df.iloc[:, 4:].values
    return X,Y,df

def tokenize(text):
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # Tokenize text
    words = word_tokenize(text)
    
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    
    # Reduce words to their root form
    lemmatized = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    # iterate through each token
    clean_words = []
    for tokens in lemmatized:
        
        # lemmatize, normalize case, and remove leading/trailing white spac
        clean_words.append(tokens.lower().strip())
    return clean_words


def build_model():
    pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
                ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    
    # predict on test data
    y_pred = model.predict(X_test)
    
    accuracy = []

    y_testData = pd.DataFrame(Y_test)
    y_predData = pd.DataFrame(y_pred)

    for col in range(len(y_testData.columns)):
        accuracy.append(accuracy_score(y_testData[col],y_predData[col]))
 		
    target_colums = (category_names.iloc[:,4:].columns).tolist()
	
    acc_score = pd.DataFrame(accuracy,columns=['Accuracy_score'], index=target_colums)
	
    target_names = (category_names.iloc[:,4:].columns).tolist()
	
    output = classification_report(Y_test, y_pred, target_names=target_names)
	
    print("Accuracy for each category:\n")
    print(acc_score)
    
    print("\n\nPrecision,recall, f1-score and support for each category:\n")
    print(output)



def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

def fxn():
    warnings.warn("deprecated", DeprecationWarning)
    
    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        database_filepath = os.path.basename(database_filepath)
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fxn()        
            evaluate_model(model, X_test, Y_test, category_names)

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