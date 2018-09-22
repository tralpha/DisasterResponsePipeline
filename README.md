# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Report 

Github link: https://github.com/tralpha/DisasterResponsePipeline

#### ETL Pipeline
The first part of my data pipeline is the Extract, Transform, and Load process. I implemented the first part in the `data/process_data.py` file. This implementation follows perfectly the steps below:
- Loads the messages and categories datasets (`load_data(messages_filepath, categories_filepath): messages, categories`) 
- Cleans the data (`clean_data(df): df`)
- Stores it in a SQLite database (`save_data(df, database_filename)`)

Note that, `disaster_messages.csv` is the `csv` file to load messages dataset, and `disaster_categories.csv` is the `csv` file to load categories dataset.

To test this part of our project, type  `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db` as suggested in the instructions part above. A database will be created. The notebook has been completed and the html file (`ETL Pipeline Preparation.html`) has been generated for this part.


#### ML Pipeline
For the machine learning part, I split the code into a training and test set. Furthermore, I created a machine learning pipeline using NLTK, as well as scikit-learn's Pipeline and GridSearchCV to generate a final model. Moreover, I exported my model to a pickle file. I worked in the notebook that I attached as an html file (`ML Pipeline Preparation.html`). Note that, for our web application I completed the `models/train_classifier.py` file. The steps followed in this files are:

- Loads data from the SQLite database (`load_data(database_filepath):X,Y,df`)
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline (`build_model():pipeline`)
- Trains and tunes a model (`evaluate_model(model, X_test, Y_test, category_names)`)
- Outputs results on the test set
- Exports the final model as a pickle file (`save_model(model, model_filepath)`).


Our model reaches the value 0.74. You can see the pipeline and its improvements in the html file provided. For our `models/train_classifier.py` file, I used the simple model because the precision obtained is good and gives good results.

To test this part, type `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`. A pickle file will be created.


#### Flask Web App
This part uses the first two parts and displays the visualizations (scatter and bar chats) of the results returned by the database. The file paths for accessing the database and the pickle file are changed and the graphs are implemented in the `app/run.py` file. To run the app, type `python run.py` and launch on your browser the http://0.0.0.0:3001/ address.

Queries used:
```
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    genre_countsh = df.groupby('genre').count()['original']
```

Source codes of the graphs used are below

Graph 1:
```
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
```

Graph 2:
```
        {
            'data': [
                {'marker': {'color': 'blue'},
                   'mode': 'markers+lines',
                   'name': 'Original',
                   'text': genre_names,
                   'type': 'scatter',
                   'x': genre_names,
                   'y': genre_countsh},
                {'marker': {'color': 'red'},
                   'mode': 'markers+lines',
                   'name': 'Message',
                   'text': genre_names,
                   'type': 'scatter',
                   'x': genre_names,
                   'y': genre_counts}
            ],

            'layout': {
                'title': 'Distribution of Original and Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
```

Graph 3:
```
        {
            'data': [
                {
                   'type': 'bar',
                   'x': genre_names,
                   'y': genre_counts},
                {'marker': {'color': 'red', 'size': '10'},
                   'mode': 'markers+lines',
                   'text': genre_names,
                   'type': 'scatter',
                   'x': genre_names,
                   'y': genre_counts}
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
```





