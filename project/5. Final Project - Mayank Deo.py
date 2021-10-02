# Importing the libraries
import pickle
import pandas as pd
import webbrowser
import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import random
from wordcloud import WordCloud, STOPWORDS
from matplotlib import pyplot as plt
import os

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
project_name = "Sentiment Analysis with Insights"

def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")

def load_model():
    global pickle_model
    global vocab
    global scrappedReviews
    scrappedReviews = pd.read_csv('scrappedReviews.csv')
    
    file = open("pickle_model.pkl", 'rb')
    pickle_model = pickle.load(file)
    
    file = open("features.pkl", 'rb')
    vocab = pickle.load(file)
    print('Loading Data......')
    temp = []
    for i in scrappedReviews['Reviews']:
        temp.append(check_review(i)[0])
    scrappedReviews['sentiment'] = temp
    
    positive = len(scrappedReviews[scrappedReviews['sentiment']==1])
    negative = len(scrappedReviews[scrappedReviews['sentiment']==0])
    
    explode = (0.1,0)  

    langs = ['Positive', 'Negative',]
    students = [positive,negative]
    colors = ['#41fc1c','red']
    plt.pie(students,explode=explode,startangle=90,colors=colors, labels = langs,autopct='%1.2f%%')
    cwd = os.getcwd()
    if 'assets' not in os.listdir(cwd):
        os.makedirs(cwd+'/assets')
    plt.savefig('assets/sentiment.png')
    #wordcloud
    dataset = scrappedReviews['Reviews'].to_list()
    str1 = ''
    for i in dataset:
        str1 = str1+i
    str1 = str1.lower()

    stopwords = set(STOPWORDS)
    cloud = WordCloud(width = 800, height = 400,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(str1)
    cloud.to_file("assets/wordCloud2.png")
    
def check_review(reviewText):
    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    reviewText = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))
    return pickle_model.predict(reviewText)

def create_app_ui():
    Random_reviews = []
    for i in range(0,100):
        Random_reviews.append(scrappedReviews['Reviews'][random.randint(0,7639)])

    global project_name
    main_layout = dbc.Container(
        dbc.Jumbotron(
                [
                    html.H1(id = 'heading', children = project_name, className = 'display-3 mb-4'),
                    
                    #Section1
                    html.Img(src=app.get_asset_url('sentiment.png'),style={'width':'700px','height':'400px'}),
                    
                    #Section2
                    html.H3(children='Top 100 most frequently used words'),
                    html.Img(src=app.get_asset_url('wordCloud2.png'),style={'width':'700px','height':'400px'}),
                    
                    #Section3
                    dbc.Container([
                        dcc.Dropdown(
                            id='dropdown',
                            placeholder = 'Select a Review',
                            options=[{'label': i[:100] + "...", 'value': i} for i in Random_reviews],
                            value = scrappedReviews.Reviews[0],
                            style = {'margin-bottom': '30px'}
                                    )
                                 ],
                        style = {'padding-left': '50px', 'padding-right': '50px'}
                                ),
                    html.Div(id = 'result1'),
                    dbc.Button("Submit", color="primary", className="mt-2 mb-3", id = 'Section 3 button', style = {'width': '100px'}),
                    
                    #Section4
                    dbc.Textarea(id = 'textarea', className="mb-3", placeholder="Enter the Review", value = 'My daughter loves these shoes', style = {'height': '150px'}),
                    html.Div(id = 'result'),
                    dbc.Button("Submit", color="primary", className="mt-2 mb-3", id = 'Section 4 button', style = {'width': '100px'}),
                    
                ],
                className = 'text-center'
                ),
        className = 'mt-4'
        )
    
    return main_layout

@app.callback(
    Output('result', 'children'),
    [
    Input('Section 4 button', 'n_clicks')
    ],
    [
    State('textarea', 'value')
    ]
    )    
def update_app_ui(n_clicks, textarea):
    result_list = check_review(textarea)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")

@app.callback(
    Output('result1', 'children'),
    [
    Input('Section 3 button', 'n_clicks')
    ],
    [
     State('dropdown', 'value')
     ]
    )
def update_dropdown(n_clicks, value):
    result_list = check_review(value)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")
    
def main():
    global app
    global project_name
    load_model()
    open_browser()
    app.layout = create_app_ui()
    app.title = project_name
    app.run_server()
    app = None
    project_name = None
if __name__ == '__main__':
    main()
