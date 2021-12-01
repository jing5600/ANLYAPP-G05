# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 07:49:19 2021

@author: Lenovo
"""


from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# from sklearn.linear_model import Ridge, Lasso, LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor, StackingRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import uuid 
import os


app = Flask(__name__)

# @app.route("/") # Start here
@app.route("/",methods=['GET','POST']) # We need to change the first line to include GET and POST methods

def hello_world():
    request_type_str = request.method
    if request_type_str=='GET':
            return render_template("index.html",href="static/baseimage.svg")
    else:
        text = request.form['text']
        random_string = uuid.uuid4().hex
        path = "app/static/"+random_string +".svg"

        # Load and Create Dataframe
        student_g05 = pd.read_csv("app/student-por.csv",header=0,sep=';')
        student_g05.drop(['Mjob', 'Fjob', 'reason', 'guardian', 'famsize','Pstatus','nursery','activities','higher','internet','Dalc','Walc'], axis=1, inplace=True)
        student_g05 = student_g05.reset_index(drop = True)
        sc = {'GP':0,'MS':1}
        se = {'F':0,'M':1}
        ad = {'R':0,'U':1}
        yn = {'yes':0,'no':1}
# use for loop to convert
        for i in range(student_g05.shape[0]):
            student_g05.school[i] = sc[student_g05.school[i]]
            student_g05.sex[i] = se[student_g05.sex[i]]
            student_g05.address[i] = ad[student_g05.address[i]]
            student_g05.schoolsup[i] = yn[student_g05.schoolsup[i]]
            student_g05.famsup[i] = yn[student_g05.famsup[i]]
            student_g05.paid[i] = yn[student_g05.paid[i]] 
            student_g05.romantic[i] = yn[student_g05.romantic[i]]

        # Split the data frame
        X_train, X_test, y_train, y_test = train_test_split(student_g05.iloc[:,0:-1],
                                        student_g05.iloc[:,-1],
                                        test_size=0.33,
                                        random_state=3158)
        # Load the model with details
        files = [f for f in os.listdir('.') if os.path.isfile(f)]
        for f in files:
          print(f)
        np_arr = floatsome_to_np_array(text).reshape(1, -1)
        pkl_filename="app/TrainedModel/StackedPickle.pkl"
        with open(pkl_filename, 'rb') as file:
            pickle_model = pickle.load(file)
        plot_graphs(model=pickle_model,new_input_arr=np_arr,output_file= path)
        return render_template("index.html",href=path[4:])


def plot_graphs(model,new_input_arr, output_file):
    student_g05 = pd.read_csv("app/student-por.csv",header=0,sep=';')
    student_g05.drop(['Mjob', 'Fjob', 'reason', 'guardian', 'famsize','Pstatus','nursery','activities','higher','internet','Dalc','Walc'], axis=1, inplace=True)
    student_g05 = student_g05.reset_index(drop = True)
    sc = {'GP':0,'MS':1}
    se = {'F':0,'M':1}
    ad = {'R':0,'U':1}
    yn = {'yes':0,'no':1}
# use for loop to convert
    for i in range(student_g05.shape[0]):
        student_g05.school[i] = sc[student_g05.school[i]]
        student_g05.sex[i] = se[student_g05.sex[i]]
        student_g05.address[i] = ad[student_g05.address[i]]
        student_g05.schoolsup[i] = yn[student_g05.schoolsup[i]]
        student_g05.famsup[i] = yn[student_g05.famsup[i]]
        student_g05.paid[i] = yn[student_g05.paid[i]] 
        student_g05.romantic[i] = yn[student_g05.romantic[i]]
  
  
    fig = make_subplots(
    rows=1, cols=2
    # subplot_titles=("Plot 1", "Plot 2", "Plot 3", "Plot 4")
    )

    fig.add_trace(
        go.Scatter(x=student_g05["G1"],y=student_g05['G3'],mode='markers',
        marker=dict(
                color="#003366"),
            line=dict(color="#003366",width=1)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=student_g05['G2'],y=student_g05['G3'],mode='markers',
        marker=dict(
                color="#FF6600"),
            line=dict(color="#FF6600",width=1)),
        row=1, col=2
    )

    new_preds = model.predict(new_input_arr)
    # print(new_preds)
    RM_input = np.array(new_input_arr[0][5])
    # print(RM_input)
    LSTAT_input =np.array(new_input_arr[0][12])
    # print(LSTAT_input)

    fig.add_trace(
    go.Scatter(
        x=LSTAT_input,
        y=new_preds,
        mode='markers', name="Predicted Output",
        marker=dict(
            color="#FFCC00",size=15),
        line=dict(color="#FFCC00",width=1)),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=RM_input,
            y=new_preds,
            mode='markers', name="Predicted Output",
            marker=dict(
                color="#6600cc",size=15),
            line=dict(color="red",width=1)),
            row=1, col=2
    )

    # Update xaxis properties
    fig.update_xaxes(title_text="No.1 Midterm Grade", row=1, col=1)
    fig.update_xaxes(title_text="No.2 Midterm Grade", row=1, col=2)

    # Update yaxis properties
    fig.update_yaxes(title_text="Final Grade", row=1, col=1)
    # fig.update_yaxes(title_text="yaxis 2 title", range=[40, 80], row=1, col=2)
    # Update title and height
    fig.update_layout(height=600, width=1400, title_text="Variation in Final Grade")    
    #output_file="app/static/scatterplot.svg"
    fig.write_image(output_file,width=1200,engine="kaleido")
    fig.show()


def floatsome_to_np_array(floats_str):
  def is_float(s):
    try:
      float(s)
      return True
    except:
      return False
  floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
  return floats.reshape(len(floats), 1)






# example comment made by me














# def hello_world():
#     request_type_str = request.method
#     if request_type_str == 'GET':
#         return render_template('index.html', href='static/base_pic.svg')
#     else:
#         text = request.form['text']
        # random_string = uuid.uuid4().hex
        # path = "static/" + random_string + ".svg"
        # model = load('model.joblib')
        # np_arr = floats_string_to_np_arr(text)
        # make_picture('AgesAndHeights.pkl', model, np_arr, path)
    # return render_template("index.html",href=path)

# This is the first and second step
# def hello_world():
#     return "<p>Hello, World</p>" # First Step
#     # return render_template('index.html', href='static/baseimage.svg') # Comment first step and then uncomment this

# def hello_world():
#     request_type_str = request.method
#     if request_type_str == 'GET':
#         return render_template('index.html', href='static/base_pic.svg')
#     else:
#         text = request.form['text']
        # random_string = uuid.uuid4().hex
        # path = "static/" + random_string + ".svg"
        # model = load('model.joblib')
        # np_arr = floats_string_to_np_arr(text)
        # make_picture('AgesAndHeights.pkl', model, np_arr, path)
    # return render_template("index.html",href=path)

    # boston = load_boston()
    # pkl_filename = "TrainedModel/StackedPickle.pkl"
    # testvalue = boston.data[1].reshape(1, -1)
    # test_input = testvalue
    # with open(pkl_filename, 'rb') as file:
    #     pickle_model = pickle.load(file)
    # predict = pickle_model.predict(test_input)
    # predict_as_str = str(predict)
    # return predict_as_str