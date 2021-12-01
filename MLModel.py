# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 22:41:35 2021

@author: Lenovo
"""


import random
from seaborn.palettes import color_palette
random.seed(3158)
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split

# Load the dataset
student_g05 = pd.read_csv("app/student-por.csv",header=0,sep=';')

# Display the first six points in the dataset
print(student_g05.head(6))

# drop the irrelevant columns
student_g05.drop(['Mjob', 'Fjob', 'reason', 'guardian', 'famsize','Pstatus','nursery','activities','higher','internet','Dalc','Walc'], axis=1, inplace=True)
# check the dtypes
student_g05.dtypes
student_g05.describe()

# make dict to convert string columns
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
  
  
# make heatmap to show the correlationship
corrMatt_g05 = student_g05[['G1',
                    'G2', 
                    'G3', 
                    'age', 
                    'studytime', 
                    'traveltime', 
                    'freetime']].corr()
# Generate a numpy array
mask_g05 = np.array(corrMatt_g05)
# Turning the lower-triangle of the array to false
mask_g05[np.tril_indices_from(mask_g05)] = False
fig,ax = plt.subplots(figsize = (12,12))
# Create the correlation heat plot of the entire dataset
sns.heatmap(corrMatt_g05, 
            mask=mask_g05,
            vmax=.8, 
            square=True,
            annot=True,
            cmap="PiYG"
            
)




# Change the datatype

student_g05['Medu'] = student_g05.Medu.astype('category')
student_g05['absences'] = student_g05.absences.astype('category')
student_g05['health'] = student_g05.health.astype('category')
student_g05['goout'] = student_g05.goout.astype('category')
student_g05['freetime']=student_g05.freetime.astype('category')
student_g05['famrel']=student_g05.famrel.astype('category')
student_g05['failures']=student_g05.failures.astype('category')
student_g05['studytime']=student_g05.studytime.astype('category')
student_g05['traveltime']=student_g05.traveltime.astype('category')
student_g05['Fedu']=student_g05.Fedu.astype('category')

# Describe the dataset
student_g05.describe()


from matplotlib import rcParams

plt.figure(figsize=(20, 5),dpi=200)

features = ['G1', 'G2']
target = student_g05['G3']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = student_g05[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title("Variation in Final Grade",loc="Left")
    plt.xlabel('No. '+str(i+1) +' Midterm Grade')
    plt.ylabel('Final Grade')

# Firstgraph
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

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

# Update xaxis and yaxis properties
fig.update_xaxes(title_text="No.1 Midterm Grade", row=1, col=1)
fig.update_xaxes(title_text="No.2 Midterm Grade", row=1, col=2)
fig.update_yaxes(title_text="Final Grade", row=1, col=1)

# fig.update_yaxes(title_text="yaxis 2 title", range=[40, 80], row=1, col=2)
# Update title and height
fig.update_layout(height=600, width=1400, title_text="Variation in Final Grade")
output_file="app/static/baseimage.svg"
fig.write_image(output_file,width=1200,engine="kaleido")
fig.show()



##  The second graph

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# split the dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(student_g05.iloc[:,0:-1],
                                        student_g05.iloc[:,-1],
                                        test_size=0.33,
                                        random_state=3158)
# Apply Linear Regression Model as Base Model

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit( X_train, y_train)

print('labels\n', student_g05.columns)
print('Coefficients: \n', lm.coef_)
print('Intercept: \n', lm.intercept_)
print('R2 for Train)', lm.score( X_train, y_train ))
print('R2 for Test (cross validation)', lm.score(X_test, y_test))



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.linear_model import Ridge,Lasso,LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# one-hot encode the categorical features

# define a function to run all models
def all_models_g05():
    models={}
    models['Lasso'] = Lasso()
    models['KNeighbors'] = KNeighborsRegressor(n_neighbors=5)
    models['DecisionTree'] = DecisionTreeRegressor()
    models['RandomForest'] = RandomForestRegressor()
    models['Bagging'] = BaggingRegressor()
    models['Gradient'] = GradientBoostingRegressor()
    return models

# define a function to evaluate all models by cross validation with mae
def eval_scores_g05(model):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = -cross_val_score(model, X_train, y_train,scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')
    return scores
    
# initialize all models
models_g05 = all_models_g05()

# build empty list to store results
results_g05 = []
names_g05 = []

# use for loop to evaluate each model
for name, model in models_g05.items():
    scores_g05 = eval_scores_g05(model)
    results_g05.append(scores_g05)
    names_g05.append(name)
    # print the mean and standard deviation of mae of each model
    print('>%s %.3f (%.3f)' % (name, scores_g05.mean(), scores_g05.std()))


regressmod_g05 = pd.DataFrame(np.transpose(results_g05), columns = 
    ["Lasso","KNeighbors","DecisionTree","RandomForest","Bagging","Gradient"])
regressmod_g05 = pd.melt(regressmod_g05.reset_index(), 
    id_vars='index',value_vars=["Lasso","KNeighbors","DecisionTree","RandomForest","Bagging","Gradient"])

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

fig_g05 = px.box(regressmod_g05, x="variable", y="value",color="variable",points='all',
    labels={"variable": "Machine Learning Model","value": "MSE"},
    title="Model Performance")
fig_g05.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
import warnings
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor



def get_stacking_g05():
	# define the base models
  level0 = list()
  #level0.append(('Tree', DecisionTreeRegressor()))
  level0.append(('RF', RandomForestRegressor()))
  #level0.append(('Bagging', BaggingRegressor()))
  #level0.append(('GBM', GradientBoostingRegressor()))
  #level0 = RandomForestRegressor()
	# define meta learner model
  level1 = LGBMRegressor()
	# define the stacking ensemble
  model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
  return model

def base_models_g05():
  models = dict()
  models["Tree"] = DecisionTreeRegressor()
  models["Random Forest"] = RandomForestRegressor()
  models["Bagging"] = BaggingRegressor()
#   models["XGB"] = XGBRegressor()
  models["Stacked Model"] = get_stacking_g05()
  return models

# Function to evaluate the list of models
def eval_models_g05(model):
  cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
  scores = -cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, 
                            error_score='raise')
  return scores


models_g05 = base_models_g05()
# evaluate the models and store results
results, names = list(), list() 

for name, model in models_g05.items():
  scores = eval_models_g05(model)
  results.append(scores)
  names.append(name)
  print('>%s %.3f (%.3f)' % (name, scores.mean(), scores.std()))

regressmod1_g05 = pd.DataFrame(np.transpose(results), columns = ["Tree","Random Forest","Bagging","Stacked Reg"])
regressmod1_g05 = pd.melt(regressmod1_g05.reset_index(), id_vars='index',value_vars=["Tree","Random Forest","Bagging","Stacked Reg"])
fig = px.box(regressmod1_g05, x="variable", y="value",color="variable",points='all',
labels={"variable": "Machine Learning Model",
        "value": "MSE"
        },title="Model Performance")
fig.show()
# fig.write_image("HW3/Boxplot-candidate.jpeg",engine="kaleido",format="png", width=1600, height=700, scale=0.75)


###  run above

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

level0 = list()
level0.append(('Tree', DecisionTreeRegressor()))
level0.append(('RF', RandomForestRegressor()))
level0.append(('Bagging', BaggingRegressor()))
level0.append(('GBM', GradientBoostingRegressor()))
  #level0 = RandomForestRegressor()
	# define meta learner model
level1 = LGBMRegressor()
	# define the stacking ensemble
model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
model.fit(X_train, y_train)

import pickle

# Save to file in the current working directory
pkl_filename = "app/TrainedModel/StackedPickle.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

# Load from file
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)
    
    
score = pickle_model.score(X_test, y_test)
print("Test score: {0:.2f} %".format(100 * score))
Ypredict = pickle_model.predict(X_test)


np.set_printoptions(suppress=True)



def hello_world():
    
    pkl_filename = "app/TrainedModel/StackedPickle.pkl"
    testvalue = np.array(student_g05.iloc[1,0:-1]).reshape(1,-1)
    test_input = testvalue
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    predict = pickle_model.predict(test_input)
    predict_as_str = str(predict)
    return predict_as_str

hello_world()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

level0 = list()
level0.append(('Tree', DecisionTreeRegressor()))
level0.append(('RF', RandomForestRegressor()))
level0.append(('Bagging', BaggingRegressor()))
level0.append(('GBM', GradientBoostingRegressor()))
  #level0 = RandomForestRegressor()
	# define meta learner model
level1 = LGBMRegressor()
	# define the stacking ensemble
model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
model.fit(X_train, y_train)

def plot_graphs(model,new_input_arr, output_file):
    


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
    output_file="app/static/scatterplot.svg"
    fig.write_image(output_file,width=1200,engine="kaleido")
    fig.show()



testvalue = np.array(student_g05.iloc[1,0:-1]).reshape(1,-1)
plot_graphs(model,new_input_arr=testvalue,output_file="app/static/scatterplot.svg")
