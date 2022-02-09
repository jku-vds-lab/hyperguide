
from ipywidgets import Layout, Button, Box, VBox
from IPython.display import display, clear_output, Markdown, HTML

#disable some annoying warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import math
import pandas as pd
import os
import sys
pd.options.mode.chained_assignment = None 
import os.path

import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


class Hyper_Parameter_Provenance(widgets.DOMWidget):
    def __init__(self, X_train, X_test, y_train, y_test, dataset_name):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.dataset_name = dataset_name
        
        self.run = widgets.Button(description='Confirm!', disabled=False, button_style='info',
                                  tooltip='Click me', icon='check')
        
        self.algo_level = 2
        self.guidance_level = 4
        self.param_level = 6

        if not os.path.exists(self.dataset_name):
            os.mkdir(self.dataset_name)
            
    def init(self):
        type_question = widgets.HTML('<h1>What kind of algorithm do you want to use?</h1>')
        
        self.ml_types = [
            Button(description='Classification', layout=Layout(flex='2 1 0%', width='auto')),
            Button(description='Regression', layout=Layout(flex='2 1 0%', width='auto'))
        ]
        for ml_type in self.ml_types:
            ml_type.on_click(self.show_types)
        
        self.box_layout = Layout(display='flex', flex_flow='row', align_items='stretch', width='100%')
        ml_types_box = Box(children=self.ml_types, layout=self.box_layout)
        
        self.container = VBox([type_question, ml_types_box])
        display(self.container)

        
    def show_types(self, button):
        print('clicked', button)
        for btn in self.ml_types:
            btn.style.button_color = 'lightgray'
        button.style.button_color = 'lightblue'
        
        algo_question = widgets.HTML('<h2>Which {} algorithm do you want to run?</h2>'.format(button.description))
        self.container.children = tuple(list(self.container.children)[:self.algo_level] + [algo_question])
        
        algo_box = Box()
 
        if button.description == 'Classification':
            algo_box = self.get_classification_algos()
        elif button.description == 'Regression':
            algo_box = self.get_regression_algos()
            
        self.container.children = tuple(list(self.container.children)[:self.algo_level+1] + [algo_box])

    
    def get_classification_algos(self):
        self.classification_algos = [
            Button(description='Random Forest', layout=Layout(flex='3 1 auto', width='auto')),
            Button(description='knn', layout=Layout(flex='3 1 auto', width='auto')),
            Button(description='SVM', layout=Layout(flex='3 1 auto', width='auto'))
        ]
        for algo in self.classification_algos:
            algo.on_click(self.show_algos)
        return Box(children=self.classification_algos, layout=self.box_layout)
    
    def show_algos(self, button):
        print('clicked', button)
        for btn in self.get_current_algo_btns():
            btn.style.button_color = 'lightgray'
        button.style.button_color = 'lightblue'
        guidance_question = widgets.HTML('<h3>In which setting do you prefer to run {}?</h3>'.format(button.description))
        self.container.children = tuple(list(self.container.children)[:self.guidance_level] + [guidance_question])
                
        self.guidance_types = [Button(description='Default', layout=Layout(flex='3 1 auto', width='auto')),
                                 Button(description='Supported', layout=Layout(flex='3 1 auto', width='auto')),
                                 Button(description='Profi', layout=Layout(flex='3 1 auto', width='auto'))]
        
        guidance_box = Box(children=self.guidance_types, layout=self.box_layout)
        for btn in self.guidance_types:
            btn.on_click(self.show_hyperparamters)
        self.container.children = tuple(list(self.container.children)[:self.guidance_level+1] + [guidance_box])

    
    def get_regression_algos(self):
        self.regression_algos = [
            Button(description='Random Forest', layout=Layout(flex='3 1 auto', width='auto')),
            Button(description='Linear Regression', layout=Layout(flex='3 1 auto', width='auto')),
            Button(description='Logistic Regression', layout=Layout(flex='3 1 auto', width='auto'))
        ]
        for algo in self.regression_algos:
            algo.on_click(self.show_algos)
        return Box(children=self.regression_algos, layout=self.box_layout)
    
    def show_hyperparamters(self, button):
        print('clicked', button)
        for btn in self.guidance_types:
            btn.style.button_color = 'lightgray'
        button.style.button_color = 'lightblue'
        if self.get_active_btn(self.ml_types).description == 'Classification':
            self.show_classification_hyperparams(button)
        else:
            self.show_regression_hyperparams(button)
            
    def show_classification_hyperparams(self, button):
        if self.get_active_btn(self.classification_algos).description == 'Random Forest':
            self.show_rf_classification_hyperparams(button)
        elif self.get_active_btn(self.classification_algos).description == 'knn':
            self.show_knn_hyperparams(button)
        else:
            self.show_svm_params(button)
            
    def show_regression_hyperparams(self, button):
        if self.get_active_btn(self.regression_algos).description == 'Random Forest':
            self.show_rf_regression_hyperparams(button)
        elif self.get_active_btn(self.regression_algos).description == 'Linear Regression':
            self.show_lin_regression_hyperparams(button)
        else:
            self.show_log_regression_params(button)
            
    def show_rf_regression_hyperparams(self, button):
        if button.description == 'Default':
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [widgets.HTML('Default Random Forest for Regression')])
        elif button.description == 'Supported':
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [widgets.HTML('Supported Random Forest for Regression')])
        elif button.description == 'Profi':
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [wiwidgetsdget.HTML('Profi Random Forest for Regression')])
            
        self.container.children = tuple(list(self.container.children)[:self.param_level+1] + [self.run])
            
    def show_rf_classification_hyperparams(self, button):
        if button.description == 'Default':
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [widgets.HTML('Default Random Forest for Classification')])
        elif button.description == 'Supported':
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [widgets.HTML('Supported Random Forest for Classification')])
        elif button.description == 'Profi':
            self.container.children = tuple(list(self.container.children)[:self.param_level] + [widgets.HTML('Profi Random Forest for Classification')])
            
        self.container.children = tuple(list(self.container.children)[:self.param_level+1] + [self.run])

        
    def show_lin_regression_hyperparams(self, button):
        self.container.children = tuple(list(self.container.children)[:self.param_level] + [widgets.HTML('TODO: linear gression')])
        
    def show_log_regression_params(self, button):
        self.container.children = tuple(list(self.container.children)[:self.param_level] + [widgets.HTML('TODO: logistic regression')])
        
    def show_knn_hyperparams(self, button):
        self.container.children = tuple(list(self.container.children)[:self.param_level] + [widgets.HTML('TODO: knn')])
        
    def show_svm_params(self, button):
        self.container.children = tuple(list(self.container.children)[:self.param_level] + [widgets.HTML('TODO: svm')])
        
        
        
    def get_active_btn(self, btn_array):
        return [btn for btn in btn_array if btn.style.button_color == 'lightblue'][0]
    
    def get_current_algo_btns(self):
        return self.classification_algos if self.get_active_btn(self.ml_types).description == 'Classification' \
            else self.regression_algos