#!/usr/bin/env python
# coding: utf-8

# In[103]:


import pandas as pd
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import norm


# In[104]:


class BlackBoxFunction():
    '''A class that will hold attributes for the functions in the capstone
        Aim to have attributes such as X, y, dimensionality, length, a full df holding all data,'''
    
    def __init__(self, name):
        load_path = f"working_data/{name}_data.xlsx"
        self.data = pd.read_excel(load_path, index_col=None)
#         self.data.drop('Unnamed: 0', inplace=True, axis=1)
        self.X = self.data.drop(['Y', 'Type'], axis = 1)
        self.Y = self.data['Y']
        x_dims = self.X.shape[1]
        y_dims = 1
        self.io = (x_dims, y_dims) # dimensions
        self.length = len(self.data)
        self.name = name
        self.mesh_counts = {1: 1000, 2: 1000, 3: 200, 4: 50, 5: 12, 6: 10, 7: 6, 8: 6}
        self.sample_density = self.mesh_counts[self.io[0]]
        self.grid = pd.DataFrame(make_mesh(self.io[0], 0, 0.999999, self.sample_density), columns=self.X.columns)


    
    def add_query(self, quiery, result):
        '''query must be a list, result a number'''
        to_append = []
        for x in quiery:
            to_append.append(x)
        to_append.append(result)
        to_append.append('query')
        to_append = pd.DataFrame(to_append).T
        to_append.columns = self.data.columns
        self.data = self.data.append(to_append, ignore_index=True)
        self.X = self.data.drop(['Y', 'Type'], axis = 1)
        self.Y = self.data['Y']


        
    def export_data_excel(self):
        export_name = f"working_data/{self.name}_data.xlsx"
        self.data.to_excel(export_name, index=False)
    
    def backup_data(self, date, week, backup_number):
        backup_folder = f"data_backups/{date}_Week{week}_backup{backup_number}"
        backup_path = f"{backup_folder}/{self.name}_data.xlsx"
        self.data.to_excel(backup_path)
        
    def fit_GP_model(self, lengthscale, beta=5, noise_assumption=1e-10):
        self.kernel = RBF(length_scale=lengthscale, length_scale_bounds='fixed')
        self.model = GaussianProcessRegressor(kernel=self.kernel, alpha=noise_assumption)
        self.lengthscale = lengthscale
        self.beta = beta
        self.noise_assumption = noise_assumption
        self.model.fit(self.X, self.Y)
    
    def predict(self):
        '''Returns a 2-tuple of UCB-PI predictions'''
        self.post_mean, self.post_std = self.model.predict(self.grid, return_std=True)
        self.UCB_function = self.post_mean + self.beta*self.post_std
        self.PI_function = norm.cdf((self.post_mean-max(self.Y)/self.post_std))
        UCB_prediction = self.grid.iloc[np.argmax(self.UCB_function)].tolist()
        PI_prediction = self.grid.iloc[np.argmax(self.PI_function)].tolist()
        self.UCB_prediction_lst = [round(i, 6) for i in UCB_prediction]
        self.PI_prediction_lst = [round(i, 6) for i in PI_prediction]
        
        self.UCB_prediction_str = self.format_prediction(UCB_prediction)
        self.PI_prediction_str = self.format_prediction(PI_prediction)
        
        self.redundant_UCB = self.redundant(self.UCB_prediction_lst)
        self.redundant_PI = self.redundant(self.PI_prediction_lst)
        
        return (self.UCB_prediction_lst, self.PI_prediction_lst)
    
    
    def redundant(self, query):
        '''input must be a list'''
        return (self.X == np.array(query)).all(1).any()
        
    def format_prediction(self, lst):
        output = []
        for number in lst:
            num_str = str(round(number, 6))
            wl = num_str.split('.')
            if len(wl[1])<6:
                wl[1] = wl[1] + '0'*(6-len(wl[1]))
                num_str = '.'.join(wl)
            output.append(num_str)
        return('-'.join(output))
    
    def localize(self, distance, n_points):
        max_idx = self.Y.idxmax()
        optima = self.X.iloc[max_idx]
        linear_spaces = []
        for optimum in optima:
            lower_range = 0 if optimum - distance < 0 else optimum - distance
            upper_range = 1 if optimum + distance > 1 else optimum + distance
            linear_spaces.append(np.linspace(lower_range, upper_range, n_points))
#         [[np.linspace(optimum-distance, optimum+distance, n_points).tolist()] for optimum in optima]
        meshgrid = np.meshgrid(*linear_spaces)
        meshgrid = [grid.flatten() for grid in meshgrid]
        self.grid = pd.DataFrame(np.transpose(meshgrid), columns = self.X.columns)


# In[105]:


# f1 = BlackBoxFunction('function_1')


# In[106]:


# print(f1.Y.idxmax())
# x = (f1.localize(0.1, f1.sample_density))
# print(x)
# f1.grid


# In[89]:


# meshgrid = ((np.meshgrid(*x)))
# print(meshgrid)
# mygrid = [array.flatten() for array in meshgrid]
# print(pd.DataFrame(np.transpose(mygrid)))


# In[7]:


def import_weekly_queries_results(query_number):
    '''loads the weekly queries and observations into data types readable by my code
    returns 1) queries (list of lists) and 2) observations (list)'''
    
    # initializing lists
    queries_import = []
    results_import = []
    master_list = []
    working_list = []
    
    
    # First load the queries and manipulate to make it readible then convert to float
    load_path = f"queries/{query_number}/queries.txt"
    with open(load_path) as myfile:
        queries_import = myfile.read()
    x = queries_import.replace(' ','').replace('(','').replace(')','').replace('array','').replace('[[','[').replace(']]',']').split('],')
    y = [i.strip().replace('[','').replace(']','').split(',') for i in x]
    for array in y:
        working_list = []
        for item in array:
            working_list.append(float(item))
        master_list.append(working_list)
    queries_import = master_list

    
    # Next load the observations and manipulate to make it readible then convert to float
    load_path = f"queries/{query_number}/observations.txt"
    with open(load_path) as myfile:
        results_import = myfile.read()
    master_list = []
    master_list = [float(item) for item in results_import.replace('[','').replace(']','').replace(' ','').split(',')]
    results_import = master_list
    
    return queries_import, results_import


# In[ ]:


def save_all_data():
    for function in all_bbox_functions:
        function.export_data_excel()


# In[ ]:


def backup_all_data(date, week, backup_number):
        for function in all_bbox_functions:
            function.backup_data(date, week, backup_number)


# In[8]:


def make_mesh(dimensions, lower, upper, count):
    mesh = []
    
    if dimensions == 1:
        mesh = [i for i in np.linspace(lower, upper, count)]
        
    if dimensions == 2:
        for x1 in np.linspace(lower, upper, count):
            for x2 in np.linspace(lower, upper, count):
                mesh.append([x1, x2])

    if dimensions == 3:
        for x1 in np.linspace(lower, upper, count):
            for x2 in np.linspace(lower, upper, count):
                for x3 in np.linspace(lower, upper, count):
                    mesh.append([x1, x2, x3])

    if dimensions == 4:
        for x1 in np.linspace(lower, upper, count):
            for x2 in np.linspace(lower, upper, count):
                for x3 in np.linspace(lower, upper, count):
                    for x4 in np.linspace(lower, upper, count):
                        mesh.append([x1, x2, x3, x4])
                    
    if dimensions == 5:
        for x1 in np.linspace(lower, upper, count):
            for x2 in np.linspace(lower, upper, count):
                for x3 in np.linspace(lower, upper, count):
                    for x4 in np.linspace(lower, upper, count):
                        for x5 in np.linspace(lower, upper, count):
                            mesh.append([x1, x2, x3, x4, x5])                    
                    
                    
    if dimensions == 6:
        for x1 in np.linspace(lower, upper, count):
            for x2 in np.linspace(lower, upper, count):
                for x3 in np.linspace(lower, upper, count):
                    for x4 in np.linspace(lower, upper, count):
                        for x5 in np.linspace(lower, upper, count):
                            for x6 in np.linspace(lower, upper, count):
                                mesh.append([x1, x2, x3, x4, x5, x6]) 
                                
    if dimensions == 7:
        for x1 in np.linspace(lower, upper, count):
            for x2 in np.linspace(lower, upper, count):
                for x3 in np.linspace(lower, upper, count):
                    for x4 in np.linspace(lower, upper, count):
                        for x5 in np.linspace(lower, upper, count):
                            for x6 in np.linspace(lower, upper, count):
                                for x7 in np.linspace(lower, upper, count):
                                    mesh.append([x1, x2, x3, x4, x5, x6, x7]) 
                                    
    if dimensions == 8:
        for x1 in np.linspace(lower, upper, count):
            for x2 in np.linspace(lower, upper, count):
                for x3 in np.linspace(lower, upper, count):
                    for x4 in np.linspace(lower, upper, count):
                        for x5 in np.linspace(lower, upper, count):
                            for x6 in np.linspace(lower, upper, count):
                                for x7 in np.linspace(lower, upper, count):
                                    for x8 in np.linspace(lower, upper, count):
                                        mesh.append([x1, x2, x3, x4, x5, x6, x7, x8]) 
    return mesh


# In[ ]:




