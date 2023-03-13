# Dakota Kosiorek
# Goal: Find the smallest amount of lead needed to block 1-10 mega electron volts (MeV)

import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn import svm
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score
import warnings
import pickle

mpl.rcParams['figure.figsize'] = (11, 6)
mpl.rcParams['axes.grid'] = False

def main():
    # Material is key, density is value
    materials = pickle.load(open("materials.pickle", "rb"))
    
    print("Training all low/high material models...")
    for material in materials:
        material_density = materials[material]
        
        #print("Gathering data...")
        # To get thickness of material divide MeV cm2/g by the materials density
        #df = pd.read_csv(f"data/{material}.csv", sep="|")
        df = pd.read_csv(f"data/{material}.csv")
        df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    
        df['Kinetic Energy MeV'] = pd.to_numeric(df['Kinetic Energy MeV'])
        #print("Data gathered!\n")
        
        """
        print("Processing data...")
        thickness = list()
        for i in df['CSDA Range g/cm2']:
            thickness.append(to_thickness(mat_range=i, density=material_density))
        df['Thickness (cm)'] = thickness

        print("Data processing complete!\n")
        """
        
        for col in df:
            df[col] = pd.to_numeric(df[col])
        
        #df.to_csv(f"data/{material}.csv", index=False)
        
        y = df['Thickness (cm)']
        X = df.drop('Thickness (cm)', axis=1)
        X = X.drop('Total Stp. Pow. MeV cm2/g', axis=1)
        X = X.drop('CSDA Range g/cm2', axis=1)
        
        # Model for low MeV
        low_model = train_models(X=X[:82], y=y[:82], type="low")
        # Model for high MeV
        high_model = train_models(X=X, y=y, type="high")
        
        model_output_path = os.path.join("models", f"{material}")
        if not os.path.exists(model_output_path):
            os.makedirs(model_output_path)
            
        pickle.dump(low_model, open(os.path.join(model_output_path, "low.pickle"), "wb"))
        pickle.dump(high_model, open(os.path.join(model_output_path, "high.pickle"), "wb"))
        
    print("Training complete!")
    """
    
    """   
def to_thickness(mat_range: float, density: float):
    return mat_range / density

def train_models(X, y, type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1776)
    
    model_pipelines = {
        'lr': make_pipeline(StandardScaler(), LinearRegression()),
        'l': make_pipeline(StandardScaler(), Lasso()),
        'en': make_pipeline(StandardScaler(), ElasticNet()),
        'sgdr': make_pipeline(StandardScaler(), SGDRegressor()),
        'svm': make_pipeline(StandardScaler(), svm.SVR()),
        'krr': make_pipeline(StandardScaler(), KernelRidge()),
        'knn': make_pipeline(StandardScaler(), KNeighborsRegressor()),
        'gpr': make_pipeline(StandardScaler(), GaussianProcessRegressor())
    }
    
    
    # Train models
    #print("Training models...")
    fit_models = {}
    for algorithm, pipeline in model_pipelines.items():
        model = pipeline.fit(X_train, y_train)
        fit_models[algorithm] = model
    #print("Model training complete!\n")
    
    # Get accuracy of each model
    #print("Getting model accuracy...")
    accuracy_models = {}
    for algorithm, model in fit_models.items():
        #yhat = model.predict(X_test)
        accuracy_models[algorithm] = model.score(X_test, y_test)
    #print("Gathered all model accuracy!\n")
    
    #print(f"Algorithm\tAccuracy\t({type} type)")
    #print("-----------------------------")
    for algorithm, accuracy in accuracy_models.items():
        accuracy_models[algorithm] = round(accuracy * 100, 2)
        #print(f"{algorithm}\t\t {round(accuracy * 100, 2)}")
    #print("")
    
    most_accurate_model = max(accuracy_models, key=accuracy_models.get)
    
    return fit_models[most_accurate_model]
    
    
if __name__ == "__main__":
    print("Starting 'radiation_train.py'...\n")
    main()
    print("\n'radiation_train.py' complete!")