#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 09:45:11 2023

@author: diech
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.impute import KNNImputer

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier

import statsmodels.api as sm

import time

all_start_time= time.time()

#Se cargan los datos
df  = pd.read_csv('Loan_Default.csv', index_col=0)

df.columns= df.columns.str.lower()

"""
Se quitan las columnas que no determinan a alguien que va a pagar
"""
df = df.drop(['year', 'loan_limit', 'gender', 'approv_in_adv','loan_purpose', 'credit_worthiness',
              'open_credit','business_or_commercial','interest_rate_spread', 'upfront_charges', 'neg_ammortization',
              'interest_only', 'lump_sum_payment','construction_type', 'occupancy_type', 'secured_by', 'total_units',
              'credit_type', 'co-applicant_credit_type','submission_of_application', 'ltv', 'region', 'security_type'],  axis=1)

#variable a predecir
target_var = 'status'

#Se analizan los nan en la base de datos
print(df.info())

"""
Como podemos ver hay muchos valores nan en toda la dataset, al analizar los datos faltantes en variables categoricas,
nos dimos cuenta de que los valores faltantes tienen un motivo, por lo tanto se creara otra categoria en la variables
categoricas.
"""

#lista con las columnas que contienen nan
nan_cols = [i for i  in df.columns if df[i].isnull().sum() > 1]
        
#lista con los nombres de las columnas que tienen nans y son variables categoricas
cols_categoricas_nan = [i for i in nan_cols if df[i].dtype == 'object']

#Se sustituyen los nan con de las columnas categoricas con faltante, de esta forma se agrega una nueva categoria
df[cols_categoricas_nan] = df[cols_categoricas_nan].fillna('faltante')

""" 
Ahora debemos de cambiar las variables categoricas a variables numericas para poder modelar los datos,
para hacer esto convertiremos las variables categoricas a dummy variables.

"""
#Nuevo Dataframe con columnas dummy

print(pd.DataFrame(df.nunique().sort_values(), columns=["Count of unique values"]))

"""
Como podemos ver, las variables que tienen 2 valores nada mas, o en otras palabras son binarias, estas variables se les puede aplicar
One-Hot Encoding

"""

categoricas_cols = [i for i in df.columns if df[i].dtype == 'object']

for i in categoricas_cols :
    if df[i].nunique() == 2:
        
       df[i]=pd.get_dummies(df[i], drop_first=True, prefix=str(i))
       
    elif df[i].nunique() > 2:
        
        df = pd.concat([df.drop([i], axis=1), pd.DataFrame(pd.get_dummies(df[i], drop_first=True, prefix=str(i)))],axis=1)
del i

del categoricas_cols
     
"""
Ahora debemos de encontrar una forma de lidiar con los valores faltantes de variables numericas, como en la mayoria de las
variables numericas la proporcion entre nans y valores reales es realtivamente baja al 50%, se puede aplicar un modelo
para aproximar los valores faltantes, en nuestro caso usaremos el algoritmo kNN para aproximar los valores faltantes.
"""

#lista con los nombres de las columnas son numericas
cols_num = [i for i in df.columns if df[i].dtype != 'object']

# Se declara una instancia de la clase KNNImputer, y se asigna el numero de vecinos a 4
knn = KNNImputer(n_neighbors=4)

#Se ajusta el modelo a los datos
knn.fit(df[cols_num])

start_time = time.time()

#Se obtienen un array con los valores aproximados de los nans de las columnas numericas
X = knn.transform(df[cols_num])

end_time = time.time()
elapsed_time = (end_time - start_time)
print("knn finish, time:", elapsed_time)

#Se convierte el array X a un Dataframe
df_num = pd.DataFrame(X, columns = cols_num )

#lista de los nombres de todas las variables categoricas
categoricas_cols = [i for i in df.columns if df[i].dtype == 'object']

#Se cambian los indices del nuevo dataframe de columnas numericas sin nan para que correspondan con los del dataframe original
df_num.index = df.index

# Se une el dataframe que solo contiene variables categoricas sin nan con el dataframe que solo contiene variables numericas son nan
df =  pd.concat([df[categoricas_cols], df_num ], axis = 1)

#Se combruba que el Dataframe no contiene nans
print(df.info())

#Se limpian las variables que ya no se usaran
del X, df_num, cols_categoricas_nan, cols_num, knn, nan_cols

# Se separan los datos, en entrenamiento y evaluacion
X =  df.drop([target_var], axis=1)
y = df[target_var]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

#Se ensamablan los dataset de entrenamiento y evaluacion
df = pd.concat([X_train, y_train],axis=1)
df_b_s = pd.concat([X_test, y_test],axis=1)

del X_train, X_test, y_train, y_test

"""
Antes de entrenar el modelo se debe de eliminar la colinealidad entre las variables independientes, para esto primero vamos a vizualizar
la matriz de correlacion del Dataframe

"""
# Calcula la matriz de correlación
correlation_matrix = pd.DataFrame(df.corr())

# Crea un mapa de calor con Seaborn, para vizualizar si hay correlacion
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matriz de Correlación')
plt.show()


"""
Ahora vamos a evaluar si tienen colinealidad las variables mediante VIF
"""
df_vif = df.copy()

vif_data = df_vif.drop([target_var], axis=1)
vif_dict = {}

for column in vif_data.columns:
    
    X = vif_data.drop([column], axis=1)
    
    y = df_vif[target_var]
    
    X = sm.add_constant(X)
    # Fit a linear regression model
    model = sm.OLS(y, X).fit()
    
    r_squared = model.rsquared
    if 0.0 < r_squared < 1.0:
        vif = 1 / (1 - r_squared)
    else:
        vif = np.inf  
    vif_dict[column] = vif
del column, vif, model, X, y, r_squared, vif_data

#Se quitan las variables que tengan un VIF mayor a 5
drop_cols_vif = [i for i in vif_dict.keys() if vif_dict[i] > 5]
del vif_dict

#Se quitan en los datos de entranamiento y en los de evaluacion
df_drop = df.drop(drop_cols_vif,axis=1)
df_b_s = df_b_s.drop(drop_cols_vif,axis=1)

del drop_cols_vif

#Se limpian los nombres de las columnas, para que no contengan espacios ni otrso caracteres extraños
df_drop.columns = [col.replace(" ", "_").replace("[", "").replace("]", "").replace("<", "") for col in df_drop.columns]
df_b_s.columns = [col.replace(" ", "_").replace("[", "").replace("]", "").replace("<", "") for col in df_b_s.columns]

"""
Ya que se preprocesaron los datos, debemos de encontrar la forma de balancear la base de datos,
para esto vamos tomar muestras aleatorias de diferentes tamaños de la clase 0 y se van a juntar con toda la clase 1
"""
df_ones = df_drop.loc[df[target_var] == 1]

df_zeros = df_drop.loc[df[target_var] == 0]


def Reequilibrar_con_Diferentes_Proporciones(df, ratios, random_seed=35):
    """
    Parameters:
    - df: DataFrame
        The original DataFrame.
    - ratios: list of float
        A list of values representing the proportions for each split.
        Each value should be in the range [0, 1].
    - random_seed: int, optional
        Seed for the random number generator to ensure reproducibility.

    Returns:
    - splits: list of DataFrames
        A list of DataFrames rebalanced according to the specified ratios.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The first argument must be a DataFrame.")
    
    if not isinstance(ratios, list) or not all(0 <= ratio <= 1 for ratio in ratios):
        raise ValueError("The ratios list should contain values in the range [0, 1].")

    num_samples = len(df)
    num_splits = len(ratios)
    num_samples_splits = [int(num_samples * ratio) for ratio in ratios]

    splits = []
    for num_samples_split in num_samples_splits:
        split = df.sample(n=num_samples_split, replace=True, random_state=random_seed)
        splits.append(split)
    del num_samples_split

    return splits

#Lista con el porcentaje del tamaño de la muestra aleatoria de 0
ratios = [1-(i*0.1) for i in range(9)]

#numero de modelos a crear
n_splits = 3*len(ratios)

# RObtenemos los nuevos dataframes con diferentes balances de clases
df_spliteado = Reequilibrar_con_Diferentes_Proporciones(df_drop, ratios)

#Loop en donde se entran 3 modelos por cada df, xgb, catboost y random forest
lst_models = []
for i in df_spliteado:
    #Se separa el dataset en las variables independientes y dependientes
    df_sub =  pd.concat([df_ones, i ], axis = 0)
    X =  df_sub.drop([target_var], axis=1)
    y = df_sub[target_var]

    #Se instancia los modelo con los hiperparamentros correspondientes
    
    model1 = xgb.XGBClassifier(
        objective="multi:softmax",  
        num_class=len(set(y)),  
        max_depth=10,  
        learning_rate=0.1,  
        n_estimators=100,  
        random_state=56,
        feature_names=list(X.columns)  
    )
    
    model2 = CatBoostClassifier(iterations=1000,
                                depth=10,         
                                learning_rate=0.1,
                                loss_function='Logloss',  
                                )

    model3 = RandomForestClassifier(n_estimators=150, random_state=43456,bootstrap=True)  # Specify feature names
    
    #Se entrena los modelos
    start_time = time.time()
    lst_models.append(model1.fit(X, y))
    end_time = time.time()
    elapsed_time = (end_time - start_time)
    print("mod2 complete, time =", elapsed_time)
    
    start_time = time.time()
    lst_models.append(model2.fit(X, y))
    end_time = time.time()
    elapsed_time = (end_time - start_time)
    print("mod2 complete, time =", elapsed_time)
    
    start_time = time.time()
    lst_models.append(model3.fit(X, y))
    end_time = time.time()
    elapsed_time = (end_time - start_time)
    print("mod2 complete, time =", elapsed_time)
del i, X, y, df_ones, df_zeros, df_sub, model1, model2, model3, df_spliteado, df_drop, df, ratios
print("---------------------------------Resultados--------------------------------")
#Se evaluan lqas predicciones
X_test =  df_b_s.drop([target_var], axis=1)
y_test = df_b_s[target_var]

del df_b_s

#Los modelos votan y se guardan en un array
y_pred = np.zeros((X_test.shape[0],))
for i in lst_models:
    y_pred += i.predict(X_test)
del i

#Se cuentan los 'votos' de los modelos
y_pred = np.where(y_pred < (n_splits // 2), 0, 1)

del n_splits

#Se obtienen las metricas
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

class_report_lines = class_report.split('\n')

# Crear una lista de listas para almacenar las filas de datos
data_rows = [line.split() for line in class_report_lines[2:-5]]  # Ignorar las primeras 2 líneas de encabezado y las últimas 4 líneas de resumen

# Crear un DataFrame a partir de las filas de datos
class_report_df = pd.DataFrame(data_rows, columns=['class', 'precision', 'recall', 'f1-score', 'support'])

print(class_report)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicción 0', 'Predicción 1'],
            yticklabels=['Valor real 0', 'Valor real 1'])
plt.xlabel('Predecido')
plt.ylabel('Valor actual')
plt.title('Matriz de Confusion')
plt.show()

print("Tiempo total:", (time.time() - all_start_time))
del elapsed_time, all_start_time, start_time, end_time, categoricas_cols, class_report, class_report_lines, conf_matrix, correlation_matrix, data_rows,df_vif, target_var


