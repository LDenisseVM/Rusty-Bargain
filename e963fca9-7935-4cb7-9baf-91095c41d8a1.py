#!/usr/bin/env python
# coding: utf-8

# # ¡Hola, Denisse!  
# 
# Mi nombre es Carlos Ortiz, soy code reviewer de TripleTen y voy a revisar el proyecto que acabas de desarrollar.
# 
# Cuando vea un error la primera vez, lo señalaré. Deberás encontrarlo y arreglarlo. La intención es que te prepares para un espacio real de trabajo. En un trabajo, el líder de tu equipo hará lo mismo. Si no puedes solucionar el error, te daré más información en la próxima ocasión. 
# 
# Encontrarás mis comentarios más abajo - **por favor, no los muevas, no los modifiques ni los borres**.
# 
# ¿Cómo lo voy a hacer? Voy a leer detenidamente cada una de las implementaciones que has llevado a cabo para cumplir con lo solicitado. Verás los comentarios de esta forma:
# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Si todo está perfecto.
# </div>
# 
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Si tu código está bien pero se puede mejorar o hay algún detalle que le hace falta.
# </div>
# 
# 
# <div class="alert alert-block alert-danger">
#     
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
#     
# Si de pronto hace falta algo o existe algún problema con tu código o conclusiones.
# </div>
# 
# 
# Puedes responderme de esta forma: 
# 
# 
# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
# </div>
# ¡Empecemos!

# El servicio de venta de autos usados Rusty Bargain está desarrollando una aplicación para atraer nuevos clientes. Gracias a esa app, puedes averiguar rápidamente el valor de mercado de tu coche. Tienes acceso al historial: especificaciones técnicas, versiones de equipamiento y precios. Tienes que crear un modelo que determine el valor de mercado.
# A Rusty Bargain le interesa:
# - la calidad de la predicción;
# - la velocidad de la predicción;
# - el tiempo requerido para el entrenamiento

# ## Preparación de datos

# Importar librerias

# In[130]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import time


# Examinar los datos

# In[131]:


data= pd.read_csv('/datasets/car_data.csv')


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Buen trabajo con la importación de datos y de librerías.
# </div>

# In[132]:


data.info()


# In[133]:


data.head()


# In[134]:


data.describe()


# Vemos valores absurdos, asi que los abordamos 
# 
# Empezamos con los de Registration Year
# Nos quedaremos con los datos que su año de registro es menor a 2024, pues es el año actual, y para evitar valores problematicos, eliminaremos el 1% mas bajo

# In[135]:


data['RegistrationYear'].quantile(.001)


# In[136]:


data['RegistrationYear']= data['RegistrationYear'].mask((data['RegistrationYear'] >= 2024) & (data['RegistrationYear'] <= data['RegistrationYear'].quantile(.001)), np.nan)


# In[137]:


data.sort_values('RegistrationYear').head(10)


# In[138]:


data.info()


# Y hacemos lo mismo con Power

# In[139]:


data['Power'].quantile(.1)


# Vemos que el 10%  muestra valores de 0 cv, lo más probable es que sea un error, asi que probamos con el 15%

# In[140]:


data['Power'].quantile(.15)


# Y vemos que este valor es más razonable, asi que usaremos esta medida, y checamos lo mismo con los máximos

# In[141]:


data['Power'].quantile(.99)


# Asi que eliminaremos ese último 1%

# In[142]:


data['Power']= data['Power'].mask((data['Power'] <= data['Power'].quantile(.15)) & (data['Power'] >= data['Power'].quantile(.99)), np.nan)


# In[143]:


data.info()


# Y lo mismo con Price

# In[144]:


data.sort_values('Price').head(1000)


# In[145]:


data['Price'].quantile(.01)


# Vemos un problema con Price, pues no es posible que vendan un carro con un precio de $0, asi que buscamos un precio más razonable

# Esto parece un poco más razonable, asi que eliminaremos ese 5%

# In[146]:


data = data[(data['Price'] >= data['Price'].quantile(.05))]


# Revisamos la data

# In[147]:


data.info()


# In[148]:


data.describe()


# Y vemos que ahora tenemos datos mucho más coherentes

# Asi que ahora pasaremos a tratar los valores ausentes

# Empezaremos eliminando todas las columnas que no son relevantes para construir nuestro modelo predictor 

# In[149]:


data= data.drop(['DateCrawled', 'RegistrationMonth', 'DateCreated', 'NumberOfPictures', 'PostalCode', 'LastSeen'], axis=1)


# In[150]:


data.info()


# Crearemos una data, sin valores nulos, para poder rellenar lo que podamos con ella 

# In[151]:


data2=data.dropna()
data2.info()


# In[152]:


data[data['VehicleType'].isna()]


# Creamos una función que rellene los valores ausentes con la moda de tipo de vehiculo según el modelo, por lo que eliminaremos las filas dónde hay valor ausente en "Model"

# In[153]:


data= data.dropna(subset=['Model'])


# In[154]:


VehicleType_grouped = data2.groupby('Model')['VehicleType'].agg(lambda x: x.mode()[0])


def fill_vehicle_type(row):
    vehicle_type = row['VehicleType']
    model = row['Model']
    if pd.isna(vehicle_type):
        return VehicleType_grouped.get(model)
    else:
        return vehicle_type

data['VehicleType'] = data.apply(fill_vehicle_type, axis=1)


# In[155]:


data.info()


# In[156]:


data[data['VehicleType'].isna()]


# Hacemos lo mismo con 'FuelType'

# In[157]:


FuelType_grouped = data2.groupby('VehicleType')['FuelType'].agg(lambda x: x.mode()[0])


def fill_fuel_type(row):
    vehicle_type = row['VehicleType']
    fueltype = row['FuelType']
    if pd.isna(fueltype):
        return FuelType_grouped.get(vehicle_type)
    else:
        return fueltype

data['FuelType'] = data.apply(fill_fuel_type, axis=1)


# In[158]:


Gearbox_grouped = data2.groupby('Model')['Gearbox'].agg(lambda x: x.mode()[0])


def fill_gearbox(row):
    Model = row['Model']
    Gearbox = row['Gearbox']
    if pd.isna(Gearbox):
        return Gearbox_grouped.get(Model)
    else:
        return Gearbox

data['Gearbox'] = data.apply(fill_gearbox, axis=1)


# In[159]:


data.info()


# Ahora rellenaremos con "Unknown" todas las demás columnas que también tengan valores ausentes

# In[160]:


data= data.fillna('Unknown')


# El único que tiene un tipo de dato diferente al que debería tener es Registration Year, sin embargo, para nuestro modelo es mejor un tipo de objeto númerico, asi que lo dejaremos así

# In[161]:


data.info()


# In[162]:


data.head()


# Ahora que tenemos listos los datos, simplemente transformaremos los nombres de las columnas a snake_case, y ahora si empezaremos con la creación del modelo

# In[163]:


data.columns = ['Price', 'Vehicle_Type', 'Registration_Year', 'Gearbox', 'Power', 'Model', 'Mileage', 'Fuel_Type', 'Brand', 'Not_Repaired']


# Arreglar valores nulos, valores anormales, tipo de datos

# <div class="alert alert-block alert-success">
#     
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Buen trabajo con las correcciones realizadas.
# </div>

# ## Entrenamiento del modelo 

# Entrena diferentes modelos con varios hiperparámetros (debes hacer al menos dos modelos diferentes, pero más es mejor. Recuerda, varias implementaciones de potenciación del gradiente no cuentan como modelos diferentes). El punto principal de este paso es comparar métodos de potenciación del gradiente con bosque aleatorio, árbol de decisión y regresión lineal.

# Convertiremos los datos a númericos pues es una tarea de regresión, lo haremos con OHE, y con etiqueta para ver cual funciona mejor

# In[164]:


data= data.reset_index(drop=True)


# OHE

# In[165]:


data_ohe= pd.get_dummies(data, drop_first=True)


# In[166]:


data_ohe.head()


# Ordinal Encoder

# In[167]:


data_category= data[['Vehicle_Type', 'Gearbox', 'Model', 'Fuel_Type', 'Brand', 'Not_Repaired']]


# In[168]:


data_numeric= data[['Price', 'Registration_Year', 'Power', 'Mileage']]


# In[169]:


encoder= OrdinalEncoder()


# In[170]:


data_category_ordinal=  pd.DataFrame(encoder.fit_transform(data_category), columns=data_category.columns)


# In[171]:


data_ordinal= data_category_ordinal.join(data_numeric, how='outer')


# In[172]:


data_ordinal.info()


# In[173]:


data_ordinal.head()


# In[174]:


data_ordinal=data_ordinal.astype('int')


# In[175]:


data_ordinal.head()


# In[176]:


data.head()


# Separamos los datos para entrenamiento y prueba

# In[177]:


features_ohe= data_ohe.drop('Price', axis=1)
target= data['Price']
features_train_ohe, features_test_ohe, target_train, target_test= train_test_split(features_ohe, target, test_size=.40, random_state=12345)
features_valid_ohe, features_test_ohe, target_valid, target_test= train_test_split(features_test_ohe, target_test, test_size=.20, random_state=12345)


# In[178]:


features_ordinal= data_ordinal.drop('Price', axis=1)
features_train_ordinal, features_test_ordinal= train_test_split(features_ordinal, test_size=.40, random_state=12345)
features_valid_ordinal, features_test_ordinal= train_test_split(features_test_ordinal, test_size=.20, random_state=12345)


# In[179]:


features= data.drop('Price', axis=1)

features_train, features_test= train_test_split(features, test_size=.40, random_state=12345)
features_valid, features_test= train_test_split(features_test, test_size=.20, random_state=12345)


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Buen trabajo con esta separación de datos.
# </div>

# Usé solo una fracció de los datos al crear el código, pues al ser demasiados, tardaba mucho en correr, lo dejé en markdown por si hay que hacer alguna corrección, pueda olver a ponerlo 

# features_train_ordinal=features_train_ordinal.sample(frac=.1, random_state=12345)
# features_test_ordinal=features_test_ordinal.sample(frac=.1, random_state=12345)
# 
# features_train_ohe=features_train_ohe.sample(frac=.1, random_state=12345)
# features_test_ohe=features_test_ohe.sample(frac=.1, random_state=12345)
# 
# target_train=target_train.sample(frac=.1, random_state=12345)
# target_test=target_test.sample(frac=.1, random_state=12345)
# 
# features_train=features_train.sample(frac=.1, random_state=12345)
# features_test=features_test.sample(frac=.1, random_state=12345)
# 
# features_valid_ohe=features_valid_ohe.sample(frac=.1, random_state=12345)
# 
# features_valid_ordinal=features_valid_ordinal.sample(frac=.1, random_state=12345)
# target_valid=target_valid.sample(frac=.1, random_state=12345)

# In[180]:


model_rf = RandomForestRegressor(random_state=12345, n_estimators=50, max_depth=10)
model_rf.fit(features_train_ordinal, target_train)
predictions_rf = model_rf.predict(features_test_ordinal)
mse = mean_squared_error(target_test, predictions_rf)  # Primero calculamos el MSE
rmse = np.sqrt(mse)
rmse


# In[181]:


model_lr=LinearRegression()
    
model_lr.fit(features_train_ordinal, target_train)
predictions_lr=model_lr.predict(features_test_ordinal)
    

    
RECM= mean_squared_error(target_test, predictions_lr, squared=False)
RECM


# Vemos que el RECM es más alto en el modelo de Regresión Lineal

# In[182]:


params = {
    'objective':'mse',
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Crear el conjunto de datos LightGBM
lgb_train = lgb.Dataset(features_train_ohe, target_train)
lgb_test = lgb.Dataset(features_test_ohe, target_test)

# Entrenar el modelo LightGBM
num_boost_round = 1000
bst = lgb.train(params, lgb_train, num_boost_round, early_stopping_rounds=10, valid_sets=lgb_test)


# In[183]:


params = {
    'objective':'mse',
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Crear el conjunto de datos LightGBM
lgb_train = lgb.Dataset(features_train_ordinal, target_train)
lgb_test = lgb.Dataset(features_test_ordinal, target_test)

# Entrenar el modelo LightGBM
num_boost_round = 1000
bst = lgb.train(params, lgb_train, num_boost_round, early_stopping_rounds=10, valid_sets=lgb_test)


# In[184]:


params = {
    'objective':'mse',
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'learning_rate': 0.1
}

# Crear el conjunto de datos LightGBM
lgb_train = lgb.Dataset(features_train_ordinal, target_train)
lgb_test = lgb.Dataset(features_test_ordinal, target_test)

# Entrenar el modelo LightGBM
num_boost_round = 10000
bst = lgb.train(params, lgb_train, num_boost_round, early_stopping_rounds=50, valid_sets=lgb_test)


# In[185]:


params = {
    'objective':'mse',
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'learning_rate': 0.1
}

# Crear el conjunto de datos LightGBM
lgb_train = lgb.Dataset(features_train_ohe, target_train)
lgb_test = lgb.Dataset(features_test_ohe, target_test)

# Entrenar el modelo LightGBM
num_boost_round = 10000
bst = lgb.train(params, lgb_train, num_boost_round, early_stopping_rounds=50, valid_sets=lgb_test)


# Vemos que el rmse más bajo es para el modelo con descenso de gradiente con las caracteristicas con formato ohe

# ## Análisis del modelo

# Random Forest Regressor:

# Calidad de la predicción:

# In[186]:


predictions_rf_valid=model_rf.predict(features_valid_ordinal)


# In[187]:


rmse_rf= mean_squared_error(target_valid, predictions_rf_valid, squared=False)


# Velocidad de la predicción:

# In[188]:


start_time = time.time()
predictions = model_rf.predict(features_valid_ordinal)
end_time = time.time()

time_pred_rf= end_time - start_time


# Tiempo requerido para el entrenamiento:

# In[189]:


start_time = time.time()
fit = model_rf.fit(features_train_ordinal, target_train)
end_time = time.time()

time_fit_rf= end_time - start_time


# In[190]:


rf_data = ['Random Forest', rmse_rf, time_pred_rf, time_fit_rf]


# Linear Regressor:

# Calidad de la predicción:

# In[191]:


predictions_lr_valid=model_lr.predict(features_valid_ordinal)


# In[192]:


rmse_lr=mean_squared_error(target_valid, predictions_lr_valid, squared=False)


# Velocidad de la predicción:

# In[193]:


start_time = time.time()
predictions = model_lr.predict(features_valid_ordinal)
end_time = time.time()

time_pred_lr= end_time - start_time


# Tiempo requerido para el entrenamiento:

# In[194]:


start_time = time.time()
fit = model_lr.fit(features_train_ordinal, target_train)
end_time = time.time()

time_fit_lr= end_time - start_time


# In[195]:


lr_data=['LinearRegression', rmse_lr, time_pred_lr, time_fit_lr]


# Light GBM:

# Calidad de la predicción:

# In[196]:


bst_valid_predictions = bst.predict(features_valid_ohe, num_iteration=bst.best_iteration)


# In[197]:


rmse_bst=mean_squared_error(target_valid, bst_valid_predictions, squared=False)


# Velocidad de la predicción:

# In[198]:


start_time = time.time()
predictions = bst.predict(features_valid_ohe, num_iteration=bst.best_iteration)
end_time = time.time()

time_pred_bst= end_time - start_time


# Tiempo requerido para el entrenamiento:

# In[201]:


start_time = time.time()
fit = lgb.train(params, lgb_train, num_boost_round, early_stopping_rounds=50, valid_sets=lgb_test)
end_time = time.time()

time_fit_bst= end_time - start_time


# In[202]:


bst_data=['LightGbm', rmse_bst, time_pred_bst, time_fit_bst]


# In[204]:


model_data= [rf_data, lr_data, bst_data]
list_columns= ['Model', 'RMSE', 'Time_pred', 'Time_fit']
models_info= pd.DataFrame(data= model_data, columns= list_columns)


# In[205]:


models_info


# Vemos que el modelo con mayor cálidad en cuanto a predicción es Light GBM, pues es el que tiene el RMSE más bajo, sin embargo también es el que tarda más haciendo predicciones y con el entrenmiento, mientras que Linear Regression es el modelo más rápido, pero con peor cálidad de los 3, pues tiene el RMSE más alto 

# <div class="alert alert-block alert-success">
#     
# 
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# ¡Perfecto! Como se esperaba, el modelo de Boosting tiene mejores resultados.
# </div>

# # Lista de control

# Escribe 'x' para verificar. Luego presiona Shift+Enter

# - [x]  Jupyter Notebook está abierto
# - [ ]  El código no tiene errores- [ ]  Las celdas con el código han sido colocadas en orden de ejecución- [ ]  Los datos han sido descargados y preparados- [ ]  Los modelos han sido entrenados
# - [ ]  Se realizó el análisis de velocidad y calidad de los modelos

# <div class="alert alert-block alert-success">
#     
# # Comentarios generales
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Muy buen trabajo, Denisse. Todo ha sido corregido y has aprobado un nuevo proyecto.
# </div>
