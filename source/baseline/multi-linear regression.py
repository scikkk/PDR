from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv("data/processed/0.csv")

X = np.array(df[['gyro_x','gyro_y','gyro_z','acce_x','acce_y','acce_z','linacce_x','linacce_y','linacce_z','magnet_x','magnet_y','magnet_z']])
# Y = np.array(df['Location_dir'])
Y = np.array(df['Location_delta_x'])
# Y = np.array(df['Location_delta_y'])

X_train,X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.1,random_state=100)
linreg = LinearRegression()
model = linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
print(y_pred)
MSE = metrics.mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print(MSE,RMSE)