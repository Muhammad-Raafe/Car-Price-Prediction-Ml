import pandas as pd
import numpy as np
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("CarPrice_Assignment.csv")
print(df.isnull().sum())
print(df["fueltype"])

df=pd.get_dummies(df,columns=["fueltype","doornumber","carbody","enginetype"])
print(df.columns)

x=df[['fueltype_diesel','fueltype_gas','doornumber_four','doornumber_two','carbody_convertible','carbody_hardtop','carbody_hatchback','carbody_sedan','carbody_wagon', 'enginetype_dohc', 'enginetype_dohcv', 'enginetype_l',
       'enginetype_ohc', 'enginetype_ohcf', 'enginetype_ohcv', 'enginetype_rotor',"enginesize","curbweight","horsepower"]]

y=df["price"]


x_train,x_test,y_train,y_test=train_test_split(

    x,
    y,
    test_size=0.2,
    random_state=42


)

lr=linear_model.LinearRegression()
lr.fit(x_train,y_train)
prediction=lr.predict(x_test)
result=pd.DataFrame({
    "Actual Price":y_test,
    "Predicted":prediction,
    "Difference":y_test-prediction
})
print(result)
score=lr.score(x_test,y_test)
print("Score Is:",score)


print("Coefficient Value Is: ",lr.coef_)
print("Intercept Value Is: ",lr.intercept_)

# Visualization

sns.scatterplot(x=y_test,y=prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()






