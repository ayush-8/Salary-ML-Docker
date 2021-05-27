import pandas
dataset = pandas.read_csv("salaryData.csv")
y=dataset['Salary']
x=dataset['YearsExperience']
from sklearn.linear_model import LinearRegression
model = LinearRegression()
x=x.values
x=x.reshape(-1, 1)
model.fit(x,y)
import joblib
joblib.dump(model, 'salary.pk1')
