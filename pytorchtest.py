from sklearn import linear_model 

X = [[200,2, 8], [300,3, 10]]

Y = [[4,7], [5,8]]

regr = linear_model.LinearRegression()
regr.fit(X, Y) 


#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = regr.predict([[2300, 1300, 3000]])

print(predictedCO2)