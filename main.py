import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


customers = pd.read_csv("Ecommerce Customers")

customers.head()
customers.describe()
customers.info()

#jointplot to compare the Time on Website and Yearly Amount Spent columns
sns.jointplot(data = customers, x = "Time on Website", y = "Yearly Amount Spent")
plt.savefig("jointplot.png")
plt.close()

#jointplot to compare the Time on App and Yearly Amount Spent columns
sns.jointplot(data = customers, x = "Time on App", y = "Yearly Amount Spent")
plt.savefig("jointplot2.png")
plt.close()

#comparing Time on App and Length of Membership
sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)
plt.savefig("jointplot3.png")
plt.close()

#pairplot for all data
sns.pairplot(customers)
plt.savefig("pairplot.png")
plt.close()

#Length of Membership is the most correlated feature with Yearly Amount Spent

# linear model of Yearly Amount Spent vs. Length of Membership
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)
plt.savefig("linear_model.png")
plt.close()


y = customers["Yearly Amount Spent"]
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

#coefficient off the model
coefficient = lm.coef_
print(coefficient)

predictions = lm.predict(X_test)

#real test values vs the predicted values
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.savefig("result.png")
plt.close()

#Evaluating the Model
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

print(metrics.explained_variance_score(y_test, predictions))

#histogram of the residuals
sns.distplot((y_test-predictions),bins=50)
plt.savefig("residuals.png")
plt.close()

cdf = pd.DataFrame(lm.coef_, X.columns, columns = ["Coeff"])
print(cdf)

#We still want to figure out the answer to the original question,, do we focus our efforst on mobile app or website development?

#interpret these coefficients
'''
Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.'''

'''
Develop the Website to catch up to the performance of the mobile app, or develop the app more since that is what is working better.
!'''
