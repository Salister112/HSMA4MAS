#Import required libraries
#git location: https://github.com/Salister112/HSMA4MAS

#Load standard modules numpy and pandas
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#import missingno as msno

#library 
#import textwrap as twp

#Import Machine Learning modules
from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler

#Import spreadsheet manipulation modules
#import openpyxl
#from openpyxl.styles import Alignment, Font, Color, colors, PatternFill, Border

#Import file and directory navigation modules
#import glob, os


#----------------------------------------------------

#Load data
#data = pd.read_csv('processed_data/processed_data_1B_DNA_Profile.csv')
data = pd.read_csv('processed_data/processed_data_1B_DNA_Profile_manually_fixed.csv')
# Make all data 'float' type

#data = data.drop('appt_date_time',1)

data = data.astype(float)

#data.describe()

"""
Looking at a summary of patients who attended or did not attend
Before running ML models, good to look at your data. 
Here we will separate patients who attended from those who DNA'd, and we 
will have a look at differences in features.
We will use a mask to select and filter passengers.
"""

mask = data['encoded_Attended'] == 1 # Mask for patients who attended
attended = data[mask] # filter using mask

mask = data['encoded_Attended'] == 0 # Mask for patients who DNA'd
dna = data[mask] # filter using mask

#Now let's look at average (mean) values for attended and dna.

attended.mean()
dna.mean()


#We can make looking at them side by side more easy by putting these values in a new DataFrame
summary = pd.DataFrame() # New empty DataFrame
summary['attended'] = attended.mean()
summary['dna'] = dna.mean()
summary

"""
Divide into X (features) and y (labels)
We will separate out our features (the data we use to make a prediction) 
from our label (what we are truing to predict). 
By convention our features are called X (usually upper case to denote 
multiple features), and the label (survived or not) y.
"""

X = data.drop('encoded_Attended',axis=1) # X = all 'data' except the 'encoded_attended' column
y = data['encoded_Attended'] # y = 'encoded_attended' column from 'data'

"""
Divide into training and tets sets
When we test a machine learning model we should always test it on data that 
has not been used to train the model. We will use sklearn's train_test_split 
method to randomly split the data: 75% for training, and 25% for testing.
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

X_train.std(), X_train.mean()

"""
Standardise data
We want all of out features to be on roughly the same scale. This generally 
leads to a better model, and also allows us to more easily compare the 
importance of different features.
One simple method is to scale all features 0-1 (by subtracting the minimum 
value for each value, and dividing by the new remaining maximum value).
But a more common method used in many machine learning methods is 
standardisation, where we use the mean and standard deviation of the training 
set of data to normalise the data. We subtract the mean of the test set values, 
and divide by the standard deviation of the training data. Note that the mean 
and standard deviation of the training data are used to standardise the test 
set data as well.
Here we will use sklearn's StandardScaler method. This method also copes with 
problems we might otherwise have (such as if one feature has zero standard 
deviation in the training set).
"""

X_train.astype(float)

def standardise_data(X_train, X_test):
    
    # Initialise a new scaling object for normalising input data
    sc = StandardScaler() 

    # Set up the scaler just on the training set
    sc.fit(X_train)

    # Apply the scaler to the training and test sets
    train_std=sc.transform(X_train)
    test_std=sc.transform(X_test)
    
    return train_std, test_std

X_train_std, X_test_std = standardise_data(X_train, X_test)

"""
Fit logistic regression model
Now we will fir a logistic regression model, using sklearn's 
LogisticRegression method. Our machine learning model fitting is only two 
lines of code! By using the name model for our logistic regression model we 
will make our model more interchangeable later on.
"""

model = LogisticRegression()
model.fit(X_train_std,y_train)

"""
Predict values
Now we can use the trained model to predict attendance. 
We will test the accuracy of both the training and test data sets.
"""

# Predict training and test set labels
y_pred_train = model.predict(X_train_std)
y_pred_test = model.predict(X_test_std)

"""
Calculate accuracy
In this example we will measure accuracy simply as the proportion of 
patients where we make the correct prediction. 
Need to consider other measures of accuracy which explore false positives and 
false negatives in more detail.
"""

accuracy_train = np.mean(y_pred_train == y_train)
accuracy_test = np.mean(y_pred_test == y_test)

print ('Accuracy of predicting training data =', accuracy_train)
print ('Accuracy of predicting test data =', accuracy_test)

"""
Examining the model coefficients (weights)
Not all features are equally important. And some may be of little or no use 
at all, unnecessarily increasing the complexity of the model. In a later 
notebook we will look at selecting features which add value to the model 
(or removing features that don’t).
Here we will look at the importance of features – how they affect our 
estimation of attendance. These are known as the model coefficients (if you 
come from a traditional statistics background), or model weights (if you come 
from a machine learning background).
Because we have standardised our input data the magnitude of the weights 
may be compared as an indicator of their influence in the model. Weights with 
higher negative numbers mean that that feature correlates with reduced chance 
of attendance. Weights with higher positive numbers mean that that feature 
correlates with increased chance of attendance. Those weights with values closer 
to zero (either positive or negative) have less influence in the model.
We access the model weights my examining the model coef_ attribute. The model 
may predict more than one outcome label, in which case we have weights for 
each label. Because we are predicting a signle label (survive or not), the 
weights are found in the first element ([0]) of the coef_ attribute.
"""

co_eff = model.coef_[0]

co_eff_df = pd.DataFrame() # create empty DataFrame
co_eff_df['feature'] = list(X) # Get feature names from X
co_eff_df['co_eff'] = co_eff
co_eff_df['abs_co_eff'] = np.abs(co_eff)
co_eff_df.sort_values(by='abs_co_eff', ascending=False, inplace=True)

co_eff_df

"""
Show predicted probabilities
The predicted probabilities are for the two alternative classes 0 (does not survive) or 1 (survive).
Ordinarily we do not see these probabilities - the predict method used above applies a cut-off of 0.5 to classify passengers into survived or not, but we can see the individual probabilities for each passenger.
Later we will use these to adjust sensitivity of our model to detecting survivors or non-survivors.
Each passenger has two values. These are the probability of not surviving (first value) or surviving (second value). Because we only have two possible classes we only need to look at one. Multiple values are important when there are more than one class being predicted.
"""

# Show first ten predicted classes
classes = model.predict(X_test_std)
classes[0:10]

# Show first ten predicted probabilities 
# (note how the values relate to the classes predicted above)
probabilities = model.predict_proba(X_test_std)
print(probabilities[0:10])
