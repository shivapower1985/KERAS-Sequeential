###########Problem: Based on the input={passenger class, age, fare} predict output={sex}#########
###########After training the data set with user_input={passenger class, age, fare} predict output={sex} ########
########### KERAS= SEQUENTIAL model is created for this prediction problem ############

###########Created by : SHIVASHANKAR SUKUMAR #############

################ START of the Problem ##############
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

data=pd.read_csv("train_data.csv") #####IMPORTING DATA FILE AS .CSV ##############


##### DROP NON-RELEVANT OR IRRELEVANT COLUMN/FEATURE FROM DATAFRAME
to_drop=['PassengerId','Embarked','SibSp','Survived','Ticket','Parch','Name','Cabin']#feature name or column name to be dropped
data.drop(columns=to_drop, inplace=True)#command to drop the selected column or feature


#CHECK AND FILL NAN OR NO VALUE ROWS UNDER EACH FEATURE
# FILLIN THE MISSING VALUES OR NAN WITH SOME NUMBER
data=data.fillna(method='ffill')#Fill NaN values by forward or backward fill; for backward use 'bfill'
print(data.count()) #command to check all the feature or column count to len(data)


# CHANGING 'male=1' AND 'female=2'
data.loc[data['Sex']=='male','Sex'] = 1
data.loc[data['Sex']=='female','Sex'] = 2


############pop out only the target ###############
target = data.pop('Sex')


############## convert input and target from pandas.DataFrame to NumPy
data=data.to_numpy()          #######Data type for input #### data [Pclass(1,2 and 3) Age(variable) Fare(variable)]
target=target.to_numpy()      #######Data type for output #### Target [male=1 or female=2]


############## Pre-processing the data : Standardize if with StandardScaler ###############
############# Split training and testing data ###############
X_train, X_test, y_train, y_test = train_test_split(data, target)
########################Standardizing the input alone - standardize - input of training and testing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
############## END of Pre-processing the data ###############



############# Build KERAS - sequential model #################
model = Sequential()
model.add(Dense(32, input_dim=3, kernel_initializer='normal', activation='relu'))#12
model.add(Dense(16, activation='relu'))#8
model.add(Dense(16, activation='relu'))#8
model.add(Dense(1, activation='linear'))#linear#1
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae','accuracy'])


############## Train the MODEL #####################
model.fit(X_train, y_train, epochs=150, batch_size=25,  verbose=1, validation_split=0.2)


############## Do prediction from the model: Input - X_test ##############
y_predict= model.predict(X_test)


####### Y_predict will be floating number between 1 and 2 >>>> Converting it to '1' or '2'
for y in range(0,len(y_predict)):
    if y_predict[y] >= 1.5:
        y_predict[y]=2
    else:
        y_predict[y]=1



# Plot outputs - Actual vs predicted
no_elements=range(0,len(y_predict)) # NOTE: plotting the: result (VS) Time, there create number = 1 to length of predicted data

plot_1, =plt.plot(no_elements,y_test, color='red', linewidth=2)
plot_2, =plt.plot(no_elements,y_predict,color='blue', linewidth=2, linestyle='--')
plt.legend([plot_1, plot_2], ["Fare_actual", "Fare_predicted"])
plt.show()

#####END of program#####


###########Print confusion matrix, report and accuracy score##################
results = confusion_matrix(y_test, y_predict)
print('Confusion Matrix :')
print(results)
print('Accuracy Score :',accuracy_score(y_test, y_predict))
print('Report : ' + classification_report(y_test, y_predict))


######### Cross check = Is predicted values same as Actual ????
######### Cross check = If Predict = Actual : Confirm = 1 ########
######### Cross check = If Predict != Actual : Confirm = 0 ########
confirm=[]
for x in range(0,len(y_predict)):
    if y_test[x]==y_predict[x]:
        confirm.append('1')
    else:
        confirm.append('0')

print(confirm.count('1'))
print(confirm.count('0'))
