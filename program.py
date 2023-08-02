import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = pd.read_csv("Salary.csv")
# drop duplicate rows
data.dropna(inplace=True)
# drop un-nessary columns
data.drop(["Job Title","Country","Race"],axis=1,inplace=True)

# convert object type into integer
o = ["Gender","Education Level"]
o2 = ["Age","Years of Experience","Salary"]
for item in o:
    data[item] = data[item].astype("category")
for item in o2:
    data[item] = data[item].astype("int")

# convert category type into boolean form
gender = pd.get_dummies(data["Gender"],drop_first=True)
education = pd.get_dummies(data["Education Level"],drop_first=True)

Data = pd.concat([data,gender,education],axis=1)
Data.drop(["Gender","Education Level","Other","phD","Bachelor's Degree","High School","Master's","Master's Degree","PhD","Years of Experience"],axis=1,inplace=True)

x = Data.drop("Male",axis=1)
y = Data["Male"]

# splitting the data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.20,random_state=1000)

# train the model
training = LogisticRegression()
training.fit(x_train,y_train)

# test the model
testing = training.predict(x_test)
print(testing)

# checking accuracy of model
print("Accuracy")
print("--"*40)
accuracy = round(accuracy_score(y_test,testing)*100,2)
print(accuracy,"%")