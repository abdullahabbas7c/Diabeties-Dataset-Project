import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data= pd.read_csv("C:/Users/ASUS/Desktop/Python Course 2024/Assignments/archive/diabetes.csv")
print(data)

data.dropna(inplace=True)

X= data.drop(["Outcome"], axis=1)
Y= data["Outcome"]

X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2, random_state=42)

scaler= StandardScaler()
X_train= scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)

lr= LogisticRegression()
lr.fit(X_train, Y_train)
Y_pred_lr= lr.predict(X_test)

knc= KNeighborsClassifier()
knc.fit(X_train, Y_train)
Y_pred_knc= knc.predict(X_test)

rfc= RandomForestClassifier()
rfc.fit(X_train, Y_train)
Y_pred_rfc= rfc.predict(X_test)

feature_importances= rfc.feature_importances_
features= X.columns
indices= np.argsort(feature_importances)

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), feature_importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

def make_pred(input_data):
    input_data= pd.DataFrame([input_data], columns=X.columns)
    input_data= scaler.transform(input_data)
    pred= rfc.predict(input_data)
    if pred == 0:
        print("Non-Diabetic")
    else:
        print("Diabetic")

def user_input():
    print("Enter the following details:")
    preg= float(input("Pregnancies: "))
    glucose= float(input("Glucose: "))
    bp= float(input("Blood Pressure (mm Hg): "))
    skin_thick= float(input("Skin Thickness (mm): "))
    ins= float(input("Insulin (mu U/ml): "))
    bmi= float(input("BMI (weight in kg/(height in m)^2): "))
    diabetes_pedigree= float(input("Diabetes Pedigree Function: "))
    age= float(input("Age: "))
    return[preg, glucose, bp, skin_thick, ins, bmi, diabetes_pedigree, age]

input_data= user_input()
print("Prediction:", end=" ")
make_pred(input_data)

lr_accuracy= accuracy_score(Y_test, Y_pred_lr)
knc_accuracy= accuracy_score(Y_test, Y_pred_knc)
rfc_accuracy= accuracy_score(Y_test, Y_pred_rfc)

mean_accuracy= (lr_accuracy+knc_accuracy+rfc_accuracy)/3
print("Mean Accuracy:", mean_accuracy * 100, "%")
