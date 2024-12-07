import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

dataset = pd.read_excel("Data.xlsx")
#print(dataset.to_string())
print(dataset.describe())
# 1.duplicates
dataset.drop_duplicates(inplace = True)
#2. null values
# print(dataset.columns)
meanOfQuizzes = dataset["QUIZZES "].mean()
# print(meanOfQuizzes)
dataset.fillna({"QUIZZES ":meanOfQuizzes}, inplace=True)
#3. wrong formats
#4. wrong data
for x in dataset.index:
    if dataset.loc[x, "QUIZZES "] < 0:
        dataset.loc[x, "QUIZZES "] = 0
    
    if dataset.loc[x, "QUIZZES "] > 30:
        dataset.loc[x, "QUIZZES "] = 30

# ML
X=dataset.drop(columns=['SNAMES ','Total Marks','Marks /20', 'Grading '])
y=dataset['Grading ']

X_train, x_test, y_train, y_test=train_test_split(X,y, test_size=0.2)


Decision_tree_model= DecisionTreeClassifier()
Logistic_regression_Model=LogisticRegression(solver='lbfgs',max_iter=10000)
SVM_model=svm.SVC(kernel='linear')
RF_model=RandomForestClassifier(n_estimators=100)

Decision_tree_model.fit(X_train, y_train)
Logistic_regression_Model.fit(X_train, y_train)
SVM_model.fit(X_train, y_train)
RF_model.fit(X_train, y_train)


DT_Prediction =Decision_tree_model.predict(x_test)
LR_Prediction =Logistic_regression_Model.predict(x_test)
SVM_Prediction =SVM_model.predict(x_test)
RF_Prediction =RF_model.predict(x_test)


DT_score=accuracy_score(y_test, DT_Prediction)
lR_score=accuracy_score(y_test, LR_Prediction)
SVM_score=accuracy_score(y_test, SVM_Prediction)
RF_score=accuracy_score(y_test, RF_Prediction)

print ("Decistion Tree accuracy =", DT_score*100,"%")
print ("Logistic Regression accuracy =", lR_score*100,"%")
print ("Suport Vector Machine accuracy =", SVM_score*100,"%")
print ("Random Forest accuracy =", RF_score*100,"%")



predict = Logistic_regression_Model.predict([[20, 27,30,15]])

print(predict)

joblib.dump(Logistic_regression_Model, 'aur.joblib')