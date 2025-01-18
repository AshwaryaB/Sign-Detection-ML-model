import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

model_data=pickle.load(open('./data.pkl','rb'))

fixed_length = 42
trimmed_data = [item[:fixed_length] for item in model_data['data']]
data = np.array(trimmed_data)

label=np.array(model_data['label'])
x_train, x_test, y_train, y_test= train_test_split(data, label, test_size=0.2,shuffle=True,stratify=label,random_state=42)

#Random Forest
model1=RandomForestClassifier()
model1.fit(x_train,y_train)
y_pred=model1.predict(x_test)
score = accuracy_score(y_pred,y_test)
print('Accuracy of the model1 is {}'.format(score*100))



#Logistic Regression
model2=LogisticRegression(max_iter=1000)
model2.fit(x_train,y_train)
y_pred=model2.predict(x_test)
score = accuracy_score(y_pred,y_test)
print('Accuracy of the model2 is {}'.format(score*100))



#KNN
model3=KNeighborsClassifier(n_neighbors=5)
model3.fit(x_train,y_train)
y_pred=model3.predict(x_test)
score = accuracy_score(y_pred,y_test)
print('Accuracy of the model3 is {}'.format(score*100))

#decision tree
model4=DecisionTreeClassifier()
model4.fit(x_train,y_train)
y_pred=model4.predict(x_test)
score=accuracy_score(y_pred,y_test)
print('Accuracy of the model4 is {}'.format(score*100))



f=open('model.pickle','wb')
pickle.dump({'model1' : model1,'model2' : model2, 'model3' : model3, 'model4': model4},f)
f.close()


'''
Accuracy of the model1 is 100.0
Accuracy of the model2 is 96.42857142857143
Accuracy of the model3 is 100.0
Accuracy of the model4 is 99.57142857142857
Thus Random Forest and K Neighbours are the two best models to be used
'''