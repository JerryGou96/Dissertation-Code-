from sklearn.svm import SVC
from sklearn import multiclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pickle

def m_svm(X, Y, model_path):

    svm_model=SVC(kernel='linear',verbose=1)
    model = multiclass.OneVsRestClassifier(svm_model)
    scores = cross_val_score(model, X, Y, cv=5)
    pickle.dump(model,open(model_path,'wb'))
    return scores.max()

def m_randomForset(X_train, Y_train, X_test, model_path):
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
    model.fit(X_train, Y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    pickle.dump(model, open(model_path, 'wb'))
    return y_pred_train, y_pred_test

def predict(input, model_path):
    with open(model_path, 'rb') as fr:
        model = pickle.load(fr)
    pre = model.predict(input)
    return pre