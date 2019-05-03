from gensim.models import Word2Vec
import pandas as pd 
import preproses as pp
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn import svm, linear_model,neural_network
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.feature_selection import SelectKBest, chi2
import proses as p
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold,GridSearchCV
from sklearn.pipeline import Pipeline
import scipy
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

import pickle

def sentence_vector(model, text):
    text_split = text.split(' ')
    matrix = list()
    for t in text_split:
        try:
            matrix.append(model.wv[t])
            print(model.wv[t].shape)
        except:
            print('kata tidak ada di vocab')
            # matrix.append(np.zeros(((100,))))
    return np.average(matrix, axis=0)

def optimisasiClasSVC(x,y,range_c, range_gamma):
    # scoring = {'prec': make_scorer(precision_score),'recall': make_scorer(recall_score),'f1': 'f1_macro','accuracy': make_scorer(accuracy_score)}
    scoring = {'acc': 'accuracy','prec_macro':'precision_macro','rec_micro':'recall_macro'}
    svc = SVC(kernel='rbf')
    pipe = Pipeline([('regressor', svc)])
    kfolds = KFold(n_splits=10)
    parameter = [{'regressor__C': range_c, 'regressor__gamma':range_gamma}]
    grid = GridSearchCV(pipe,param_grid=parameter,scoring=scoring, cv= kfolds, refit='acc')
    grid.fit(x,y)

    best_c = grid.best_params_['regressor__C']
    best_gamma = grid.best_params_['regressor__gamma']

    result = {}
    result['C']= best_c
    result['gamma']=best_gamma
    # result['precission'] = grid.best_score_['prec']
    # result['recall']=grid.best_score_['recall']
    result['acc'] = grid.best_score_
    # result['accuracy']= grid.best_score_['accuracy']
    return result
    
def classRBFSVM(x,y,range_c,gamma,k):
    kf = KFold(n_splits=k)
    scoring = ['precision_macro', 'recall_macro','f1_macro','f1_micro']
    clf = SVC(kernel='rbf',C= range_c, gamma=gamma)
    scores = cross_validate(clf, x, y, scoring=scoring,cv=kf, return_train_score=False)
    return scores
# file = r"stance-data2.csv"

def classreg(x,y):
    # oversampler = SMOTE(kind = "regular")
    # x,y = oversampler.fit_sample(x,y)
    k=10
    kf = KFold(n_splits=k)
    # pca = PCA(n_components = 10)
    scoring = ['precision_macro', 'recall_macro','f1_macro','f1_micro','accuracy','f1_weighted']
    logreg = svm.LinearSVC()
    # pipeline = Pipeline([('pca',pca),('logreg',logreg)])
    scores = cross_validate(logreg, x, y, scoring=scoring,cv=kf, return_train_score=False)
    return scores

df = pd.read_excel('databaru-stance.xlsx')
topik = list(set(list(df['topic'])))


data = {}
hasil_acc ={}
for i in topik:
    data[i] = df[(df['topic'].str.contains(i) )]
    # print(data[i])
    judul = list(data[i]['title'])
    isi = list(data[i]['content'])
    stannce = list(data[i]['stance'])
    y = np.array(stannce)
    doc = []
    for j in range(0,len(judul)):
        doc.append(judul[j]+" "+isi[j])

    model = Word2Vec.load('word2Vec/Word2Vec_artikel/artikel.txt')
    isidata = p.deleteStopwordsandDigit(doc)
    matrix_kalimat = list()
    for d in isidata:
        # print(d)
        vec = sentence_vector(model, d)
        matrix_kalimat.append(vec)

    print(len(matrix_kalimat))
    with open ('pickle_matrix/'+i+'matrix_kalimat.pickle', 'wb') as f_buff : 
        pickle.dump (matrix_kalimat, f_buff)

    # matrix_kalimat = pickle.load (open (i+'matrix_kalimat.pickle', 'rb'))

    
    # x1 = scipy.sparse.csc_matrix(p.useTFngram(doc,1,1))
    x1 = p.useTFngram(doc,1,1)
    x2= np.array(matrix_kalimat)
    # y = np.array(stance)
    x = np.concatenate((x1,x2),axis=1)
    # oversampler = SMOTE(random_state=22,k=3,kind="regular")
    # x,y = oversampler.fit_sample(x,y)
    # x = np.absolute(x)
    # hasil = optimisasiClasSVC(x,y,range_c = np.arange(1,11,1),range_gamma=list(np.logspace(-3,2,15)))
    # hasil_acc[i]=hasil
    # scores = classRBFSVM(x,y,2,0.7196856730011522,10)
    scores = classreg(x,y)
    hasil_acc[i]=scores
    # print(scores['test_f1_micro'].mean())
    # print(hasil)
    # mtd = svm.LinearSVC()
    # # mtd = MultinomialNB()
    # # mtd = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20,), random_state=1)
    # k=10
    # kf = KFold(n_splits=k)
    # sum = 0
    # for train_index, test_index in kf.split(x):
    #     x_train, x_test = x[train_index], x[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #     print(len(y_train[y_train=='for']), len(y_train[y_train=='against']),len(y_train[y_train=='unknown']))
    #     # x_train,y_train = oversampler.fit_sample(x_train,y_train)
    #     # print(len(y_train[y_train=='for']), len(y_train[y_train=='against']),len(y_train[y_train=='unknown']))
    #     mtd.fit(x_train, y_train)
    #     preds = mtd.predict(x_test)
    #     print(confusion_matrix(y_test, preds))
    #     print('\n')
    #     print(classification_report(y_test, preds))
    #     accuracy = accuracy_score(y_test, preds)
    #     print("Accuracy per iterasi %f"%accuracy)
    #     sum += accuracy
        
    # average = sum/k
    # print("============================\n")
    # print("Rata-rata accuracy = %f"%average)

print(hasil_acc)