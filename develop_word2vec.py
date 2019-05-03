import gensim
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import data_util as fe

# define training data
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
data = pd.read_csv('./datasetliar/train.tsv',delimiter='\t',encoding='utf-8',header=None)
# Keeping only the neccessary columns
# data.head()
data2 = pd.read_csv('./datasetliar/test.tsv',delimiter='\t',encoding='utf-8',header=None)
x_train = data[2]
y_train = data[1]
x_test = data2[2]
y_test = data2[1]
X_data = np.concatenate((x_train.values,x_test.values),axis=0)
X_ = pd.DataFrame(X_data)
X = X_[0]
y_data = np.concatenate((y_train.values,y_test.values),axis=0)
y_ = pd.DataFrame(y_data)
y = y_[0]
X = X.apply(lambda x: fe.text_to_wordlist(x,remove_stopwords=True))
X = X.apply(lambda x: x.split(" "))
sentences = list(X.values)

# train model
model = Word2Vec(sentences, min_count=1)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['sentence'])
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)


