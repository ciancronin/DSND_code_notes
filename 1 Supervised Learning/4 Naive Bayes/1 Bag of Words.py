#
# @author - Cian Cronin (croninc@google.com)
# @description - 1 Bag of Words
# @date - 12/08/2018
#

from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()

documents = ['Hello, how are you!',
                'Win money, win from home.',
                'Call me now.',
                'Hello, Call hello you tomorrow?']

print(count_vector)
count_vector.fit(documents)
count_vector.get_feature_names()

doc_array = count_vector.transform(documents).toarray()
print(doc_array)

frequency_matrix = pd.DataFrame(doc_array, columns=count_vector.get_feature_names())
frequency_matrix

#Example code below from dataset used from here: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
import pandas as pd
df = pd.read_table('smsspamcollection/SMSSpamCollection', names=['label','sms_message'])

df['label'] = df.label.map({'ham':0, 'spam':1})
df.head()

#From scratch bag of words implementation
documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

lower_case_documents = []
for i in documents:
    lower_case_documents.append(i.lower())
print(lower_case_documents)

sans_punctuation_documents = []
import string

for i in lower_case_documents:
    tempString = ""
    for j in i:
        if j not in string.punctuation: tempString += j
    sans_punctuation_documents.append(tempString)
    
print(sans_punctuation_documents)

preprocessed_documents = []
for i in sans_punctuation_documents:
        preprocessed_documents.append(i.split())
print(preprocessed_documents)

frequency_list = []
import pprint
from collections import Counter

for i in preprocessed_documents:
    frequency_list.append(collections.Counter(preprocessed_documents))
    
pprint.pprint(frequency_list)