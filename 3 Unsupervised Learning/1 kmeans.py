#
# @author - Cian Cronin (croninc@google.com)
# @description - 1 KMeans Clustering
# @date - 20/10/2018
#

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import helpers2 as h
import tests as t
from IPython import display

# Try instantiating a model with 4 centers
kmeans_4 = KMeans(4)#instantiate your model

# Then fit the model to your data using the fit method
model_4 = kmeans_4.fit(data) #fit the model to your data using kmeans_4

# Finally predict the labels on the same data to show the category that point belongs to
labels_4 = model_4.predict(data)#predict labels using model_4 on your dataset

# function to get kmeans score for scree plotting
def kmeans_get_score(data, centers):
    
    k_means = KMeans(centers)
    model = k_means.fit(data)
    score = np.abs(model.score(data))
    
    return score

scores = []
centers = list(range(1, 11))

for center in centers:
    scores.append(kmeans_get_score(data, center))

plt.plot(centers, scores, linestyle='--', marker='o', color='b');
plt.xlabel('K');
plt.ylabel('SSE');
plt.title('SSE vs. K');