#K-Closest Clusters
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import time

def most_common(lst):
      return max(set(lst), key=lst.count)

def train(k, X, y):
  kmeans = KMeans(k, random_state=0).fit(X)
  centers = kmeans.cluster_centers_

  labels = KMeans(k, random_state=0).fit_predict(X)

  #majority voting in each cluster assigns a class to each cluster center
  X = [list(pt) for pt in X]
  y = list(y)
  centers = [tuple(pt) for pt in centers]
  labels = list(labels)
  points_per_cluster = [[y[idx] for idx, num in enumerate(labels) if num == i] for i in range(k)] # for the points in each cluster, identify the acutal class of the points
  center_to_class = {centers[idx]:most_common(lst) for idx,lst in enumerate(points_per_cluster)}
  return [centers, center_to_class]

def classify(point, centers, center_to_class):
  all_dists = []
  for center in centers:
    dist = 0
    for i in range(len(point)):
      dist += (point[i]-center[i])**2
    all_dists.append((dist)**(1/2))
  return center_to_class[centers[all_dists.index(min(all_dists))]]


def test(test_values, test_labels, centers, center_to_class):
   y_true = []
   total_correct = 0
   test_values = [tuple(val) for val in test_values]
   test_labels = list(test_labels)
   for idx, val in enumerate(test_values):
      if (classify(val, centers, center_to_class)) == test_labels[idx]:
        total_correct +=1
      y_true.append(classify(val, centers, center_to_class))
   y_true=np.array(y_true)
   return total_correct/len(test_labels)

def display(k, X, y):
  kmeans = KMeans(k, random_state=0).fit(X)
  centers = kmeans.cluster_centers_

  labels = KMeans(k, random_state=0).fit_predict(X)

  plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
  plt.scatter(centers[:, 0], centers[:, 1], c="red")
  plt.show()
  print("\n")

X1, y1 = datasets.make_blobs(n_samples=[1000,200,200], centers=[[0,0],[50,0],[-50,0]], random_state=3, cluster_std=[20, 5, 5])
X_train, X_valid_and_test, y_train, y_valid_and_test = train_test_split(
    X1, y1, test_size=0.30, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_valid_and_test, y_valid_and_test, test_size=0.50, random_state=42)

#validation to optimize k-value
validation_accuracies = []
models = [()]
for k in range(1,15):
  train_results=train(k, X_train, y_train)
  models.append(train_results)
  validation_accuracy = test(X_valid, y_valid, train_results[0], train_results[1])
  validation_accuracies.append(validation_accuracy)

k = validation_accuracies.index(max(validation_accuracies))+1


display(k, X1, y1)

plt.plot([i for i in range(1, 15)], validation_accuracies,"bx-", linewidth=1)

plt.xlabel('K-Value')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy vs. K-Value')

plt.show()

start_time=time.time()

print("Test Accuracy: " + str(test(X_test, y_test, models[k][0], models[k][1])))

print("Classification time: " + str(time.time()-start_time) + "s")
