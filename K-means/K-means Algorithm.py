# This program accesses data from a cvs file and interprets the data
# in the file using the K-mean clustering model
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# function read_file opens file and obtained values from columns so it because unlabeled data
# euclidean_dist calculates distance between centroids(euclidean distance formula) and values/points stores them
# in a list and returns the list.


def read_file(filename):
    data = pd.read_csv(filename)
    print(data)
    country_names = data[data.columns[0]].values
    values_arr = data[[
        data.columns[1], data.columns[2]]].values
    return values_arr, country_names


def euclidean_dist(centroids_, data_points):
    calculated_dist = []  # to store calculated distances
    for centroid in centroids_:
        for point in data_points:
            calculated_dist.append(
                math.sqrt((point[0] - centroid[0]) ** 2 + (point[1] - centroid[1]) ** 2))
    return calculated_dist


# User is prompted to enter filename and number of clusters and iterations for the program.
files = ('data1953.csv', 'data2008.csv', 'databoth.csv')
file = ''

while True:
    if file not in files:
        file = input(
            '''Please enter the file name you want to use:
            \ndata1953.csv"
            \ndata2008.csv
            \ndataBoth.csv 
            \nFilename:''').strip().lower()
    else:
        break

k = int(input("Please enter the number of clusters you want: "))
iterations = int(input("Please enter the number of iterations that the algorithm must run: "))
# calling function to access file and retrieve data
x = read_file(file)
data_ = np.ndarray.tolist(x[0][0:, :])
centroids = random.sample(data_, k)  # random sample for centroids
print('Random Centroids are: ' + str(centroids))


# function to initialize clusters.


def init_clusters(x_in=x, centroids_in=centroids, num_clusters=k):
    distances = np.reshape(euclidean_dist(
        centroids_in, x_in[0]), (len(centroids_in), len(x_in[0])))
    data_centers = []
    min_distances = []
    for value in zip(*distances):
        min_distances.append(min(value))
        data_centers.append(np.argmin(value) + 1)
    # dictionary stores number of clusters as specified by no_clusters variable and loops will
    # place data points in their closest cluster
    clusters = {}
    for cluster_i in range(0, num_clusters):
        clusters[cluster_i + 1] = []
    for d_point, center in zip(x_in[0], data_centers):
        clusters[center].append(d_point)

    #  loop will write the new centroid values with the newly calculated means
    for i, cluster in enumerate(clusters):
        reshaped = np.reshape(clusters[cluster], (len(clusters[cluster]), 2))
        centroids[i][0] = sum(reshaped[0:, 0]) / len(reshaped[0:, 0])
        centroids[i][1] = sum(reshaped[0:, 1]) / len(reshaped[0:, 1])
    return data_centers, clusters


# Function plot_visual() produces scatter plot for data points and centers. Graph
# of life expectancy vs. birth rate


def plot_visual(_data, num_clusters_):
    model = KMeans(num_clusters_)
    model.fit(_data)
    plt.scatter(x[0][0:, 0], x[0][0:, 1], cmap='rainbow', c=model.labels_)
    plt.xlabel('Birthrate')
    plt.ylabel('Life Expectancy')
    centers = np.reshape(centroids, (num_clusters_, 2))
    plt.plot(centers[0:, 0], centers[0:, 1], c='black', marker="*", markersize=14, linestyle=None, linewidth=0,
             label='Centroids')
    plt.title('Graph of Life expectancy vs. Birth Rate(per 100) ')
    plt.legend()
    plt.show()


for iteration in range(0, iterations):
    # function returns more that 1 value, stored in a variable easier access
    # dataframe created for better format when displaying to console
    assigning = init_clusters()
    cluster_data = pd.DataFrame({'Birth Rate': x[0][0:, 0],
                                 'Life Expectancy': x[0][0:, 1],
                                 'Cluster': assigning[0],
                                 'Country': x[1]})
    group_by_cluster = cluster_data[['Country', 'Birth Rate', 'Life Expectancy', 'Cluster']].groupby('Cluster')
    count_clusters = group_by_cluster.count()

# Data displayed to consoled
print("\nCOUNTRIES PER CLUSTER: \n" + str(count_clusters))
print("\nLIST OF COUNTRIES PER CLUSTER: \n", list(group_by_cluster))
print("\n BIRTH RATE & LIFE EXPECTANCY AVERAGES PER CLUSTER: \n", str(cluster_data.groupby(['Cluster']).mean()))

# variable mean contains clusters. dictionary will store difference between each data points it's mean
mean = assigning[1]
means = {}
for cluster_ in range(0, k):
    means[cluster_ + 1] = []

# loop used to calculate square distances between data points and cluster mean
for index, data in enumerate(mean):
    array = np.array(mean[data])
    array = np.reshape(array, (len(array), 2))
    # each variable holds calculation of cluster mean of values
    birth_rate = sum(array[0:, 0]) / len(array[0:, 0])
    life_exp = sum(array[0:, 1]) / len(array[0:, 1])

    # this loop appends squared distances between each data point in it's cluster and it's mean
    for data_point in array:
        distance = math.sqrt((birth_rate - data_point[0]) ** 2 + (life_exp - data_point[1]) ** 2)
        means[index + 1].append(distance)

plot_visual(data_, k)
