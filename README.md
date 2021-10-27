# K-Means Algorithm

This program accesses data from a cvs file and interprets the data in the file using the K-mean clustering model

### What is the k-means algorithm?

The K-means algorithm allows us to interpret data and train a model to make future predictions based on the data.
It allows us to cluster similar data together and based on these clusters, we can predict which clusters future data belongs to.


### How it works

The k-means algorithm clusters similar data together. The "k" in k-means stands for
the number of clusters we wish to group ou data into. The “Mean” just refers to a simple average and it is the centre of each 
cluster that all the data points that belong to also known as the centroid.
Each data point belongs to the cluster with the nearest centroid and that 
is how the data is clustered.

This program reads data from a csv file and:
* Finds the Euclidean distance between two points (Each data point and each centroid)
* Compute the two-dimensional mean
* Visualises the processed data as a scatter plot indicating the different clusters

Since this program particularly works with files containing details about various countries Birth Rates and Life Expectancies,
it also displays the following output: 
1. The number of countries belonging to each cluster
1. The list of countries belonging to each cluster
1. The mean Life Expectancy and Birth Rate for each cluster
