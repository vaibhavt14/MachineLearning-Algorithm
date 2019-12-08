
#Assignment-Clustering
#Dataset - Airlines

library(data.table)
library(readxl)

airlines <- read_xlsx('EastWestAirlines.xlsx',sheet='data')
View(airlines)
summary(airlines)
library(skimr)
skim(airlines)
summary(airlines)
colnames(airlines)
ncol(airlines)
new_airlines <- airlines[,2:12]
classifier <- scale(new_airlines)
# Using the dendrogram to find the optimal number of clusters

dendogram = hclust(d=dist(classifier,method = 'euclidean'),method='ward.D')
plot(dendogram,
     main=paste('Dendogram'),
     xlab = 'airlines',
     ylab = 'Euclidean distances')
plot(dendogram, hang = -1)

#Fitting hierarchical clustering
hc = hclust(d=dist(classifier,method = 'euclidean'),method = 'ward.D')
y_hc = cutree(hc,5)


#visulaising the cluster
library(cluster)
clusplot(classifier, 
         y_hc,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels= 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of airlines'))

#K-means

#importing dataset
airlines_1 <- read_xlsx('EastWestAirlines.xlsx',sheet='data')
X = airlines_1[,2:12]

#Using the elbow method to find optimal number of cluster
wcss = vector()
for (i in 1:10) wcss[i] = sum(kmeans(X, i)$withinss)
plot(1:10,
     wcss,
     type = 'b',
     main = paste('The Elbow Method'),
     xlab = 'Number of clusters',
     ylab = 'WCSS')

# Fitting K-Means to the dataset
set.seed(29)
kmeans <-  kmeans(X,5,iter.max = 300,nstart=100)
y_kmeans = kmeans$cluster

#Visulaising

clusplot(X,
         y_kmeans,
         lines = 0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Clusters of customers'))
