#importing dataset

crime_data <- read.csv('crime_data.csv')
ncol(crime_data)
crime_data_update <- crime_data[2:5]

#Normalized the data
norm_crime_data <- scale(crime_data_update)

dendogram <- hclust(dist(crime_data_update,method='euclidean'),method='ward.D')
plot(dendogram,
     main = paste('Dendrogram'),
     xlab = 'Customers',
     ylab = 'Euclidean distance')

hc = hclust(dist(crime_data_update,method='euclidean'),method='ward.D')
y_hc = cutree(hc,3)

library(cluster)
clusplot(crime_data_update,
         y_hc,
         lines=0,
         shade = TRUE,
         color = TRUE,
         labels = 2,
         plotchar = FALSE,
         span = TRUE,
         main = paste('Cluster of crime'),
         xlab = 'income',
         ylab = 'year')