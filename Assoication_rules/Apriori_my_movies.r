#Assignment-Apriori Algorithm

#importing libraries
library(arules)
library(arulesViz)
library(rmarkdown)
#data Preprocessing
dataset <- read.csv('my_movies.csv',header = FALSE)
dataset <- read.transactions('my_movies.csv',sep = ',',rm.duplicates = TRUE)
summary(dataset)

#Training Apriori algorithm
rules <- apriori(dataset[,6:15],parameter = list(support=0.4,confidence=0.8))
summary(rules)

#Improving model performance
inspect(head(sort(rules,by='lift')))
head(quality(rules))

plot(rules,method = "scatterplot")
plot(rules, method = "grouped")
plot(rules,method = "graph")
