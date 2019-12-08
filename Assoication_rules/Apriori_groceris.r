
#Assignment - Apriori algorithm

#importing libraries
library(arules)
#data Preprocessing
dataset <- read.csv('groceries.csv',header=FALSE)
dataset <- read.transactions('groceries.csv',sep=',',rm.duplicate=TRUE)
View(dataset)
summary(dataset)
inspect(dataset[1:5])
itemFrequencyPlot(dataset,topN=10)
image(dataset[1:5])
image(sample(dataset,100))
#Training Apriori on the dataset
rules <-  apriori(dataset,parameter = list(support=0.004,confidence=0.25))
rules
#evaluating model performance
summary(rules)
inspect(rules[1:3])

#Improving model performance
inspect(sort(rules,by='lift')[1:5])

write(rules,file = 'groceryrules.csv',sep=',',quote=TRUE,row.names=FALSE)

rules.df <- as(rules,'data.frame')
str(rules.df)
                  