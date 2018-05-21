install.packages("arules")
install.packages("arulesViz")
library(arules)
library(arulesViz)
data(Groceries)
head(g)
head(Groceries)
itemFrequencyPlot(Groceries,topN=20,type="relative")

#Absolute real values %
#relavite gives percentile values

##supp=0.001 obs with values more than 0.001 are considered as rules, conf>0.8
#Get rules
rules<-apriori(Groceries,parameter = list(supp=0.001,conf=0.8))
#Show the first 5 rules but only 2 digits
inspect(rules[1:5])
##COunt -- no of times that rule occurs in data set
##Whenever lhs is  purchased, rhs is purchased.rhs is the one which you want to market
## or publicize
##Sort by rules
rules<-sort(rules,by="confidence",decreasing = TRUE)
inspect(rules[1:5])

#What are customers likely to buy before buying whole milk?

rules1<-apriori(data=Groceries,parameter = list(supp=0.001,conf=0.8),
                appearance = list(default="lhs",rhs="whole milk"),control=list(verbose=F))
rules1<-sort(rules1,decreasing = TRUE,by="confidence")
inspect(rules1[1:5])
rules1
#What are custmoers likely to buy if they purchase whole milk?
#rules2<-apriori(data=Groceries,parameter = list(supp=0.001,conf=0.8),
#                appearance = list(default="rhs",lhs="whole milk"),control=list(verbose=F))
#rules2<-sort(rules2,decreasing = FALSE,by="confidence")
#rules2
#inspect(rules2[1:2])
