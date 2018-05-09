profit_data<-read.csv(file.choose(),sep=',',header=TRUE)
head(profit_data)

#Converting Categorical Data

profit_data$State<-factor(profit_data$State,levels = c('New York','California','Florida'),
                          labels=c(1,2,3))

#Splitting Data into Training and Testing Dataset
library(caret)
index<-createDataPartition(profit_data$Profit,p=0.8,list = FALSE)
trainset<-profit_data[index,]
testset<-profit_data[-index,]
dim(trainset)
dim(testset)

#Fit the model
regressor=lm(Profit~R.D.Spend+Administration+Marketing.Spend+State,data=trainset)
summary(regressor)

#Test the model
y_pred<-predict(regressor,testset)
y_pred

#Building optimal model using Backward Elimination Method1
#This can be done on entire dataset so that all observations contibute to the impact of variables
regressor1<-lm(Profit~R.D.Spend+Administration+Marketing.Spend,data=profit_data)
summary(regressor1)

regressor2<-lm(Profit~R.D.Spend+Marketing.Spend,data=profit_data)
summary(regressor2)
#P-Value of Marketing.spend is more than 0.05.Thus removing it,
regressor3<-lm(Profit~R.D.Spend,data=profit_data)
summary(regressor3)

#Method 2:Using step function for step wise regression
#AIC-Akaike Information Criteria.AIC=nlog)(SSE/n)+2(p+1).Error component present.Lower AIC better model
null<-lm(Profit~1,data = profit_data)
#summary(null)
full<-lm(Profit~.,data=profit_data)
#summary(full)
step(null,scope =list(lower=null,upper=full),direction="forward")
step(full,scope=list(lower=null,upper=full),direction="backward")
step(null,scope=list(lower=null,upper=full),direction="both")
