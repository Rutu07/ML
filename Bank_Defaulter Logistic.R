bankloan<-read.csv(file.choose(),sep = ",",header=TRUE)
head(bankloan)
str(bankloan)
## Convert age into factor ie categorical variable so that in future it can be used to categorize
bankloan$AGE<-factor(bankloan$AGE)

##Create model
riskmodel<-glm(DEFAULTER~AGE+EMPLOY+ADDRESS+DEBTINC+CREDDEBT+OTHDEBT,family = binomial,data = bankloan)
summary(riskmodel)

riskmodel1<-glm(DEFAULTER~EMPLOY+ADDRESS+DEBTINC+CREDDEBT,family=binomial,data = bankloan)
summary(riskmodel1)

##Create NUll model
nullmodel<-glm(DEFAULTER~1,family = binomial,data=bankloan)
summary(nullmodel)

##ANoVA checks the variance

anova(nullmodel,riskmodel1,test="Chisq")
#Anova method proves. Here p value<0/05 shows that our model is more significant with respect to null model

#Fitted used in MLR gives continuous value.Here fitted function gives probability based on our model for existing values
bankloan$predprob<-round(fitted(riskmodel1),2)
head(bankloan)

### TO check goodness of curve
install.packages("gmodels")
library(gmodels)
table(bankloan$DEFAULTER,fitted(riskmodel)>0.5)
no_of_rows<-dim(bankloan)
no_of_rows
misclassification_rate=((41+88)/700)*100
misclassification_rate

####How the cut off affects sensitivity and specificity
table(bankloan$DEFAULTER,fitted(riskmodel)>0.1)
table(bankloan$DEFAULTER,fitted(riskmodel)>0.2)
table(bankloan$DEFAULTER,fitted(riskmodel)>0.3)
table(bankloan$DEFAULTER,fitted(riskmodel)>0.4)
table(bankloan$DEFAULTER,fitted(riskmodel)>0.5)
missClassificationRate_of_0.4<-((68+67)/700)*100
missClassificationRate_of_0.4

install.packages("ROCR")
library(ROCR)
head(bankloan$predprob)
pred<-prediction(bankloan$predprob,bankloan$DEFAULTER)
perf<-performance(pred,"tpr","fpr")
plot(perf)
abline(0,1)
pred
perf
## To understand better use this plot/curve
plot(perf,colorize=T,print.cutoffs.at=seq(0.1,by=0.1))

### Co-efficient is expressed in the form of odds (log). So take exponential of that
##These values are relatvie(under influence of other variables) All are relative effects
coef(riskmodel1)
exp(coef(riskmodel1))
abs((exp(coef(riskmodel1))-1)*100)
### - sign indicates it will reduce. If number of years  at a particular addresses increases,the percentage that the person will not default reduces by 7.8%
library(car)
influencePlot(riskmodel1)
library(car)
vif(riskmodel1)

step(riskmodel,scope = list(lower=nullmodel,upper=riskmodel),direction = "backward")
step(nullmodel,scope = list(lower=nullmodel,upper=riskmodel),direction = "forward")

##Write file w/o age abd otherdebt

#Hold out validation

library(caret)
index<-createDataPartition(bankloan$DEFAULTER,P=0.7,list=F)
head(index)
dim(index)
traindata1<-bankloan[index,]
testdata1<-bankloan[-index,]
dim(traindata1)
dim(testdata1)
head(traindata1)
##Like if else we have a substitute for it function:cut
riskmodel2<-glm(DEFAULTER~EMPLOY+ADDRESS+DEBTINC+CREDDEBT,family = binomial,data=traindata1)
traindata1$predprob<-predict(riskmodel2,traindata1,type='response')
head(traindata1)
traindata1$PredDefalutersY<-ifelse(traindata1$predprob>0.3,1,0)
confusionMatrix(traindata1$Predicted_DefalutersY,traindata1$DEFAULTER,positive ="1")

###Testing model

testdata1$predprob<-predict(riskmodel2,testdata1,type = 'response')
head(testdata1)
testdata1$PredictedDefaultersY<-ifelse(testdata1$predprob>0.3,1,0)
head(testdata1)
confusionMatrix(testdata1$PredictedDefaultersY,testdata1$DEFAULTER)

###ROC CURVE on Train data Study prediction and performance
pred_train<-prediction(traindata1$predprob,traindata1$DEFAULTER)
perf_train<-performance(pred_train,"tpr","fpr")
plot(perf_train,colorize=T,print.cutoffs.at=seq(0.1,by=0.1))
###ROC CURVE on Test Data

pred_test<-prediction(testdata1$predprob,testdata1$DEFAULTER)
perf_test<-performance(pred_test,"tpr","fpr")
plot(perf_test,colorize=T,print.cutoffs.at=seq(0.1,by=0.1))

#value more than 0.6 is considered the good model

auc<-performance(pred_train,"auc")
auc@y.values

auc<-performance(pred_test,"auc")
auc@y.values