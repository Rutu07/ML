motorfraud<-read.csv(file.choose(),sep = ',',header=TRUE)
head(motorfraud)
summary(motorfraud)
str(motorfraud)
#COnvert relevent int type into factors
motorfraud$police<-factor(motorfraud$police)
str(motorfraud)

motorfraud$witness<-as.factor(motorfraud$witness)
str(motorfraud)

motorfraud$rural<-factor(motorfraud$rural)
str(motorfraud)
## CREATE model
fraud_model<-glm(fraud~nfiles+police+witness+rural+records,family = binomial,data=motorfraud)
summary(fraud_model)

## nfiles is not significant SO remove it
fraud_model1<-glm(fraud~police+witness+rural+records,family = binomial,data=motorfraud)
summary(fraud_model1)

#Create Null model
null_fraud_model<-glm(fraud~1,family = binomial,data = motorfraud)
summary(null_fraud_model)

#ANOVA TEST
anova(fraud_model1,null_fraud_model,test = "Chisq")
#p value is 2.2e-16 which is much smaller than 0.05.Hence we reject null hypothesis
#which says there is no significant difference bw both models.Thus there is significant 
# difference bw this our null model

#Fit the model
motorfraud$predicted_probability<-round(fitted(fraud_model1),2)
head(motorfraud)

#TO check the goodness of curve
library(gmodels)
table(motorfraud$fraud,fitted(fraud_model1)>0.4)

## ROCR curve to select the cutoff value
head(motorfraud)
pred_fraud<-prediction(motorfraud$predicted_probability,motorfraud$fraud)
perf_fraud<-performance(pred_fraud,"tpr","fpr")
plot(perf_fraud,colorize=T,print.cutoffs.at=seq(0.1,by=0.1))
##From the ROC Curve 0.2 can be used as a good cut off

vif(fraud_model1)
library(car)
influencePlot(fraud_model1)

auc<-performance(pred_fraud,"auc")
auc@y.values




