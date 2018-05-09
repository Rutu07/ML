per_index<-read.csv(file.choose(),sep=",",header = T)
head(per_index)
pairs(~jpi+aptitude+tol+technical+general,data=per_index,col="blue")

#lm-> linear model
jpimodel<-lm(jpi~aptitude+tol+technical+general,data=per_index)
jpimodel   
summary(jpimodel)

#To model performance Index (find values of b1,b2,b3...)
per_index$jpi_pred<-fitted(jpimodel)
per_index$jpi_resi<-residuals(jpimodel)
head(per_index)

#Implement this model on new Data Set

per_index_new<-read.csv(file.choose(),header=T)
per_index_new$pred<-predict(jpimodel,per_index_new)
per_index_new
#31st March
install.packages("car")
library(car)
vif(jpimodel)

plot(per_index$jpi_pred,per_index$jpi_resi,col="red")

qqnorm(per_index$jpi_resi,col="blue")
qqline(per_index$jpi_resi,col="blue")

shapiro.test

##ncv test
ncvTest(jpimodel,~aptitude+technical+general+tol)

#Obtain correlated standard errors
library(car)
se_correct<-hccm(jpimodel)
library(lmtest)
coeftest(jpimodel,vcov=se_correct)

#### Outliers in reg model_detection
influencePlot(jpimodel,id.method = "identify",main="Influence Plot",sub="Circle size is proportional to Cook's Distance")
perindex_inf<-per_index[-(33)]
tail(perindex_inf)
perindex_model<-lm(jpi~aptitude+tol+technical+general,data=perindex_inf)
influencePlot(perindex_model,id.method="identify",main="Influence Plot",sub="Circle size is proportional to Cook's Distance")
