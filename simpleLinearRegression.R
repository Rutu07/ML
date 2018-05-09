#IMport Data
dataset<-read.csv(file.choose(),sep=",",header = T)
head(dataset)
#Split Data in Training and Testing
library(caTools)
sp<-sample.split(dataset$Salary,SplitRatio = 2/3)
train1<-subset(dataset,sp==TRUE)
test1<-subset(dataset,sp==FALSE)
#Build Model on Train data
regressor<-lm(formula=Salary~YearsExperience,data=dataset)
summary(regressor)
#More * in summary represent that variable is highly statistically significance

y_pred<-predict(regressor,test1)
y_pred

#Visualizing Training and Testing Dataset
library(ggplot2)
ggplot()+
  geom_point(aes(x=train1$YearsExperience,y=train1$Salary),color="Red")+
  geom_line(aes(x=train1$YearsExperience,y=predict(regressor,train1)),color="Blue")+
  ggtitle('Salary vs Years of Experience')+
  xlab('Years of Experience')+
  ylab('Years of Experience')

ggplot()+
  geom_point(aes(x=test1$YearsExperience,y=test1$Salary),color="Red")+
  geom_line(aes(x=train1$YearsExperience,y=predict(regressor,train1)),color='Blue')+
  ggtitle('Salary vs Years of Experience')+
  xlab("Years of exp")+
  ylab("Salary")