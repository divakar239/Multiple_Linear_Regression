#insatlling packages
#install.packages("caTools")
#install_packages("ggplot2")

#importing libraries
library(caTools)

#importing dataset
dataset <- read.csv("/Users/DK/Documents/Machine_Learning/Python-and-R/Machine_Learning_Projects/Multiple_Linear_Regression/50_Startups.csv")

#categorical data
dataset$State<- factor(dataset$State, levels=c('New York','Florida','California'), labels=c(1,2,3))


#test and train sets
split <- sample.split(dataset$Profit,SplitRatio = 0.8)
training_set <- subset(dataset,split = TRUE)
test_set <- subset(dataset,split = FALSE)

#multiple linear regression model
mult_lin_reg <- lm(Profit~., data=training_set)
y_pred <- predict(mult_lin_reg,test_set)
summary(mult_lin_reg)

#back elimination
mult_lin_reg <- lm(Profit~.-State, data=training_set)
summary(mult_lin_reg)

mult_lin_reg <- lm(Profit~.-State-Administration, data=training_set)
summary(mult_lin_reg)

mult_lin_reg <- lm(Profit~R.D.Spend, data=training_set)
summary(mult_lin_reg)

