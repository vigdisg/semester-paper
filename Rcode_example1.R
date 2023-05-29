library(InvariantCausalPrediction)
library(AnchorRegression)
library(tidyverse)
set.seed(111)

n <- 10000*5
A <- rep(c(0, 1, 2, 3, 4), each = 10000)
X1 <- A + rnorm(n)
Y <- X1 + rnorm(n)
X2 <- Y + A + rnorm(n)

train <- data.frame(X1,X2,Y,A) %>% filter(A %in% c(0,1))
test <- data.frame(X1,X2,Y,A) %>% filter(A %in% c(2,3,4))

## Causal linear regression model

causal_model <- lm(Y ~ X1, data = train)
summary(causal_model)
mean((train$Y - predict(causal_model, newdata = train))^2)
#mean squared error for test data:
mean((test$Y[test$A == 2] - predict(causal_model, newdata = test[test$A == 2,]))^2)
mean((test$Y[test$A == 3] - predict(causal_model, newdata = test[test$A == 3,]))^2)
mean((test$Y[test$A == 4] - predict(causal_model, newdata = test[test$A == 4,]))^2)
#mse for train environments:
mean((train$Y[train$A == 0] - predict(causal_model, newdata = train[train$A == 0,]))^2)
mean((train$Y[train$A == 1] - predict(causal_model, newdata = train[train$A == 1,]))^2)

## OLS

ols <- lm(Y ~ X1 + X2, data = train)
summary(ols)
mean((train$Y - predict(ols, newdata = train))^2)
#MSE for test data:
mean((test$Y[test$A == 2] - predict(ols, newdata = test[test$A == 2,]))^2)
mean((test$Y[test$A == 3] - predict(ols, newdata = test[test$A == 3,]))^2) 
mean((test$Y[test$A == 4] - predict(ols, newdata = test[test$A == 4,]))^2) 
#mse for train environments:
mean((train$Y[train$A == 0] - predict(ols, newdata = train[train$A == 0,]))^2)
mean((train$Y[train$A == 1] - predict(ols, newdata = train[train$A == 1,]))^2)

## ICP

icp <- ICP(cbind(train$X1, train$X2), train$Y, as.character(train$A))
summary(icp)

## Anchor regression

X <- train %>% select(X1, X2, Y)
gamma <- 5
anchor_model <- anchor_regression(X, train$A, gamma, 'Y', lambda = 0) 
#lambda = 0 because we are not in high-dim setting
anchor_model$coeff
testmodel <- function(x1, x2) anchor_model$coeff[1] + anchor_model$coeff[2]*x1 + anchor_model$coeff[3]*x2
mean((train$Y-testmodel(train$X1, train$X2))^2)
#mse for test data:
mean((test$Y[test$A == 2]-testmodel(test$X1[test$A == 2], test$X2[test$A == 2]))^2)
mean((test$Y[test$A == 3]-testmodel(test$X1[test$A == 3], test$X2[test$A == 3]))^2)
mean((test$Y[test$A == 4]-testmodel(test$X1[test$A == 4], test$X2[test$A == 4]))^2)
#mse for train environments:
mean((train$Y[train$A == 0]-testmodel(train$X1[train$A == 0], train$X2[train$A == 0]))^2)
mean((train$Y[train$A == 1]-testmodel(train$X1[train$A == 1], train$X2[train$A == 1]))^2)
