library(InvariantCausalPrediction)
library(AnchorRegression)
library(tidyverse)
set.seed(111)

n <- 10000*3
sigma <- rep(c(1, 0.1, 0.01), each = 10000)
X1 <- rnorm(n)*sigma
X2 <- X1 + rnorm(n)
Y <- X1 + rnorm(n)*sigma
X3 <- Y + rnorm(n)

train <- data.frame(X1,X2,X3,Y,sigma) %>% filter(sigma != 0.01)
test <- data.frame(X1,X2,X3,Y,sigma) %>% filter(sigma == 0.01)

## Causal linear regression model

causal_model <- lm(Y ~ X1, data = train)
summary(causal_model)
mean((train$Y - predict(causal_model, newdata = train))^2)
mean((test$Y - predict(causal_model, newdata = test))^2)

## OLS

ols <- lm(Y ~ X1 + X2 + X3, data = train)
summary(ols)
mean((train$Y - predict(ols, newdata = train))^2) 
mean((test$Y - predict(ols, newdata = test))^2) 

## ICP

icp <- ICP(cbind(train$X1, train$X2, train$X3), train$Y, as.character(train$sigma))
summary(icp)

## Anchor regression

X <- train %>% select(X1,X2,X3, Y)
gamma <- 5
anchor_model <- anchor_regression(X, train$sigma, gamma, 'Y', lambda = 0)
anchor_model$coeff
testmodel <- function(x1, x2, x3) anchor_model$coeff[1] + anchor_model$coeff[2]*x1 + anchor_model$coeff[3]*x2 + anchor_model$coeff[4]*x3
mean((train$Y-testmodel(train$X1, train$X2, train$X3))^2)
mean((test$Y-testmodel(test$X1, test$X2, test$X3))^2)
