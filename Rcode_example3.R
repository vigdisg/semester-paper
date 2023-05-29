library(InvariantCausalPrediction)
library(AnchorRegression)
library(tidyverse)
set.seed(111)

n <- 10000*3
sigma <- rep(c(0, 1, 4), each = 10000)
intervY <- rnorm(n,mean = 0, sd = sigma)
H <- rnorm(n)
X1 <- H + rnorm(n)
Y <- X1 + 2*H + rnorm(n) + intervY
X2 <- Y + H + rnorm(n)

train <- data.frame(X1,X2,Y,sigma, intervY) %>% filter(sigma != 4)
test <- data.frame(X1,X2,Y,sigma, intervY) %>% filter(sigma == 4)

## OLS

causal_model <- lm(Y ~ X1+X2, data = train)
summary(causal_model)
mean((train$Y - predict(causal_model, newdata = train))^2)
mean((test$Y - predict(causal_model, newdata = test))^2)

## ICP

icp <- ICP(cbind(train$X1, train$X2), train$Y, as.character(train$sigma))
summary(icp)

## Anchor regression

X <- train %>% select(X1,X2,Y)
gamma <- 5
anchor_model <- anchor_regression(X, train$intervY, gamma, 'Y', lambda = 0)
anchor_model$coeff
testmodel <- function(x1,x2) anchor_model$coeff[1] + anchor_model$coeff[2]*x1 + anchor_model$coeff[3]*x2
mean((train$Y-testmodel(train$X1, train$X2))^2)
mean((test$Y-testmodel(test$X1, test$X2))^2)

#test anchor variable as sigma:
anchor_model2 <- anchor_regression(X, train$sigma, gamma, 'Y', lambda = 0)
anchor_model2$coeff
testmodel2 <- function(x1,x2) anchor_model2$coeff[1] + anchor_model2$coeff[2]*x1 + anchor_model2$coeff[3]*x2
mean((train$Y-testmodel2(train$X1, train$X2))^2)
mean((test$Y-testmodel2(test$X1, test$X2))^2)

