library(InvariantCausalPrediction)
library(tidyverse)
set.seed(111)

n <- 1000*4
environment <- rep(c(0,1,2,3), each = 1000)
X1 <- rnorm(n) + (environment==1)*rnorm(n, mean = 2)
X2 <- rnorm(n) + (environment==2)*rnorm(n, mean = 2)
Y <- X1+ X2 + rnorm(n)
X3 <- Y + rnorm(n)+(environment==3)*rnorm(n, mean = 2)

data <- data.frame(X1,X2,X3,Y,environment)
data$environment <- as.factor(as.character(data$environment))

icp <- ICP(cbind(data$X1, data$X2, data$X3), data$Y, data$environment, showAcceptedSets = TRUE)
summary(icp)

# Try for only one intervention

n <- 1000*2
environment <- rep(c(0,1), each = 1000)
X1 <- rnorm(n)
X2 <- rnorm(n)
Y <- X1+ X2 + rnorm(n)
X3 <- Y + rnorm(n) + (environment==1)*rnorm(n, mean = 2)

data <- data.frame(X1,X2,X3,Y,environment)
data$environment <- as.factor(as.character(data$environment))

icp <- ICP(cbind(data$X1, data$X2, data$X3), data$Y, data$environment, showAcceptedSets = TRUE)
summary(icp)
