# Import libraries
library(ggplot2)
library(mice)
library(caret)
library(e1071)
library(FSelector)
library(kknn)

# Initiate random seed
set.seed(101)

# Data reading
# Mark "Source on Save" checkbox in RStudio to make sure the following line works!
setwd(dirname(parent.frame(2)$ofile))
train = as.data.frame(read.csv("../../Data/train.csv", na.strings = "?"))
test = as.data.frame(read.csv("../../Data/test.csv", na.strings = "?"))

# Take a first look at the data...
dim(train)
summary(train)

# ... and at the target (class) variable, labeled as "C"
ggplot(train, aes(C)) + geom_histogram(fill="red", col="darkred")

# Binary classification, with several more examples from 0 class.
# We might want to perform some oversampling/undersampling operation.

# Check for any missing values in train
anyNA(train)

# Check the amount of NAs and rows that have at least one NA
sum(is.na(train))
na.rows = apply(is.na(train), 1, function (x) Reduce('|', x))
sum(na.rows)

# There is the same amount of NAs in total and per row, which means that each row
# that has NAs has EXACTLY one missing value.

# Check the amount of NAs by column
na.cols = apply(is.na(train), 2, function(x) Reduce('|', x))
sum(na.cols)

# Every variable but the Class one has NAs
# Instead of removing the NAs now, we'll impute them later

# Check for duplicated rows
train[duplicated(train),]

# There are some duplicated rows that we want to remove
train = train[!duplicated(train),]

# Check attributes variance
apply(train, 2, var, na.rm = TRUE)

# Check for outliers
# Put NA where there is an outlier
train[,1:50] = as.data.frame(apply(train[,1:50], 2, function(column) {
  q1 = quantile(column, na.rm=TRUE)[2]
  q3 = quantile(column, na.rm=TRUE)[4]
  IQR.column = q3-q1
  lower.bound = q1 - 3*IQR.column
  # Only lower bound because upper extreme outliers look more like normal data 
  
  column[column <= lower.bound] = NA
  column
}))

# Check again the new amount of NAs
na.rows = apply(is.na(train), 1, function (x) Reduce('|', x))
sum(na.rows)

# There's a huge amount of outliers in at least one attribute

# Impute the "new missing values" (outliers) using the "mice" library
# and the "Predictive mean matching" method
impute = mice(train, m=3, maxit=3, method="pmm", seed=101)

# Check the correlations between instances of the imputation
cor(impute$imp$X1$`1`,impute$imp$X1$`3`)
cor(impute$imp$X1$`2`,impute$imp$X1$`3`)
cor(impute$imp$X1$`1`,impute$imp$X1$`2`)

# Note that imputation 3 has more similarities with 2 and 3 than those two have with each other

# Use imputation 3 for completing the dataset
train = complete(impute, 3)

# Check there are no missing values/outliers anymore
anyNA(train)

# Feature selection
# Remove similar features
correlationMatrix = cor(train)
highlyCorrelated = findCorrelation(correlationMatrix, cutoff=0.95)
train = train[,-highlyCorrelated]

# Feature selection by weights
subset = cfs(C~., train) 
formula = as.simple.formula(subset, "C")

# Treat class column as factor
train$C = factor(train$C, labels = 0:1, levels = 0:1)

# Separate data and labels
train.labels = train$C
train = train[,-51]

# Since we're working with KNN it's mandatory to normalize the data
# Get preprocess params from train dataset
preprocess.params = preProcess(train, c("center", "scale"))
train = predict(preprocess.params, train, k=21)

# Normalize test labels with the same train preprocessing
test = predict(preprocess.params, test, k=21)

# Predict test labels
knn.fit = kknn(formula, train, test)
predictions = knn.fit$fitted.values
predictions = ifelse(predictions == 1, 1, 0)

# Export the predictions to a .csv file
output = as.data.frame(cbind("Id"=1:length(predictions), "Prediction"=predictions))
write.table(output, file="submission-15.csv", quote=FALSE, sep=",", row.names=FALSE)
