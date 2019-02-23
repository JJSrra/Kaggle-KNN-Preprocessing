# Import libraries
library(ggplot2)

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

# Remove rows with NAs
train = train[-which(na.rows == TRUE),]
