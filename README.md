# Machine Learning Fraud Detection
## Leslie Bai, Rebecca Wells, Jeremy Ross, Chris Pickens

### Links

- [Competition Link](https://www.kaggle.com/c/ieee-fraud-detection/notebooks?sortBy=hotness&group=everyone&pageSize=20&competitionId=14242&language=R)
- [Notebook We Built Off](https://www.kaggle.com/psystat/ieee-extensive-eda-lgb-with-r#Modeling)

### General Background

For our second team presentation we selected the IEEE-CIS Fraud Detection competition as our topic. The goal of this competition was to calculate the probability a given card transaction was fraudulent. To do so, we needed to create a model based on Vesta's real-world transactions. Although this competition provided a number of challenges to deal with such as the amount of data in question as well its completeness, we were ultimately able to come to an appropriate solution that improved on the notebooks available through changes in data cleaning as well as the usage of the xgboost method during the modeling phase.

---

### Original Notebook

Looking at the original notebook we decided to critique and build off, there were a lot of aspects we liked and wanted to include in our own solution. Specifically, 

--- 

### Code Setup

Before starting our code it makes sense to begin by clearing your environment using`rm(list=ls())`. We then can use `needed` list as an argument for the `installifAbsentAndLoad` function defined below. This allows us to ensure R is running the packages required for the rest of the program.

```
installIfAbsentAndLoad <- function(neededVector) {
  for(thispackage in neededVector) {
    if( ! require(thispackage, character.only = T) )
    { install.packages(thispackage)}
    require(thispackage, character.only = T)
  }
}

needed <- c('tidyverse','xgboost','dplyr','data.table','inspectdf',
            'tictoc','caret')  
installIfAbsentAndLoad(needed)
```
### Load Data and Overview

After loading the data into the following four tables, we can start exploring it.

```
train_identity <- read.table("train_identity.csv", header=T, sep=',') 
test_identity <- read.table("test_identity.csv", header=T, sep=',')
train_transaction <- read.table("train_transaction.csv", header=T, sep=',')
test_transaction <- read.table("test_transaction.csv", header=T, sep=',')
```

One aspect we can investigate is Dimensions, which shows us the identity tables have roughly 125000 rows and 41 columns while the transaction tables have around 550000 rows and either 394 or 393 columns. 

```
(dim(train_identity))  # row, column 
(dim(test_identity))
(dim(train_transaction))
(dim(test_transaction))
```
We can also use `head(train_transaction)` to get an idea of just how many missing variables we'll need to deal with later on in the process. As you can see by the output to the head function, it's a significant amount.

Now that we've had a chance to get a first look at the data, we can use the left join function to combine the train_identity and test_identity tables with the train_transaction and test_transaction tables, respectively. Once this is completed we can get rid of the four initial tables as they no longer serve a function for us.

```
train <- left_join(train_transaction, train_identity)
test <- left_join(test_transaction, test_identity)

rm(train_identity, train_transaction, test_identity, test_transaction)
```

If you want, at this point you can check the dimensions of the new train and test tables to confirm the join was done properly and use the `head()` function to see if the addition of the identity tables made our missing values problem worse. Thankfully, at first glance it appears that this isn't the case. The identity tables contained signficiant amounts of categorical information that was by and large aquired uniformly for all or most transactions. While there are still new missing values that we now need to deal with in addition to our initial transaction table ones, the impact could have been worse.

```
(dim(train)) 
(dim(test))

head(train)
```

After looking at the combined data, we can get started on dealing with its missing variables. As a first step to this, we can use the `is.na()` function in combination with the `colSums()` function to calculate just how many columns in our data have missing values. We can then use print statements to show us a clean look at the scale of the missing values problem. Using the code below, we now know 409 out of 434 train columns and 380 out of 435 test columns are missing values.

```
missing_train <- colSums(is.na(train))[colSums(is.na(train)) > 0] %>% sort(decreasing=T)
missing_test <- colSums(is.na(test))[colSums(is.na(test)) > 0] %>% sort(decreasing=T)

print(paste(length(missing_train), 'columns out of', ncol(train), 'have missing values in train'))
print(paste(length(missing_test), 'columns out of', ncol(test), 'have missing values in test'))
```

While the above code is helpful for getting a grasp on the overall number of columns missing values, we still don't know how bad the problem is for each individual column. Using our previously defined `missing_train` and `missing_test` numeric_lists, we can divide each by the number of rows in the train and test tables to get and print the proportion of missing values for each column. For a more visual look at the problem, we can also create histograms showing missing values for the train and test sets. As the resulting charts show, a significant number of columns in both the train and test set are missing more than 80% of their values, suggesting drastic actions may need to be taken.

```
(missing_train_pct <- round(missing_train/nrow(train), 2))
(missing_test_pct <- round(missing_test/nrow(test), 2))

hist(missing_train_pct,xlab = 'Percent of Values Missing',main='Missing Values for Train Columns')
hist(missing_test_pct,xlab = 'Percent of Values Missing',main='Missing Values for Test Columns')
```
<img src="Test_Missing.png" alt="Missing Values for Test Columns" width="750"/>

<img src="Train_Missing.png" alt="Missing Values for Train Columns" width="750"/>



