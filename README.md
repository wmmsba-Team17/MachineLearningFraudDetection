# Machine Learning Fraud Detection
## Leslie Bai, Rebecca Wells, Jeremy Ross, Chris Pickens

### Links

- [Competition Link](https://www.kaggle.com/c/ieee-fraud-detection/notebooks?sortBy=hotness&group=everyone&pageSize=20&competitionId=14242&language=R)
- [Notebook We Built Off](https://www.kaggle.com/psystat/ieee-extensive-eda-lgb-with-r#Modeling)

### General Background

For our second team presentation we selected the IEEE-CIS Fraud Detection competition as our topic. The goal of this competition was to calculate the probability a given card transaction was fraudulent. To do so, we needed to create a model based on Vesta's real-world transactions. Although this competition provided a number of challenges to deal with such as the amount of data in question as well its completeness, we were ultimately able to come to an appropriate solution that improved on the notebooks available through changes in data cleaning as well as the usage of the xgboost method during the modeling phase.

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

