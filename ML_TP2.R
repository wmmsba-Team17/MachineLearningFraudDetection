rm(list=ls())
####################################################
####           Load required packages           ####
####################################################

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

#rm(list=ls())

####################################################
#########          Load Data           #############
####################################################
train_identity <- read.table("train_identity.csv", header=T, sep=',') 
test_identity <- read.table("test_identity.csv", header=T, sep=',')
train_transaction <- read.table("train_transaction.csv", header=T, sep=',')
test_transaction <- read.table("test_transaction.csv", header=T, sep=',')


#Exploratory Data Analysis
(dim(train_identity))  # row, column 
(dim(test_identity))
(dim(train_transaction))
(dim(test_transaction))

#head(train_transaction)

# joining transaction and identity tables by TrasactionID
train <- left_join(train_transaction, train_identity)
test <- left_join(test_transaction, test_identity)

(dim(train)) 
(dim(test))


#we will be using combined train and test from now, so remove following 
rm(train_identity, train_transaction, test_identity, test_transaction)

#head(train)
#head(test)

# Observe missing values from training and test dataset  
missing_train <- colSums(is.na(train))[colSums(is.na(train)) > 0] %>% sort(decreasing=T)
missing_test <- colSums(is.na(test))[colSums(is.na(test)) > 0] %>% sort(decreasing=T)
print(paste(length(missing_train), 'columns out of', ncol(train), 'have missing values in train'))
print(paste(length(missing_test), 'columns out of', ncol(test), 'have missing values in test'))

# Ratio of missing values for all vars. 
(missing_train_pct <- round(missing_train/nrow(train), 2))
(missing_test_pct <- round(missing_test/nrow(test), 2))

# find out which variables has more than 85 of missing values and drop them! 
drop_train_col <- names(which(missing_train_pct>0.85))
(train <- train[ , !(names(train) %in% drop_train_col)])
length(drop_train_col)  #69 
length(train)  #366

drop_test_col <- names(which(missing_test_pct>0.85))
(test <- test[ , !(names(test) %in% drop_test_col)])
length(test)  #419
length(drop_test_col)  #15

all(drop_test_col %in% drop_train_col)  #make sure all the cols in test set are in the training set 

# check if they are the same vars for training and test set
#which(drop_train_col != drop_test_col)
setdiff(drop_train_col, drop_test_col)


# Target Variable / Y variable: isFraud
target_var <- factor(train$isFraud)
class(target_var) #check type:factor 

# Make full dataset 
train$key <- "train"
test$key <- "test"
full <- bind_rows(train, test)
length(full)  #420 
#rm(train, test)
#gc()

#categorical variables 
categorical_vars <- c("ProductCD","card1","card2","card3","card4","card5","card6","addr1","addr2","P_emaildomain",
                      "R_emaildomain","M1","M2","M3","M4","M5","M6","M7","M8","M9","DeviceType","DeviceInfo","id_12",
                      "id_13","id_14","id_15","id_16","id_17","id_18","id_19","id_20","id_21","id_22","id_23","id_24",
                      "id_25","id_26","id_27","id_28","id_29","id_30","id_31","id_32","id_33","id_34","id_35","id_36",
                      "id_37","id_38")
sapply(full[categorical_vars], function(x) length(unique(x))) %>% sort(decreasing=T) %>% print

#There are two types of missing values : blank and "NA" 
# Let's see an example of "NA" in col 'DeviceType'
full['DeviceType'] %>% table(useNA = 'ifany') # " ifany " shows missing values when present.
sum(sapply(full['DeviceType'], as.character)=="", na.rm=T)


nunique <- sapply(full, function(x) length(unique(x)))
print(sum(nunique < 2))  # 0:indicates that all vars have more than one unique values

# missing values in full dataset 
(missing_rate <- colSums(is.na(full))/nrow(full))
#length(which(missing_rate >0.85))  #54

# drop again, and this time is mainly Vs... 
del_vars <- missing_rate[missing_rate> 0.85] %>% names
(full <- full[ , !(names(full) %in% del_vars)])
#length(full)  #366 

# Inspect type of vars by using 'inspectdf' package 
# for reduce computing burden in the later steps: to convert numerical to integers if we could 
(numeric_vars <- inspect_num(full)$col_name)

# Convert numeric_vars to integers 
is_int <- function(x){
  fnum <- fivenum(x)
  return(identical(fnum, floor(fnum))) 
}

tic("check is integer")
int_idx <- sapply(full[numeric_vars], is_int)
toc()
#check is integer: 15.9 sec elapsed
int_vars <- names(int_idx)[int_idx]
paste("Number of numeric variables that we can convert to interger:", length(int_vars))
#"Number of numeric variables that we can convert to interger: 311"
before <- object.size(full)
print(paste("Before :", format(before, units = "MB")))  #"Before : 2949.8 Mb"

full[int_vars] <- lapply(full[int_vars], as.integer)
after <- object.size(full)
print(paste("After :", format(after, units = "MB") ))   # "After : 1652.3 Mb"

invisible(gc())

# Find out categorical vars and as.factor them 
(categorical_vars <- inspect_cat(full)$col_name)
full[, categorical_vars] <- lapply(full[, categorical_vars], as.factor)


# Demonstrate some important variables 
# 1. card1 - card6: payment card information, such as card type, card category, issue bank, country, etc.
summary(full$card1)
summary(full$card2)
summary(full$card3)
summary(full$card4)  #issue bank
summary(full$card5)
summary(full$card6)  #card type 
# 2. TransactionAMT: transaction payment amount in USD
summary(full$TransactionAmt)
# 3.ProductCD: product code, the product for each transaction
summary(full$ProductCD)
# 4. DeviceType 
summary(full$DeviceType)


####################################################
####            Modeling Step                   ####        
####################################################
# Split into training and test dataset in 80/20 ratio
set.seed(5082)

#drop rows with NA in isFraud becuase this is supervised 
full <- full[!is.na(full$isFraud),]
n <- nrow(full)
train.indices <- sample(1:n, 0.8*n)  
train <- full[train.indices,]
test <- full[-train.indices,]

# prepare data

#convert x vars into matrix due to format requirement of XGB modeling 

x_train <- data.matrix(train %>% select(-isFraud, -TransactionID))
y_train <- as.numeric(as.factor(train$isFraud))-1

x_test <- data.matrix(test %>% select(-isFraud, -TransactionID))
y_test <- as.numeric(as.factor(test$isFraud))-1

#rm(full)  #remove full dataset after splitting 



dtrain <- xgb.DMatrix(data = x_train, label=y_train) 
#xgb.DMatrix  dim: 472432 x 364  info: label  colnames: yes
dtest <- xgb.DMatrix(data = x_test, label=y_test)
#xgb.DMatrix  dim:  118108 x 364  info: label  colnames: yes

watchlist <- list(train = dtrain, test = dtest)

tic("Start training with xgb.train")
model_xgb <- xgb.train(data = dtrain,                     
                    eval.metric = "auc",
                    max.depth = 9, 
                    eta = 0.05, 
                    subsample = 0.9,
                    colsample_bytree = 0.9,
                    nthread = 2, 
                    nrounds = 500,
                    early_stopping_rounds = 20,
                    verbose = 1,
                    watchlist = watchlist,
                    objective = "binary:logistic")
toc()

#save model
xgb.save(model_xgb, 'ml2_tp2_xgb.model')

#summary of model 
model_xgb  
#best_score : 0.964697 

# fit model and make prediction 
pred <- predict(model_xgb, x_test)

# Calculate MSE 
mean((pred - y_test)^2)  #0.01345146 

#write in to submission.csv
#submission <- read.csv('sample_submission.csv')
#submission$isFraud <- pred
#submission %>% head

## save to file
#write.csv(submission, file = "submission.csv", row.names = F)


