# Machine Learning Fraud Detection
## Leslie Bai, Rebecca Wells, Jeremy Ross, Chris Pickens

### Links

- [Competition Link](https://www.kaggle.com/c/ieee-fraud-detection/notebooks?sortBy=hotness&group=everyone&pageSize=20&competitionId=14242&language=R)
- [Notebook We Built Off](https://www.kaggle.com/psystat/ieee-extensive-eda-lgb-with-r#Modeling)

### General Background

For our second team presentation we selected the IEEE-CIS Fraud Detection competition as our topic. The goal of this competition was to calculate the probability a given card transaction was fraudulent. To do so, we needed to create a model based on Vesta's real-world transactions. Although this competition provided a number of challenges to deal with such as the amount of data in question as well its completeness, we were ultimately able to come to an appropriate solution that improved on the notebooks available through changes in data cleaning as well as the usage of the xgboost method during the modeling phase.

### Code Setup

Before starting our code it makes sense to begin by clearing your environment using`rm(list=ls())`.
