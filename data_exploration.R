library(readr)
library(BaylorEdPsych)

library(readr)
train <- read_csv("~/Downloads/train.csv", 
                       col_types = cols(MSSubClass = col_factor(levels = c("60", 
                                                                                       "20", "70", "50", "190", "45", "90", 
                                                                                       "120", "30", "85", "80", "160", "75", 
                                                                                       "180", "40"))))

library(data.table)
setDT(train)

train[,Alley]


LittleMCAR(train[,1:3])






train <- read_csv("~/Downloads/train.csv")

missing_columns = numeric(81)
for(i in 1:81){
  missing_columns[i] = sum(is.na(train[, i]))
}

cbind(names(train), missing_columns)
dim(train)


plot(as.factor(train$MSSubClass))

sum(train$MSSubClass %in% c(20,60))
# 57% (1-2 stories, newer than 1946)

plot(as.factor(train$MSZoning))
# Residential Low Density
sum(train$MSZoning=="RL")
# 79% residential low density



