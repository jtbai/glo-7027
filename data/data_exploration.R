library(readr)
library(BaylorEdPsych)
# library(data.table)
library(readr)


train <- read_csv("./train.csv", col_types = cols(MSSubClass = col_factor(levels = c("60", "20", "70", "50", "190", "45", "90", "120", "30", "85", "80", "160", "75", "180", "40"))))
# setDT(train)


missing_columns = numeric(81)
for(i in 1:81){
  missing_columns[i] = sum(is.na(train[, i]))
}
cbind(names(train), missing_columns)

plot(as.factor(train$MSSubClass))
sum(train$MSSubClass %in% c(20,60))
# 57% (1-2 stories, newer than 1946)

plot(as.factor(train$MSZoning))
sum(train$MSZoning=="RL")
# Residential Low Density
# 79% residential low density

par(mfrow=c(9,9))

for(i in 1:81){
  if(is.numeric(unlist(train[, i]))){
    plot(density(unlist(train[, i]), na.rm=T), main=names(train)[i])
  }else if(is.factor(unlist(train[, i]))){
    plot(as.factor(unlist(train[, i])), main=names(train)[i])
  }else if(is.character(unlist(train[, i]))){
    plot(as.factor(unlist(train[, i])), main=names(train)[i])
  }else{
    plot(1:10, 1:10, type="l", col="red", main=names(train)[i])
  }
}




hist(train$SalePrice)

'%!in%' <- function(x,y)!('%in%'(x,y))
for(i in 1:81){
  for(j in 1:1460){
    if(i %!in% c(4, 26, 27)) {
      if(is.na(train[j,i])){
        train[j,i] = "None"
      }
    }
  }
}


na_to_none(train, "Alley")



