library(dplyr)
library(gam)
library(readr)
setwd("~/PycharmProjects/glo-7027")

# Importation des données -------------------------------------------------

trainacp <- read_csv("trainind.csv")
trainacp <- read_csv("trainacp.csv")
names(trainacp) <- c("Id", paste0("CP", 1:10))

y <- read_csv("yval.csv")

y <- y %>% mutate(Id = as.integer(Id - 1))

# Entrainement des modèles ------------------------------------------------

trainacp <- trainacp %>% left_join(y, by = "Id") %>% select(-Id)

indexCV <- matrix(sample(1:nrow(trainacp), replace = FALSE), ncol = 7)

predictLM <- numeric(nrow(trainacp))
predictGLM <- numeric(nrow(trainacp))
predictGAM <- numeric(nrow(trainacp))

for(i in 1:7) {
  trainindex <- as.vector(indexCV[, -i])
  testindex <- indexCV[, i]
  
  lm_fit <- lm(SalePrice ~ ., data = trainacp[trainindex, ])
  glm_fit <- glm(SalePrice ~ ., family = Gamma(link = "inverse"), data = trainacp[trainindex, ])
  gam_fit <- gam(SalePrice ~ ., family = Gamma(link = "inverse"), data = trainacp[trainindex, ])
  
  predictLM[testindex] <- predict(lm_fit, newdata = trainacp[testindex, -11], type = "response")
  predictGLM[testindex] <- predict(glm_fit, newdata = trainacp[testindex, -11], type = "response")
  predictGAM[testindex] <- predict(gam_fit, newdata = trainacp[testindex, -11], type = "response")
  
}

mean((predictLM - y[, 2]) ^ 2)
mean((predictGLM - y[, 2]) ^ 2)
mean((predictGAM - y[, 2]) ^ 2)
