library(glmpath)
library(tidyverse)

data("heart.data")
heart_data <- data.frame(cbind(heart.data$x, heart.data$y))

set.seed(20)
cluster_heart <- heart_data %>% 
  select(c('sbp', 'ldl', 'obesity', 'famhist')) %>% 
  kmeans(., 2)
str(cluster_heart)

heart_data$Disease <- as.factor(cluster_heart$cluster)

# logistic regression
glm.fit <- glm(V10 ~ sbp + ldl + obesity + famhist, data = heart_data, family = binomial())

summary (glm.fit) # to observe model summaries
glm.probs <- predict(glm.fit,type = "response") # to obtain the probability that individual “i” has the disease.
glm.pred <- ifelse(glm.probs > 0.5, "Up", "Down") # for classification based on the model.

table(glm.probs, heart_data$V10)
