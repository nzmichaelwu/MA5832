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
