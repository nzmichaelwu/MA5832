# load libraries
library(tidyverse)
library(janitor)
library(xgboost)
library(keras)
library(tictoc)

### Load Data ----
data_raw <- readxl::read_xlsx("AUS_Data.xlsx", skip = 1)


### Exploratory Data Analysis ----
# a quick summary of the dataset
summary(data_raw)

# unemployment rate over period
ggplot(data_raw, aes(x = as.Date(Period), y = `Unemployment rate\r\nPercentage\r\n`)) + 
  geom_line() + 
  scale_x_date(date_labels= "%b %Y") +
  xlab("Year")
  
# exploratory analysis on the 2 columns that contain NAs
ggplot(data_raw, aes(x = as.Date(Period), y = as.numeric(`Job vacancies (000)`))) + 
  geom_point() + 
  geom_smooth(method = 'lm', formula = y~x)

ggplot(data_raw, aes(x = as.Date(Period), y = as.numeric(`Estimated Resident population(000)`))) +
  geom_point() + 
  geom_smooth(method = 'lm', formula = y~x)



### Data Pre-processing ----
data_mod <- data_raw %>% 
  mutate(Period = as.Date(Period)) %>% 
  rename(unemployment_rate = "Unemployment rate\r\nPercentage\r\n",
         gdp = "Gross domestic product: Percentage",
         govt_consumption_exp = "General government ;  Final consumption expenditure: Percentage",
         all_sectos_consumption_exp = "All sectors ;  Final consumption expenditure: Percentage",
         terms_of_trade = "Terms of trade: Index - Percentage") %>% # rename the percentage columns to a shorter format first
  mutate(unemployment_rate = unemployment_rate / 100,
         gdp = gdp / 100,
         govt_consumption_exp = govt_consumption_exp / 100,
         all_sectos_consumption_exp = all_sectos_consumption_exp / 100,
         terms_of_trade = terms_of_trade / 100) %>% # convert the percentage columns to raw number
  mutate(index = row_number())


# obtain coefficients of linear regression for imputing NAs
job_vacancies_lm <- lm(as.numeric(`Job vacancies (000)`) ~ index, data_mod)
summary(job_vacancies_lm)

est_pop_lm <- lm(as.numeric(`Estimated Resident population(000)`) ~ index, data_mod)
summary(est_pop_lm)

# create dummy columns to indicate NA or not, and impute NAs
data_mod <- data_mod %>% 
  mutate(na_job_vacancies = case_when(`Job vacancies (000)` == 'NA' ~ 1,
                                      TRUE ~ 0),
         na_est_pop = case_when(`Estimated Resident population(000)` == 'NA' ~ 1,
                                TRUE ~ 0)) %>% 
  mutate(`Job vacancies (000)` = case_when(na_job_vacancies == 1 ~ 21.07638 + 1.15091 * index,
                                           TRUE ~ as.numeric(`Job vacancies (000)`)),
         `Estimated Resident population(000)` = case_when(na_est_pop == 1 ~ 144324.7 + 656.2 * index,
                                                          TRUE ~ as.numeric(`Estimated Resident population(000)`)))


# normalise 3 columns that are on larger scales
data_mod <- data_mod %>% 
  mutate(cpi_scaled = scale(`CPI (all group)`, center = TRUE, scale = TRUE),
         job_vacancies_scaled = scale(`Job vacancies (000)`, center = TRUE, scale = TRUE),
         est_resident_pop_scaled = scale(`Estimated Resident population(000)`, center = TRUE, scale = TRUE)) %>% 
  mutate(cpi_scaled = as.vector(cpi_scaled),
         job_vacancies_scaled = as.vector(job_vacancies_scaled),
         est_resident_pop_scaled = as.vector(est_resident_pop_scaled))


# final dataframe for modelling
data_final <- data_mod %>% 
  select(c("Period", "unemployment_rate", "gdp", "govt_consumption_exp", "all_sectos_consumption_exp", "terms_of_trade",
           "cpi_scaled", "job_vacancies_scaled", "est_resident_pop_scaled")) %>% 
  clean_names()



### Machine Learning ----
# split train and test data
test_period <- seq(as.Date("2018-03-01"), as.Date("2020-09-01"), by = "quarter")

predictors <- c("gdp", "govt_consumption_exp", "all_sectos_consumption_exp", "terms_of_trade", 
                "cpi_scaled", "job_vacancies_scaled", "est_resident_pop_scaled")


train_data <- data_final %>% 
  filter(!period %in% test_period)

features_train <- train_data[,predictors] %>% as.matrix()
response_train <- train_data$unemployment_rate


test_data <- data_final %>% 
  filter(period %in% test_period)

features_test <- test_data[,predictors] %>% as.matrix()
response_test <- test_data$unemployment_rate


# simple linear regression model - for reference
simple_lm <- lm(unemployment_rate ~ gdp + govt_consumption_exp + all_sectos_consumption_exp + terms_of_trade + 
                  cpi_scaled + job_vacancies_scaled + est_resident_pop_scaled, data = train_data)

# get RMSE of the simple linear regression model
rss <- c(crossprod(simple_lm$residuals))
mse <- rss / length(simple_lm$residuals)
rmse_simple_lm <- sqrt(mse)
rmse_simple_lm

lm_predictions <- simple_lm %>% predict(test_data %>% select(-c("period")))
# caret::RMSE(lm_predictions, response_test)
caret::MAE(lm_predictions, response_test)

# Gradient Boosting Model - using xgboost package
# base model
set.seed(1234)

xgb_fit_base <- xgb.cv(
  data = features_train,
  label = response_train,
  nrounds = 1000,
  nfold = 10, # 10 folds cross validation
  objective = "reg:squarederror", # regression model, measure squared error
  verbose = 0,
  early_stopping_rounds = 30 # stop if no improvement for 30 consecutive trees
)

# get RMSE from base model as baseline
min(xgb_fit_base$evaluation_log[, test_rmse_mean])

# plot error vs number of trees on the base model
ggplot(xgb_fit_base$evaluation_log) +
  geom_line(aes(iter, train_rmse_mean), colour = "red") + 
  geom_line(aes(iter, test_rmse_mean), colour = "blue")

# tuning using hyperparameter grid
# create hyperparameter grid
hyper_grid <- expand_grid(
  eta = c(0.01, 0.05, 0.1, 0.3),
  max_depth = c(1, 2, 3),
  min_child_weight = c(1, 3, 5),
  subsample = c(0.5, 0.65, 0.8), # subsample of training set to prevent overfitting
  colsample_bytree = c(0.5, 0.8, 0.9),
  optimal_trees = 0, # to store results
  min_mae = 0 # to store results
)

tic("tunning")
print("start tunning the model...")
for(i in 1:nrow(hyper_grid)){
  print(i)
  params <- list(
    eta = hyper_grid$eta[i],
    max_depth = hyper_grid$max_depth[i],
    min_child_weight = hyper_grid$min_child_weight[i],
    subsample = hyper_grid$subsample[i]
  )
  
  set.seed(1234)
  
  xgb_tune <- xgb.cv(
    params = params,
    data = features_train,
    label = response_train,
    eval_metric = "mae",
    nrounds = 5000,
    nfold = 10, # 10 folds cross validation
    objective = "reg:squarederror", # regression model, measure squared error
    verbose = 0,
    watchlist = list(train = train_data, test = test_data),
    early_stopping_rounds = 10 # stop if no improvement for 10 consecutive trees
  )
  
  hyper_grid$optimal_trees[i] <- which.min(xgb_tune$evaluation_log$test_mae_mean)
  hyper_grid$min_mae[i] <- min(xgb_tune$evaluation_log$test_mae_mean)
}
print("...tunning done")
toc()

hyper_grid_arranged <- hyper_grid %>% 
  arrange(min_mae)

# xgboost tuned model
params <- list(
  eta = hyper_grid_arranged$eta[1],
  max_depth = hyper_grid_arranged$max_depth[1],
  min_child_weight = hyper_grid_arranged$min_child_weight[1],
  subsample = hyper_grid_arranged$subsample[1],
  eval_metric = "rmse"
)

xgb_tune_final <- xgb.cv(
  params = params,
  data = features_train,
  label = response_train,
  nrounds = hyper_grid_arranged$optimal_trees[1],
  nfold = 10, # 10 folds cross validation
  objective = "reg:squarederror", # regression model, measure squared error
  early_stopping_rounds = 30, # stop if no improvement for 30 consecutive trees
  verbose = 0
)

# plot error vs number of trees on the final tuned model
ggplot(xgb_tune_final$evaluation_log) +
  geom_line(aes(iter, train_rmse_mean), colour = "red") + 
  geom_line(aes(iter, test_rmse_mean), colour = "blue")

# train final xgboost model with the parameters determined using hyper grid.
xgb_fit_final <- xgboost(
  params = params,
  data = features_train,
  label = response_train,
  nrounds = hyper_grid_arranged$optimal_trees[1],
  objective = "reg:squarederror", # regression model, measure squared error
  verbose = 0
)

# save xgb final tuned model for later use, so dont need to run the hyper grid unless we need to retune.
saveRDS(xgb_fit_final, "xgb_fit_final.rds")
xgb_fit_final <- readRDS("xgb_fit_final.rds")

# predict on training set
pred_train_xgb_tuned <- predict(xgb_fit_final, features_train)
# caret::RMSE(pred_train_xgb_tuned, response_train)
caret::MAE(pred_train_xgb_tuned, response_train)

# plot the actual vs pred to see how well it fit - this is one way to check whether model is overfitting on training set or not, by comparing with the same plot on test data below
# train_data$pred <- pred_train_xgb_tuned
# ggplot(train_data) + 
#   geom_point(aes(x = period, y = unemployment_rate, colour = "red")) + 
#   geom_point(aes(x = period, y = pred, colour = "green"))


# predict on test set
pred_test_xgb_tuned <- predict(xgb_fit_final, features_test)
# caret::RMSE(pred_test_xgb_tuned, response_test)
caret::MAE(pred_test_xgb_tuned, response_test)

# plot the actual vs pred to see how well it fit
# test_data$pred <- pred_test_xgb_tuned
# ggplot(test_data) + 
#   geom_point(aes(x = period, y = unemployment_rate, colour = "red")) + 
#   geom_point(aes(x = period, y = pred, colour = "green"))


### Neural Network ----
# base neural net model
use_session_with_seed(1234)
nn_model_base <- keras_model_sequential() %>% 
  layer_dense(units = 8, activation = "relu", input_shape = 7) %>% 
  layer_dense(units = 1, activation = "linear")

nn_model_base %>% compile(
  loss = "mse",
  optimizer = "adam",
  metrics = list("mean_absolute_error")
)

nn_model_base %>% summary()

nn_model_base_history <- nn_model_base %>% 
  fit(features_train, as.matrix(response_train), epochs = 300, validation_split = 1/4)

nn_model_base_scores <- nn_model_base %>% evaluate(features_train, as.matrix(response_train))
nn_model_base_scores

# plot the bootstrapped validation MAE loss
df_plot1 <- data.frame(x = c(1:nn_model_base_history$params$epochs), y = nn_model_base_history$metrics$val_mean_absolute_error)
ggplot(df_plot1, aes(x = x, y = y)) +
  geom_point() +
  xlab("Epoch") +
  ylab("Estimated Validation MAE Loss")

# based on the plot, a network build on the entire training dataset with the num epochs that has the lowest val_MAE, which
# should produce a network that is a balance between the training data and future predictions.
nn_model_base %>%
  fit(features_train, as.matrix(response_train), epochs = which.min(nn_model_base_history$metrics$val_mean_absolute_error),
      validation_data = list(features_test, as.matrix(response_test)))

nn_model_base_scores_new_epochs <- nn_model_base %>% evaluate(features_train, as.matrix(response_train))
nn_model_base_scores_new_epochs

# save model
nn_model_base %>% save_model_hdf5("nn_model_base.h5")
nn_model_base <- load_model_hdf5("nn_model_base.h5")

# model prediction using train data, and get MAE
pred_train_nn_base <- nn_model_base %>% predict(features_train)
# caret::RMSE(pred_train_nn_base, response_train)
caret::MAE(pred_train_nn_base, response_train)

# model prediction using test data, and get MAE
pred_test_nn_base <- nn_model_base %>% predict(features_test)
# caret::RMSE(pred_test_nn_base, response_test)
caret::MAE(pred_test_nn_base, response_test)

# test_data$pred <- pred_test_nn_base
# ggplot(test_data) + 
#   geom_point(aes(x = period, y = unemployment_rate, colour = "red")) + 
#   geom_point(aes(x = period, y = pred, colour = "green"))


# vary the number of hidden layers, same number of neurons
use_session_with_seed(1234)
nn_model_two_layers <- keras_model_sequential() %>% 
  layer_dense(units = 8, activation = "relu", input_shape = 7) %>% 
  layer_dense(units = 8, activation = "relu") %>% 
  layer_dense(units = 1, activation = "linear")

nn_model_two_layers %>% compile(
  loss = "mse",
  optimizer = "adam",
  metrics = list("mean_absolute_error")
)

nn_model_two_layers %>% summary()

nn_model_two_layers_history <- nn_model_two_layers %>% 
  fit(features_train, as.matrix(response_train), epochs = 300, validation_split = 1/4)

# plot the bootstrapped validation MAE loss
df_plot2 <- data.frame(x = c(1:nn_model_two_layers_history$params$epochs), y = nn_model_two_layers_history$metrics$val_mean_absolute_error)
ggplot(df_plot2, aes(x = x, y = y)) +
  geom_point() +
  xlab("Epoch") +
  ylab("Estimated Validation MAE Loss")

# based on the plot, a network build on the entire training dataset with the num epochs that has the lowest val_MAE, which
# should produce a network that is a balance between the training data and future predictions.
nn_model_two_layers %>% 
  fit(features_train, as.matrix(response_train), epochs = which.min(nn_model_two_layers_history$metrics$val_mean_absolute_error), 
      validation_data = list(features_test, as.matrix(response_test)))

nn_model_two_layers_scores_new_epochs <- nn_model_two_layers %>% evaluate(features_train, as.matrix(response_train))
nn_model_two_layers_scores_new_epochs

# save model
nn_model_two_layers %>% save_model_hdf5("nn_model_two_layers.h5")
nn_model_two_layers <- load_model_hdf5("nn_model_two_layers.h5")

# model prediction using train data, and get MAE
pred_train_nn_two_layers <- nn_model_two_layers %>% predict(features_train)
# caret::RMSE(pred_train_nn_two_layers, response_train)
caret::MAE(pred_train_nn_two_layers, response_train)

# model prediction using test data, and get MAE
pred_test_nn_two_layers <- nn_model_two_layers %>% predict(features_test)
# caret::RMSE(pred_test_nn_two_layers, response_test)
caret::MAE(pred_test_nn_two_layers, response_test)

# vary the number of hidden layers and number of neurons for the 2nd dense layer
use_session_with_seed(1234)
nn_model_final <- keras_model_sequential() %>% 
  layer_dense(units = 8, activation = "relu", input_shape = 7) %>% 
  layer_dense(units = 16, activation = "relu") %>% # twice the number of inputs
  layer_dense(units = 1, activation = "linear")

nn_model_final %>% compile(
  loss = "mse",
  optimizer = "adam",
  metrics = list("mean_absolute_error")
)

nn_model_final %>% summary()

nn_model_final_history <- nn_model_final %>% 
  fit(features_train, as.matrix(response_train), epochs = 300, validation_split = 1/4)

# plot the bootstrapped validation MAE loss
df_plot3 <- data.frame(x = c(1:nn_model_final_history$params$epochs), y = nn_model_final_history$metrics$val_mean_absolute_error)
ggplot(df_plot3, aes(x = x, y = y)) +
  geom_point() +
  xlab("Epoch") +
  ylab("Estimated Validation MAE Loss")

# based on the plot, a network build on the entire training dataset with the num epochs that has the lowest val_MAE, which
# should produce a network that is a balance between the training data and future predictions.
nn_model_final %>% 
  fit(features_train, as.matrix(response_train), epochs = which.min(nn_model_final_history$metrics$val_mean_absolute_error), 
      validation_data = list(features_test, as.matrix(response_test)))

nn_model_final_scores_new_epochs <- nn_model_final %>% evaluate(features_train, as.matrix(response_train))
nn_model_final_scores_new_epochs

# save model
nn_model_final %>% save_model_hdf5("nn_model_final.h5")
nn_model_final <- load_model_hdf5("nn_model_final.h5")

# model prediction using train data, and get MAE
pred_train_nn_final <- nn_model_final %>% predict(features_train)
# caret::RMSE(pred_train_nn_final, response_train)
caret::MAE(pred_train_nn_final, response_train)

# model prediction using test data, and get MAE
pred_test_nn_final <- nn_model_final %>% predict(features_test)
# caret::RMSE(pred_test_nn_final, response_test)
caret::MAE(pred_test_nn_final, response_test)
