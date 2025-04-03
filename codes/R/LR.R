start_time <- Sys.time()

library(readxl)
library(ggplot2)
library(dplyr)
library(tidyr)
source("codes/R/time_utils.R")

data <- read_excel("input/LR/LR_1000.xlsx")
str(data)

data_clean <- data %>% drop_na(x, y)

model <- lm(y ~ x, data = data_clean)

coefficients <- coef(model)
print("Regression Coefficients:")
print(coefficients)

summary_result <- summary(model)
print(summary_result)

r_squared <- summary_result$r.squared
adj_r_squared <- summary_result$adj.r.squared
print(paste("R-squared:", r_squared))
print(paste("Adjusted R-squared:", adj_r_squared))

ggplot(data_clean, aes(x = x, y = y)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE) +
  labs(title = "Linear Regression",
       x = "Independent Variable",
       y = "Dependent Variable") +
  theme_minimal()


residuals <- residuals(model)
print("Summary of Residuals:")
print(summary(residuals))

qqnorm(residuals)
qqline(residuals)

plot(model, which = 1)
end_time <- Sys.time()
execution_time <- end_time - start_time
execution_time


append_execution_time(
    time_second = execution_time,
    method_name = "LR",
    computer_name = "Windows Ryzen 9 5900x 32GB"
)