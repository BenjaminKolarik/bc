start_time <- Sys.time()

library(readxl)
library(ggplot2)
library(dplyr)
library(tidyr)
library(lmtest)
source("codes/R/time_utils.R")

output_dir <- "output/R_plots/LR"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

data <- read_excel("input/LR/LR_1000.xlsx")

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

residuals <- residuals(model)
print("Summary of Residuals:")
print(summary(residuals))


regression_plot <- ggplot(data_clean, aes(x = x, y = y)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE) +
  labs(title = "Linear Regression",
       x = "Independent Variable",
       y = "Dependent Variable") +
  theme_minimal()
ggsave(file.path(output_dir, "linear_regression.jpeg"), plot = regression_plot, width = 10, height = 6, dpi = 300)

residual_plot <- ggplot(data_clean, aes(x = model$fitted.values, y = residuals)) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red") +
  labs(title = "Residual Plot",
       x = "Fitted Values",
       y = "Residuals") +
  theme_minimal()
ggsave(file.path(output_dir, "residual_plot.jpeg"), plot = residual_plot, width = 10, height = 6, dpi = 300)

qq_plot <- ggplot(data_clean, aes(sample = residuals)) +
  stat_qq() +
  stat_qq_line() +
  labs(title = "Q-Q Plot of Residuals") +
  theme_minimal()
ggsave(file.path(output_dir, "qq_plot.jpeg"), plot = qq_plot, width = 10, height = 6, dpi = 300)

residual_histogram <- ggplot(data_clean, aes(x = residuals)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "blue", alpha = 0.7) +
  geom_density(color = "red") +
  labs(title = "Histogram of Residuals",
       x = "Residuals",
       y = "Density") +
  theme_minimal()
ggsave(file.path(output_dir, "residual_histogram.jpeg"), plot = residual_histogram, width = 10, height = 6, dpi = 300)

bp_test <- bptest(model)
print(bp_test)

shapiro_test <- shapiro.test(residuals)
print(shapiro_test)

dw_test <- dwtest(model)
print(dw_test)

end_time <- Sys.time()
execution_time <- end_time - start_time
execution_time

append_execution_time(
    time_second = execution_time,
    method_name = "LR",
    computer_name = "Windows Ryzen 9 5900x 32GB"
)

