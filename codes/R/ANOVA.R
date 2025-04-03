start_time <- Sys.time()

library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
source("codes/R/time_utils.R")

#treba zmenit na cestu na danom pocitaci
data <- read_excel("input/ANOVA/ANOVA_medium.xlsx")
data_clean <- data %>% drop_na(value)
str(data)

anova_result <- aov(value ~ group, data = data_clean)

summary_result <- summary(anova_result)

group_means <- data %>%
  group_by(group) %>%
  summarize(
    mean = mean(value, na.rm = TRUE),
    sd = sd(value, na.rm = TRUE),
    n = sum(!is.na(value)),
  )
print(group_means)

ggplot(data, aes(x = group, y = value, fill = group)) +
  geom_boxplot() +
  theme_minimal() +
    labs(
        title = "Boxplot of Values by Group",
        x = "Groups",
        y = "Values"
    ) +
  theme(legend.position = "none")

tukey_result <- TukeyHSD(anova_result)


end_time <- Sys.time()
execution_time <- end_time - start_time
execution_time

residuals <- residuals(anova_result)
shapiro_test <- shapiro.test(residuals)
print(shapiro_test)

qqnorm(residuals)
qqline(residuals)

append_execution_time(
    time_second = execution_time,
    method_name = "ANOVA",
    computer_name = "Windows Ryzen 9 5900x 32GB"
)