start_time <- Sys.time()

library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)
library(car)

source("codes/R/time_utils.R")

output_dir <- "output/R_plots/ANOVA"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}
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

boxplot <- ggplot(data, aes(x = group, y = value, fill = group)) +
  geom_boxplot() +
  theme_minimal() +
  labs(
    title = "Boxplot of Values by Group",
    x = "Groups",
    y = "Values"
  ) +
  theme(legend.position = "none")
ggsave(file.path(output_dir, "boxplot.png"), plot = boxplot, width = 10, height = 6)

barplot <- ggplot(group_means, aes(x = group, y = mean, fill = group)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(
    title = "Barplot of Mean Values by Group",
    x = "Groups",
    y = "Mean Values"
  ) +
  theme(legend.position = "none")
ggsave(file.path(output_dir, "barplot.png"), plot = barplot, width = 10, height = 6)

tukey_result <- TukeyHSD(anova_result)


end_time <- Sys.time()
execution_time <- end_time - start_time
execution_time

residuals <- residuals(anova_result)
shapiro_test <- shapiro.test(residuals)
print(shapiro_test)
levene_test <- car::leveneTest(value ~ group, data = data_clean)
print(levene_test)


append_execution_time(
    time_second = execution_time,
    method_name = "ANOVA",
    computer_name = "Windows Ryzen 9 5900x 32GB"
)