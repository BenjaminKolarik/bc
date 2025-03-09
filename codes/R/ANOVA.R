library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)

# Read the data
data <- read_excel("../../input/mtcars/tst.xlsx")

data <- data %>%
  mutate(across(contains("Predaj"), as.numeric))

str(data)

# Reshape the data to long format
data_long <- data %>%
  pivot_longer(cols = starts_with("Predaj"),
               names_to = "Predajna",
               values_to = "Vynos") %>%
  drop_na() %>%  # Added missing pipe operator
  rename(Dizajn.Obalu = `Dizajn Obalu`)

str(data_long)
# Convert to factors
data_long <- data_long %>%
  mutate(
    Predajna = as.factor(Predajna),
    Dizajn.Obalu = as.factor(Dizajn.Obalu)  # Reference the column correctly
  )

print(data_long)  # Changed show() to print() which is more commonly used in R
# One-way ANOVA for Dizajn.Obalu effect
anova_result <- aov(Vynos ~ Dizajn.Obalu, data = data_long)
summary(anova_result)

ggplot(data_long, aes(x = Dizajn.Obalu, y = Vynos)) +
  geom_boxplot() +
  labs(title = "One-way ANOVA",
       x = "Dizajn Obalu",
       y = "Vynos") +
  theme_minimal()