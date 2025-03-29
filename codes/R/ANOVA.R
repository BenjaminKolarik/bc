start_time <- Sys.time()
#setwd("codes/R")

library(readxl)
library(dplyr)
library(tidyr)
library(ggplot2)

#treba zmenit na cestu na danom pocitaci
data <- read_excel("../../input/mtcars/tst.xlsx")
data <- data %>%
  mutate(across(contains("Predaj"), as.numeric))

str(data)

data_long <- data %>%
  pivot_longer(cols = starts_with("Predaj"),
               names_to = "Predajna",
               values_to = "Vynos") %>%
  drop_na() %>%
  rename(Dizajn.Obalu = `Dizajn Obalu`)

str(data_long)

data_long <- data_long %>%
  mutate(
    Predajna = as.factor(Predajna),
    Dizajn.Obalu = as.factor(Dizajn.Obalu)
  )

print(data_long)
anova_result <- aov(Vynos ~ Dizajn.Obalu, data = data_long)
summary(anova_result)

ggplot(data_long, aes(x = Dizajn.Obalu, y = Vynos)) +
  geom_boxplot() +
  labs(title = "One-way ANOVA",
       x = "Dizajn Obalu",
       y = "Vynos") +
  theme_minimal()

end_time <- Sys.time()
execution_time <- end_time - start_time
execution_time