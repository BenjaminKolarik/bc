library(readxl)
library(ggplot2)

# Load the data
data <- read_excel("./input/mtcars/LR.xlsx")

y <- data$y[1:10]
x <- data$x[1:10]

model <- lm(y ~ x)
summary(model)

data_graph <- data.frame(x = x, y = y)
ggplot(data_graph, aes(x = x, y = y)) +
  geom_point() +
  geom_smooth(method = "lm", se = TRUE) +
  labs(title = "Linear Regression",
       x = "x",
       y = "y") +
  theme_minimal()
