start_time <- Sys.time()
#setwd("codes/R")

library(readxl)
library(ggplot2)

data <- read_excel("input/mtcars/LR.xlsx")

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

end_time <- Sys.time()
execution_time <- end_time - start_time
execution_time