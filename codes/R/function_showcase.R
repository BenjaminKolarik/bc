#Generovanie dát pre lineárnu regresiu
x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
y <- c(2, 4, 5, 7, 8, 10, 12, 14, 16, 18)

#Model lineárnej regresie
model <- lm(y ~ x)
summary(model)

#Generovanie dvoch vektorov
x <- c(1, 2, 3, 4, 5)
y <- c(2, 4, 6, 8, 10)

#Výpočet korelácie
correlation <- cor(x, y)
correlation

x <- rnorm(20, mean = 5, sd = 2) # 20 náhodných čísel s priemerom 5 a smerodajnou odchýlkou 2
y <- rnorm(20, mean = 10, sd = 3) # 20 náhodných čísel s priemerom 10 a smerodajnou odchýlkou 3
df <- data.frame(x, y)

library(ggplot2)
ggplot(df, aes(x = x, y = y)) + # Vytvorenie grafu
  geom_point() + # Bodový graf
  labs(title = "Bodovy graf", x = "X", y = "Y") + # Názvy osí
  theme_minimal() # Minimalistický dizajn


