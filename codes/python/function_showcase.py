import pandas as pd
#načítanie a uloženie dát z/do excelu
df = pd.read_excel("../../input/mtcars/test.xlsx") #Načítanie dát z excelu
df.to_excel("../../output/function_showcase.xlsx", index=False) #Uloženie dát do excelu

#Základné operácie s DataFrame
print(df.head(5)) #Prvých 5 riadkov
print(df.tail(5)) #Posledných 5 riadkov
print(df.info()) #Informácie o dátach
print(df.describe()) #Štatistický prehľad číselných stĺpcov

#Výber a filtrovanie dát
print(df['x']) #Výber stĺpca
print(df[['x', 'y']]) #Výber viacerých stĺpcov
print(df.loc[0, ['x', 'y']]) #Výber konkrétneho riadku a stĺpca
print(df[df['x'] > 5]) #Filtrovanie dát podľa podmienky

#Manipulácia s dátami
df['z'] = df['x'] + df['y'] #Vytvorenie nového stĺpca
df.drop('z', axis=1, inplace=True) #Odstránenie stĺpca
df.rename(columns={'x': 'a', 'y': 'b'}) #Premenovanie stĺpcov

#Práca s chýbajúcimi hodnotami
df.isnull().sum() #Počet chýbajúcich hodnôt
df.dropna() #Odstránenie riadkov s chýbajúcimi hodnotami
df.fillna("NaN", inplace=True) #Nahradenie chýbajúcich hodnôt

import numpy as np
#Vytvorenie Numpy poľa
np.array([1, 2, 3]) #Vytvorenie 1D poľa (vektora)
np.array([[1 , 2], [3, 4]]) #Vytvorenie 2D poľa (matice)
np.zeros((3, 3)) #Vytvorenie 3x3 matice núl
np.ones((2, 2)) #Vytvorenie 2x2 matice jednotiek
np.eye(3) #Vytvorenie 3x3 jednotkovej matice
np.arange(0, 10, 2) #Vytvorenie poľa od 0 do 10 s krokom 2

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#Základné operácie s Numpy poľami
np.mean(arr) #Priemer hodnôt v poli
np.sum(arr) #Súčet hodnôt v poli
np.min(arr) #Minimálna hodnota v poli
np.max(arr) #Maximálna hodnota v poli
np.std(arr) #Štandardná odchýlka hodnôt v poli

#Mateatické operácie s Numpy poľami
np.dot(arr, arr)  #Skalárny súčin matíc
np.transpose(arr) #Transponovanie matice
np.linalg.inv(arr) #Inverzia matice
np.linalg.det(arr) #Determinant matice
np.linalg.eig(arr) #Vlastné čísla a vektory matice

import statsmodels.api as sm
#Generovanie náhodných dát
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.rand(100, 1)

#Pridanie konštanty (intercept)
X = sm.add_constant(X)

#Lineárna regresia metódou najmenších štvorcov
model = sm.OLS(y, X).fit()
print(model.summary())

import scipy.stats as stats
#Vytvorenie F distribúcie
df1 = 2
df2 = 50
F = stats.f.ppf(0.95, df1, df2) #Hodnota kvantilu pre danú pravdepodobnosť
#Výpočet p-hodnoty
p_value = 1 - stats.f.cdf(F, df1, df2) #P-hodnota pre danú hodnotu F
print(F, p_value)

import matplotlib.pyplot as plt
#Vizualizácia dát
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.plot(x, y, label='y = x^2', color='blue')
plt.title('Čiarový graf')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
#Vizualizácia dát
data = np.random.rand(10, 12)

#Vytvorenie heatmapy
sns.heatmap(data, cmap='coolwarm')
plt.title('Heatmapa')
plt.show()