﻿# Inštalácia
1. Stiahnite si [Python](https://www.python.org/downloads/) a nainštalujte ho. (verzia 3.13.1), následne treba nastaviť pre projekt Python interpreter
2. Nainštalujte si plugin pre R (priamo v PyCharm), znovu treba nastaviť interpreter pre R
3. Python knižnice by sa mali automaticky nainštalovať z requirements.txt
4. R knižnice treba nainštalovať ručne pomocou scriptu `install_packages.R`
5. Ak chcete spúšťať R scripty v Pythone, tak do activate.bat treba pridať cestu k Rscript.exe, napr. ' SET "PATH=C:\Program Files\R\R-4.4.3\bin;%PATH%" '
