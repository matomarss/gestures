Popis súborov:

data_extraction.py -> extrakcia zaznamenaných dát z ich formátov

data_visualization.py -> vizualizácia dát (grafy, matice, tabuľky)

files_organization.py -> pomocné metódy pre ukladanie/načítavanie súborov

parameters_evaluation-> metódy zabezpečujúce vyhľadávanie parametrov ako kernel, škálovanie, predspracovanie, PCA a iné hyperparametre

train_classifier-> metódy využívané v súbore parameters_evaluation.py zabezpečujúce trénovanie a testovanie modelov, grid_search a načítavanie dát

Zvyšné .py súbory neboli mnou modifikované a pre projekt nie sú veľmi relevantné. 

Main metóda súboru parameters_evaluation.py obsahuje v komentároch všetky/väčšinu metód, ktoré som spúšťal pri hľadaní parametrov, zaznamenávaní výsledkov, ich vizualizácii a vyhodnocovaní počas vypracovávania projektu.


test_pca_... priečinky obsahujú výsledky validácie pre kombinácie škálovania, predspracovania, PCA a klasifikátorov pre dané n.

test_svm_kernels obsahuje výsledky validácie pri testovaní pre výber kernelu pre porovnávané kombinácie

test_hyper_parameters obsahuje výsledky testovania hyperparametrov modelov SVM a RF pre dané n a danú najúspešnejšiu kombináciu škálovania, predspracovania a PCA


Súbor obsahujúci spracovávané dáta záznamov gest je možné nájsť tu: http://cogsci.dai.fmph.uniba.sk/~kocur/gestures/

Pre správne fungovanie metód spúšťaných vo funkcii main v súbore parameters_evaluation.py je potrebné umiestniť súbor s dátami do priečinka, ktorý obsahuje aj priečinok s kódom z tejto stránky a premenovať ho na "gestures".