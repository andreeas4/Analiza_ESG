# 📊 Analiza ESG și Performanța Financiară (S&P 500)

Acest proiect analizează relația dintre scorurile **ESG (Environmental, Social, Governance)** și indicatorii financiari ai companiilor din indicele **S&P 500**, utilizând tehnici de **clustering ierarhic**.

Scopul este identificarea unor **grupuri omogene de companii** pentru a evidenția tipare între sustenabilitate și performanța financiară.

---

## 🎯 Obiectiv

Analiza răspunde la următoarele întrebări:
- Există o legătură între guvernanța corporativă și randamentul bursier?
- Cum se grupează companiile în funcție de scorurile ESG și capitalizarea de piață?
- Pot fi identificate profiluri ESG-financiare distincte?

Metodologia principală utilizată este **Hierarchical Clustering**, folosind metoda **Ward**, cu reprezentare grafică prin dendrograme.

---

## 🧠 Metodologie

- Curățarea și standardizarea datelor (z-score)
- Calculul distanțelor între observații
- Aplicarea clustering-ului ierarhic (Ward)
- Determinarea automată a numărului optim de clusteri
- Analiza distribuției variabilelor pe clusteri

---

## 📂 Structura Proiectului
├── data/
│ └── processed/
│ └── date_standardizate.csv
│
├── notebooks/
│ └── analiza_cluster_script.py
│
├── outputs/ # generat automat
│ ├── Dendrograma_X_clusteri.png
│ ├── Histograma_[Variabila].png
│ └── Partitie_Optima_Script.csv
│
├── requirements.txt
└── README.md

### Descriere
- **data/** – datele utilizate în analiză  
- **notebooks/** – scriptul Python principal  
- **outputs/** – rezultate generate automat (grafice și fișiere CSV)

---

## ⚙️ Cerințe

Bibliotecile necesare sunt listate în `requirements.txt`:
- pandas
- numpy
- scipy
- matplotlib
- scikit-learn

---

## 🚀 Rulare Proiect

### Instalare dependențe
pip install -r requirements.txt
