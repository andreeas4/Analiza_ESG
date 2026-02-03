
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hclust
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import fcluster
import os

def nan_replace(tabel):
    for col in tabel.columns:
        if tabel[col].isna().any():
            if is_numeric_dtype(tabel[col]):
                tabel[col].fillna(tabel[col].mean(), inplace=True)
            else:
                tabel[col].fillna(tabel[col].mode()[0], inplace=True)

def partitie(h, nr_clusteri, p, instante):
    """
    Extragem clusteri sectionand dendrograma la pragul corespunzator lui n_clusteri
    h = matricea de legaturi (linkage matrix)
    """
    # pozitia sectionarii in linkage matrix (matricea de legaturi)
    k_diff = p - nr_clusteri

    # pragul unde sectionam dendrograma
    
    if k_diff < 0 or k_diff >= len(h):
        prag = h[-1, 2] # fallback
    else:
        # formula originala presupune k_diff+1 exista, deci k_diff < len(h)-1
        if k_diff + 1 < len(h):
            prag = (h[k_diff, 2] + h[k_diff+1, 2]) / 2
        else:
            prag = h[k_diff, 2] # fallback daca suntem la ultimul pas

    # desenare
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(f"Clusterizare ierarhica (Ward) - {nr_clusteri} clusteri")
    hclust.dendrogram(h, labels=instante, ax=ax, color_threshold=prag)
    
    # Salvare grafic
    plt.savefig(f"Dendrograma_{nr_clusteri}_clusteri.png")
   

    # numar observatii
    n = p + 1

    # initializam labels
    c = np.arange(n)

    # simulare a reuniunilor intre clusteri
    for i in range(n - nr_clusteri):
        k1 = int(h[i, 0])
        k2 = int(h[i, 1])
        c[c == k1] = n + i
        c[c == k2] = n + i

    coduri = pd.Categorical(c).codes
    return np.array([f"C{cod+1}" for cod in coduri])

def histograma(x, variabila, partitia):
    clusters = np.unique(partitia)
    n_clusters = len(clusters)
    
    fig, axs = plt.subplots(1, n_clusters, figsize=(15, 4), sharey=True)
    if n_clusters == 1:
        axs = [axs]
        
    fig.suptitle(f"Histograme pentru variabila: {variabila}")

    for ax, cluster in zip(axs, clusters):
        cluster_data = x[partitia == cluster]
        ax.hist(cluster_data, bins=10, rwidth=0.9, color='skyblue', edgecolor='black')
        ax.set_title(f"Cluster {cluster}\n(n={len(cluster_data)})")
    
    plt.tight_layout()
    plt.savefig(f"Histograma_{variabila}.png")
    # plt.show()

def execute():
    
    filepath = "data\processed\date_standardizate.csv"
    
    print(f"Incarcare date din: {filepath}")
    if not os.path.exists(filepath):
        print("Eroare: Fisierul nu exista. Verificati calea.")
        return

    tabel = pd.read_csv(filepath)
    
    # Setare index pe Symbol daca exista, pentru etichete in dendrograma
    if 'Symbol' in tabel.columns:
        tabel.set_index('Symbol', inplace=True)
    
    instante = list(tabel.index)
    
    # Selectia variabilelor numerice conform analizei din notebook
    cols_numerice = ['governanceScore', 'Stock_return', 'marketCap', 'environmentScore']
    
    # Verificam daca coloanele exista
    missing_cols = [c for c in cols_numerice if c not in tabel.columns]
    if missing_cols:
        print(f"Atentie: Coloanele {missing_cols} lipsesc din dataset. Se vor folosi doar cele existente.")
        cols_numerice = [c for c in cols_numerice if c in tabel.columns]
        
    variabile = cols_numerice
    
  

    # Extragere date
    x = tabel[variabile].values


   
    scaler = StandardScaler()


    x_std = scaler.fit_transform(x)

   
    print("Se calculeaza matricea de legaturi (Linkage Matrix)...")
    h = hclust.linkage(x_std, method='ward')
    
    
    n = len(instante)
    p = n - 1

    # Generam si exportam partitii variate
    clusteri = [2, 3, 4, 5]

    for k in clusteri:
        print(f"Generare partitie cu {k} clusteri...")
        part_k = partitie(h, k, p, instante)
        
        # Salvare in CSV
        part_k_df = pd.DataFrame(data={"Cluster": part_k}, index=instante)
        output_file = f"Partitie_{k}_clusteri_script.csv"
        part_k_df.to_csv(output_file)
        print(f"  -> Salvat in {output_file}")

    # Alegerea numarului optim de clusteri (Metoda saltului maxim)
    # h[1:, 2] sunt distantele. h[:-1, 2] sunt distantele anterioare.
    # Cautam saltul maxim in distanta de unire.
    # Indicii linkage sunt 0..n-2. 
    if len(h) > 2:
        dist_diff = h[1:, 2] - h[:-1, 2]
        k_diff_max = np.argmax(dist_diff)
        # nr_clusteri = p - k_diff_max 
        # Pasul i (0..n-2) reduce nr clusteri de la n la 1.
        # Ward linkage: pasul 0 uneste 2 pct -> n-1 clusteri. Pasul n-2 -> 1 cluster.
        # Deci pasul k_diff_max este unde avem saltul.
        # Numar clusteri = n - (i+1) = n - i - 1. 
        # Deci nr_clusteri = n - 1 - k_diff_max.
        
        nr_clusteri_optim = p - k_diff_max
        print(f"\nNumar optim de clusteri determinat automat: {nr_clusteri_optim}")
    else:
        nr_clusteri_optim = 3 # fallback
        print(f"\nNumar optim de clusteri (fallback): {nr_clusteri_optim}")

    # Extractie partitie optima
    print(f"Extragere partitie optima ({nr_clusteri_optim})...")
    partitie_optima = partitie(h, nr_clusteri_optim, p, instante)
    
    # Salvare finala optima
    pd.DataFrame({"Cluster": partitie_optima}, index=instante).to_csv("Partitie_Optima_Script.csv")

    # Extractie automata (fcluster) - pentru comparatie
    auto = fcluster(h, nr_clusteri_optim, criterion='maxclust')
   

    # Grafice histograme pentru variabilele numerice pe partitia optima
    print("Generare histograme pentru variabile...")
    for i in range(min(len(variabile), x.shape[1])):
        histograma(x[:, i], variabile[i], partitie_optima)

    # Optional: Dendrograma complete linkage
    print("Generare dendrograma Complete Linkage...")
    h_complete = hclust.linkage(x_std, method='complete')
    plt.figure(figsize=(12, 8))
    plt.title('Dendrograma - complete linkage')
    hclust.dendrogram(h_complete, labels=instante)
    plt.savefig("Dendrograma_Complete_Linkage.png")
    
    print("\nExecutie finalizata. Fisierele CSV si PNG au fost salvate.")

if __name__ == '__main__':
    execute()
