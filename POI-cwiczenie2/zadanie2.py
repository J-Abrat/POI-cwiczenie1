import random
import math
import csv
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import pyransac3d as pyrsc


# Generowanie 3 rozłącznych chmur do jednego pliku .xyz
def generuj_dane(plik="chmury_razem.xyz"):
    pts = []

    # Płaszczyzna pozioma
    for _ in range(1000):
        pts.append((random.uniform(0, 10), random.uniform(0, 10), 0.0))

    # Płaszczyzna pionowa
    for _ in range(1000):
        pts.append((random.uniform(30, 40), 0.0, random.uniform(0, 10)))

    # Cylinder
    for _ in range(1000):
        kat = random.uniform(0, 2 * math.pi)
        x = 60.0 + 5.0 * math.cos(kat)
        y = 5.0 * math.sin(kat)
        z = random.uniform(0, 10)
        pts.append((x, y, z))

    with open(plik, 'w') as f:
        for x, y, z in pts:
            f.write(f"{x} {y} {z}\n")


# Implementacja algorytmu RANSAC
def ransac_plaszczyzna(pts, iteracje=200, prog=0.1):
    best_n = None
    best_d = 0
    max_ok = 0

    for _ in range(iteracje):
        p1, p2, p3 = pts[random.sample(range(len(pts)), 3)]
        n = np.cross(p2 - p1, p3 - p1)
        norma = np.linalg.norm(n)

        if norma == 0:
            continue

        n = n / norma
        d = -np.dot(n, p1)

        odl = np.abs(np.dot(pts, n) + d)
        liczba_ok = np.sum(odl < prog)

        if liczba_ok > max_ok:
            max_ok = liczba_ok
            best_n = n
            best_d = d

    sr_odl = np.mean(np.abs(np.dot(pts, best_n) + best_d))
    return best_n, sr_odl


# Ocena chmury i wypisanie wyników
def ocen(nr, n, sr_odl):
    print(f"Chmura nr {nr}:")
    print(f"  Wektor normalny: [{n[0]:.2f}, {n[1]:.2f}, {n[2]:.2f}]")
    print(f"  Średnia odległość: {sr_odl:.4f}")

    if sr_odl < 0.5:
        if abs(n[2]) > 0.9:
            print("  -> Wynik: PŁASZCZYZNA POZIOMA")
        elif abs(n[2]) < 0.1:
            print("  -> Wynik: PŁASZCZYZNA PIONOWA")
        else:
            print("  -> Wynik: PŁASZCZYZNA UKOŚNA")
    else:
        print("  -> Wynik: NIE JEST PŁASZCZYZNĄ (cylinder)")


# --- GŁÓWNY PROGRAM ---
generuj_dane("chmury_razem.xyz")

# Wczytanie pliku .xyz przez csv
dane = []
with open("chmury_razem.xyz", "r") as f:
    for row in csv.reader(f, delimiter=" "):
        if row:
            dane.append([float(row[0]), float(row[1]), float(row[2])])

chmura = np.array(dane)

# Część 1: K-średnich + własny RANSAC
print("=== K-ŚREDNICH + WŁASNY RANSAC ===")
grupy_km = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(chmura)

for i in range(3):
    sub = chmura[grupy_km == i]
    n, sr_odl = ransac_plaszczyzna(sub)
    ocen(i + 1, n, sr_odl)

# Część 2: DBSCAN + pyransac3d
print("\n=== DBSCAN + PYRANSAC3D ===")
grupy_db = DBSCAN(eps=2.0, min_samples=10).fit_predict(chmura)

for g in set(grupy_db):
    if g == -1:
        continue
    sub = chmura[grupy_db == g]

    eq, _ = pyrsc.Plane().fit(sub, thresh=0.1, maxIteration=1000)
    n = np.array(eq[:3])
    norma = np.linalg.norm(n)
    n = n / norma
    d = eq[3] / norma

    sr_odl = np.mean(np.abs(np.dot(sub, n) + d))
    ocen(g + 1, n, sr_odl)