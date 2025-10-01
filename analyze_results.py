# Importar paquetes
import csv, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Función results.csv
def main(path="results.csv"):
    df = pd.read_csv(path)
    # Tomamos la/s fila/s de V1 como baseline (por si se corre varias veces)
    base = df[df["variant"].str.contains("V1", case=False, regex=True)]
    if base.empty:
        sys.exit("No encontré una fila baseline (V1) en results.csv. Corré primero la V1.")
    base_mean = base["mean_sec"].mean()

    # Calculamos speedup contra el baseline
    df["speedup"] = base_mean / df["mean_sec"]

    # Imprimimos tabla ordenada por mean_sec asc
    out = df.sort_values("mean_sec")[["variant","n","mean_sec","speedup","r_mean","rmse_mean"]]
    print("\nResultados (ordenados por tiempo medio):\n")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Gráfico de barras simple con mean_sec
    plt.figure()
    plt.bar(df["variant"], df["mean_sec"])
    plt.ylabel("Tiempo medio por archivo (s)")
    plt.xlabel("Variante")
    plt.title("Comparación de tiempos por variante")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("results_times.png", dpi=150)

    # Gráfico de speedup
    plt.figure()
    plt.bar(df["variant"], df["speedup"])
    plt.ylabel("Speedup vs V1")
    plt.xlabel("Variante")
    plt.title("Speedup relativo (V1 como baseline)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("results_speedup.png", dpi=150)

    print("\n✅ Guardé figuras: results_times.png y results_speedup.png")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    main(path)
