
# Importar paquetes
import argparse, glob, os
import numpy as np
import pandas as pd
from time import perf_counter

# --- Necesario para multiprocessing ---
def job_process_file(fp):
    # Se usa las mismas utilidades que ya se tiene
    a, b = load_pair_csv(fp)
    return pearson_r_np(a, b), rmse_np(a, b)

# --------- Utilidades: carga y métricas ----------
# Reemplazar "FAI" y "CHLA" por tus variables a correlacionar
def load_pair_csv(fp, fai_col="FAI", chla_col="CHLA"):
    # Esto lee columnas necesarias; tolera extras
    usecols = None
    df = pd.read_csv(fp) if usecols is None else pd.read_csv(fp, usecols=usecols)
    # Autodetección mínima de nombres si difieren. Personalizar segun cada caso personal. 
    cols = {c.lower(): c for c in df.columns}
    fai = df[cols.get(fai_col.lower(), list(cols.values())[0])] \
          if fai_col.lower() in cols else df.filter(regex="fai", axis=1).iloc[:,0]
    chla = df[cols.get(chla_col.lower(), list(cols.values())[1])] \
           if chla_col.lower() in cols else df.filter(regex="chla|chl", axis=1, case=False).iloc[:,0]
    a = fai.to_numpy(dtype=np.float32)
    b = chla.to_numpy(dtype=np.float32)
    # Máscara de datos válidos
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]

# Calculo de R
def pearson_r_np(a, b):
    if a.size < 10: return np.nan
    a = a - a.mean()
    b = b - b.mean()
    den = (np.sqrt((a*a).sum()) * np.sqrt((b*b).sum()))
    return float((a*b).sum() / (den + 1e-12))

# Calculo de RMSE
def rmse_np(a, b):
    if a.size < 10: return np.nan
    d = a - b
    return float(np.sqrt(np.mean(d*d)))

# --------- V1: Numpy ----------
def run_v1(files):
    rs, rmses = [], []
    t0 = perf_counter()
    for fp in files:
        a, b = load_pair_csv(fp)
        rs.append(pearson_r_np(a, b))
        rmses.append(rmse_np(a, b))
    total = perf_counter() - t0
    return {"variant": "V1_numpy", "n": len(files), "total_sec": total,
            "mean_sec": total/len(files) if files else np.nan,
            "r_mean": float(np.nanmean(rs)) if rs else np.nan,
            "rmse_mean": float(np.nanmean(rmses)) if rmses else np.nan}

# --------- V2: multiprocessing ----------
def run_v2(files):
    from multiprocessing import Pool, cpu_count
    t0 = perf_counter()
    with Pool(processes=cpu_count()) as pool:
        rows = list(pool.imap_unordered(job_process_file, files))
    total = perf_counter() - t0
    rs    = [r for r, _ in rows]
    rmses = [e for _, e in rows]
    return {"variant": "V2_mproc", "n": len(files), "total_sec": total,
            "mean_sec": total/len(files) if files else np.nan,
            "r_mean": float(np.nanmean(rs)) if rs else np.nan,
            "rmse_mean": float(np.nanmean(rmses)) if rmses else np.nan}

# --------- V3: Numba JIT ----------
def run_v3(files):
    try:
        from numba import njit, prange
    except Exception as e:
        raise SystemExit("Falta numba en el entorno. Instalá 'numba' y reintenta.") from e

    @njit(fastmath=True, parallel=True)
    def pearson_r_numba(a, b):
        n = a.size
        if n < 10: return np.nan
        am = 0.0; bm = 0.0
        for i in prange(n):
            am += a[i]; bm += b[i]
        am /= n; bm /= n
        num = 0.0; ad = 0.0; bd = 0.0
        for i in prange(n):
            x = a[i] - am; y = b[i] - bm
            num += x*y; ad += x*x; bd += y*y
        den = np.sqrt(ad) * np.sqrt(bd) + 1e-12
        return num / den

    @njit(fastmath=True, parallel=True)
    def rmse_numba(a, b):
        n = a.size
        if n < 10: return np.nan
        acc = 0.0
        for i in prange(n):
            d = a[i] - b[i]
            acc += d*d
        return np.sqrt(acc / n)

    # Warm-up para compilar
    if files:
        a0, b0 = load_pair_csv(files[0])
        _ = pearson_r_numba(a0, b0); _ = rmse_numba(a0, b0)

    rs, rmses = [], []
    t0 = perf_counter()
    for fp in files:
        a, b = load_pair_csv(fp)
        rs.append(float(pearson_r_numba(a, b)))
        rmses.append(float(rmse_numba(a, b)))
    total = perf_counter() - t0
    return {"variant": "V3_numba", "n": len(files), "total_sec": total,
            "mean_sec": total/len(files) if files else np.nan,
            "r_mean": float(np.nanmean(rs)) if rs else np.nan,
            "rmse_mean": float(np.nanmean(rmses)) if rmses else np.nan}

# --------- V4: GPU (opcional) ----------
def run_v4(files, device=None):
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def r_gpu(a, b):
        if a.size < 10: return np.nan
        A = torch.from_numpy(a).to(device)
        B = torch.from_numpy(b).to(device)
        A = A - A.mean(); B = B - B.mean()
        den = torch.sqrt((A*A).sum()) * torch.sqrt((B*B).sum()) + 1e-12
        return float((A*B).sum().div(den).item())

    def rmse_gpu(a, b):
        if a.size < 10: return np.nan
        A = torch.from_numpy(a).to(device)
        B = torch.from_numpy(b).to(device)
        return float(torch.sqrt(((A-B)**2).mean()).item())

    rs, rmses = [], []
    t0 = perf_counter()
    with torch.no_grad():
        for fp in files:
            a, b = load_pair_csv(fp)
            rs.append(r_gpu(a, b))
            rmses.append(rmse_gpu(a, b))
    total = perf_counter() - t0
    return {"variant": f"V4_torch_{device}", "n": len(files), "total_sec": total,
            "mean_sec": total/len(files) if files else np.nan,
            "r_mean": float(np.nanmean(rs)) if rs else np.nan,
            "rmse_mean": float(np.nanmean(rmses)) if rmses else np.nan}

# --------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="*.csv", help="glob pattern para los CSVs (uno por par)")
    ap.add_argument("--variant", choices=["v1","v2","v3","v4"], required=True)
    ap.add_argument("--save", default="results.csv", help="Archivo CSV de resultados acumulados")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"No encontré CSVs con pattern={args.pattern}")

    if args.variant == "v1": res = run_v1(files)
    elif args.variant == "v2": res = run_v2(files)
    elif args.variant == "v3": res = run_v3(files)
    elif args.variant == "v4": res = run_v4(files)
    else: raise SystemExit("variant desconocida")

    # Imprimir y guardar
    print(res)
    # Crea results CSV
    import csv
    new = not os.path.exists(args.save)
    with open(args.save, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant","n","total_sec","mean_sec","r_mean","rmse_mean"])
        if new: w.writeheader()
        w.writerow(res)

if __name__ == "__main__":
    main()