from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metodos import (Parametros, monte_carlo,
                     turnbull_wakeman, turnbull_wakeman_continuo)

BASE = Path(__file__).resolve().parent
FIG = BASE / "figuras"
FIG.mkdir(exist_ok=True)


def guardar_csv(nombre: str, filas: list[dict]) -> None:
    with (BASE / nombre).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        writer.writerows(filas)


def formatear_graficos() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 160,
    })


def main() -> None:
    formatear_graficos()
    base = Parametros()
    resultados: list[dict] = []
    for sigma in (0.20, 0.40):
        for k in (90.0, 100.0, 110.0):
            for tipo in ("call", "put"):
                p = replace(base, sigma=sigma, k=k)
                tw = turnbull_wakeman(p, tipo)
                mc = monte_carlo(p, tipo, trayectorias=200_000, semilla=2026)
                fila = {
                    "sigma": sigma, "strike": k, "tipo": tipo,
                    "precio_tw": round(tw.precio, 8),
                    "precio_mc": round(mc.precio, 8),
                    "error_absoluto": round(abs(tw.precio - mc.precio), 8),
                    "diferencia_tw_menos_mc": round(tw.precio - mc.precio, 8),
                    "error_relativo_pct": round(100 * (tw.precio - mc.precio) / mc.precio, 4),
                    "error_estandar_mc": round(mc.error_estandar, 8),
                    "ic95_inferior": round(mc.ic_inferior, 8),
                    "ic95_superior": round(mc.ic_superior, 8),
                    "tw_en_ic95": mc.ic_inferior <= tw.precio <= mc.ic_superior,
                    "tiempo_tw_seg": round(tw.segundos, 8),
                    "tiempo_mc_seg": round(mc.segundos, 8),
                    "trayectorias": mc.trayectorias,
                }
                resultados.append(fila)
                print(f"sigma={sigma:.0%} K={k:.0f} {tipo:4s} | "
                      f"TW={tw.precio:.5f} MC={mc.precio:.5f} | "
                      f"EA={fila['error_absoluto']:.5f} | "
                      f"IC95 [{mc.ic_inferior:.5f}, {mc.ic_superior:.5f}]")
    guardar_csv("resultados_comparacion.csv", resultados)

    # Figura 2
    strikes = np.arange(80.0, 120.1, 2.5)
    barrido: list[dict] = []
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=True, layout="constrained")
    for ax, sigma in zip(axs, (0.20, 0.40)):
        for tipo, color in (("call", "#1767a6"), ("put", "#ad6139")):
            rel, banda = [], []
            for k in strikes:
                p = replace(base, sigma=sigma, k=float(k))
                tw = turnbull_wakeman(p, tipo)
                mc = monte_carlo(p, tipo, trayectorias=200_000, semilla=2026)
                rel.append(100 * (tw.precio - mc.precio) / mc.precio)
                banda.append(100 * 1.96 * mc.error_estandar / mc.precio)
                barrido.append({"sigma": sigma, "strike": float(k), "tipo": tipo,
                                "precio_tw": round(tw.precio, 8), "precio_mc": round(mc.precio, 8),
                                "error_relativo_pct": round(rel[-1], 4),
                                "ic95_relativo_pct": round(banda[-1], 4)})
            rel, banda = np.array(rel), np.array(banda)
            ax.plot(strikes, rel, "o-", ms=3.5, color=color, label=tipo.upper())
            ax.fill_between(strikes, rel - banda, rel + banda, color=color, alpha=0.15)
        ax.axhline(0, color="black", lw=0.8)
        ax.set(title=f"Volatilidad {sigma:.0%}", xlabel="Strike (K)")
        ax.grid(alpha=0.2)
    axs[0].set_ylabel("Error relativo (TW − MC)/MC  [%]")
    axs[1].legend(loc="upper left")
    fig.suptitle("Error relativo de Turnbull–Wakeman respecto de Monte Carlo (banda: IC 95% de MC)")
    fig.savefig(FIG / "error_relativo_strikes.png", bbox_inches="tight")
    plt.close(fig)
    guardar_csv("resultados_barrido_strikes.csv", barrido)

    # Figura 3
    fig, ax = plt.subplots(figsize=(9, 4.0), layout="constrained")
    e = np.arange(len(resultados))
    etiquetas = [f"{f['tipo'][0].upper()} K{f['strike']:.0f}\nσ{f['sigma']:.0%}" for f in resultados]
    dif = [f["diferencia_tw_menos_mc"] for f in resultados]
    ax.bar(e, dif, color=["#457c9c" if d >= 0 else "#ad6139" for d in dif])
    ax.errorbar(e, dif, yerr=[1.96 * f["error_estandar_mc"] for f in resultados],
                fmt="none", ecolor="black", capsize=3, lw=1)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xticks=e, xticklabels=etiquetas,
           ylabel="Prima TW − prima MC",
           title="Diferencia con signo TW − MC (barras de error: IC 95% de MC)")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(FIG / "diferencias_con_signo.png", bbox_inches="tight")
    plt.close(fig)

    # trayectorias
    convergencia: list[dict] = []
    for n in (1_000, 5_000, 20_000, 100_000, 400_000):
        mc = monte_carlo(base, "call", trayectorias=n, semilla=2026)
        convergencia.append({"trayectorias": n, "precio_mc": round(mc.precio, 8),
                             "ic95_inferior": round(mc.ic_inferior, 8),
                             "ic95_superior": round(mc.ic_superior, 8),
                             "error_estandar": round(mc.error_estandar, 8),
                             "tiempo_seg": round(mc.segundos, 8)})
    guardar_csv("resultados_convergencia.csv", convergencia)

    # Validacion contra el ejemplo de Hull: S0=K=50, r=10%, sigma=40%, T=1
    precio_c, sigma_a = turnbull_wakeman_continuo(50, 50, 0.10, 0.40, 1.0)
    validacion = [{"metodo": "TW promedio continuo (Hull)", "observaciones": "continuo",
                   "precio": round(precio_c, 6), "sigma_equivalente": round(sigma_a, 6)}]
    hull = Parametros(s0=50, k=50, r=0.10, sigma=0.40, vencimiento=1.0)
    for n in (12, 52, 252):
        tw = turnbull_wakeman(replace(hull, observaciones=n))
        validacion.append({"metodo": "TW discreto", "observaciones": n,
                           "precio": round(tw.precio, 6),
                           "sigma_equivalente": round(np.sqrt(tw.varianza_log / hull.vencimiento), 6)})
    mc = monte_carlo(replace(hull, observaciones=252), "call", trayectorias=400_000, semilla=2026)
    validacion.append({"metodo": f"Monte Carlo (IC95% ±{1.96 * mc.error_estandar:.4f})",
                       "observaciones": 252, "precio": round(mc.precio, 6),
                       "sigma_equivalente": ""})
    guardar_csv("validacion_hull.csv", validacion)
    print("\nValidacion Hull:")
    for f in validacion:
        print(f"  {f['metodo']:38s} n={f['observaciones']!s:9s} precio={f['precio']:.4f}")

    print("\nArchivos creados en:", BASE)


if __name__ == "__main__":
    main()
