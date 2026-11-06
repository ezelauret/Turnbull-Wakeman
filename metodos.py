from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Literal

import numpy as np
from scipy.stats import norm

TipoOpcion = Literal["call", "put"]


@dataclass(frozen=True)
class Parametros:
    s0: float = 100.0
    k: float = 100.0
    r: float = 0.05
    sigma: float = 0.20
    vencimiento: float = 1.0
    observaciones: int = 12
    inicio_promedio: float = 0.0

    def __post_init__(self) -> None:
        if self.s0 <= 0 or self.k <= 0:
            raise ValueError("El spot y el strike deben ser positivos.")
        if self.vencimiento <= 0:
            raise ValueError("El vencimiento debe ser mayor a cero.")
        if not isinstance(self.observaciones, int) or self.observaciones < 1:
            raise ValueError("La cantidad de observaciones debe ser un entero positivo.")
        if self.sigma < 0:
            raise ValueError("La volatilidad no puede ser negativa.")
        if not (0 <= self.inicio_promedio < self.vencimiento):
            raise ValueError("El inicio del promedio debe ser >= 0 y < vencimiento.")

    def tiempos(self) -> np.ndarray:
        """Tiempos t_i = inicio + i*(T-inicio)/n"""
        return np.linspace(self.inicio_promedio, self.vencimiento,
                           self.observaciones + 1, dtype=float)[1:]


def _validar_tipo(tipo: TipoOpcion) -> None:
    if tipo not in ("call", "put"):
        raise ValueError("El tipo debe ser 'call' o 'put'.")


def momentos_promedio(p: Parametros) -> tuple[float, float]:
    t = p.tiempos()
    m1 = p.s0 * np.exp(p.r * t).mean()
    tj, tk = np.meshgrid(t, t, indexing="ij")
    m2 = (p.s0**2 * np.exp(p.r * (tj + tk) + p.sigma**2 * np.minimum(tj, tk))).mean()
    return float(m1), float(m2)


@dataclass(frozen=True)
class ResultadoTW:
    precio: float
    media: float
    varianza: float
    varianza_log: float
    segundos: float


def turnbull_wakeman(p: Parametros, tipo: TipoOpcion = "call") -> ResultadoTW:
    _validar_tipo(tipo)
    inicio = perf_counter()
    m1, m2 = momentos_promedio(p)
    var = max(0.0, m2 - m1**2)
    v = max(0.0, float(np.log1p(var / m1**2)))
    factor = float(np.exp(-p.r * p.vencimiento))
    if v < 1e-14:
        payoff = max(m1 - p.k, 0.0) if tipo == "call" else max(p.k - m1, 0.0)
        precio = factor * payoff
    else:
        raiz_v = float(np.sqrt(v))
        d1 = (np.log(m1 / p.k) + 0.5 * v) / raiz_v
        d2 = d1 - raiz_v
        if tipo == "call":
            precio = factor * (m1 * norm.cdf(d1) - p.k * norm.cdf(d2))
        else:
            precio = factor * (p.k * norm.cdf(-d2) - m1 * norm.cdf(-d1))
    return ResultadoTW(float(precio), m1, var, v, perf_counter() - inicio)


def turnbull_wakeman_continuo(s0: float, k: float, r: float, sigma: float,
                              vencimiento: float, tipo: TipoOpcion = "call") -> tuple[float, float]:
    _validar_tipo(tipo)
    t = vencimiento
    m1 = (np.exp(r * t) - 1) / (r * t) * s0
    m2 = (2 * np.exp((2 * r + sigma**2) * t) * s0**2
          / ((r + sigma**2) * (2 * r + sigma**2) * t**2)
          + 2 * s0**2 / (r * t**2) * (1 / (2 * r + sigma**2) - np.exp(r * t) / (r + sigma**2)))
    sigma_a = float(np.sqrt(np.log(m2 / m1**2) / t))
    d1 = (np.log(m1 / k) + 0.5 * sigma_a**2 * t) / (sigma_a * np.sqrt(t))
    d2 = d1 - sigma_a * np.sqrt(t)
    if tipo == "call":
        precio = np.exp(-r * t) * (m1 * norm.cdf(d1) - k * norm.cdf(d2))
    else:
        precio = np.exp(-r * t) * (k * norm.cdf(-d2) - m1 * norm.cdf(-d1))
    return float(precio), sigma_a


@dataclass(frozen=True)
class ResultadoMC:
    precio: float
    error_estandar: float
    ic_inferior: float
    ic_superior: float
    segundos: float
    trayectorias: int


def _precio_trayectoria(p: Parametros, z: np.ndarray) -> np.ndarray:
    tiempos = p.tiempos()
    dt = np.diff(np.r_[0.0, tiempos])
    log_cambios = (p.r - 0.5 * p.sigma**2) * dt + p.sigma * np.sqrt(dt) * z
    return p.s0 * np.exp(np.cumsum(log_cambios, axis=1))


def monte_carlo(
    p: Parametros,
    tipo: TipoOpcion = "call",
    trayectorias: int = 200_000,
    semilla: int = 2026,
    tam_bloque: int = 10_000,
) -> ResultadoMC:
    _validar_tipo(tipo)
    if not isinstance(trayectorias, int) or trayectorias < 4 or trayectorias % 2 != 0:
        raise ValueError("'trayectorias' debe ser un entero par de al menos 4.")
    if tam_bloque < 1:
        raise ValueError("'tam_bloque' debe ser positivo.")
    inicio = perf_counter()
    rng = np.random.default_rng(semilla)
    parejas = trayectorias // 2
    suma, suma_cuadrados = 0.0, 0.0
    desc = np.exp(-p.r * p.vencimiento)

    for desde in range(0, parejas, tam_bloque):
        cant = min(tam_bloque, parejas - desde)
        z = rng.standard_normal((cant, p.observaciones))
        a_positiva = _precio_trayectoria(p, z).mean(axis=1)
        a_negativa = _precio_trayectoria(p, -z).mean(axis=1)
        if tipo == "call":
            pagos = 0.5 * (np.maximum(a_positiva - p.k, 0.0)
                           + np.maximum(a_negativa - p.k, 0.0))
        else:
            pagos = 0.5 * (np.maximum(p.k - a_positiva, 0.0)
                           + np.maximum(p.k - a_negativa, 0.0))
        descuentos = desc * pagos
        suma += float(descuentos.sum())
        suma_cuadrados += float(np.square(descuentos).sum())

    estimacion = suma / parejas
    var_muestral = max(0.0, (suma_cuadrados - parejas * estimacion**2) / (parejas - 1))
    error = float(np.sqrt(var_muestral / parejas))
    margen = 1.96 * error
    return ResultadoMC(estimacion, error, estimacion - margen,
                       estimacion + margen, perf_counter() - inicio, trayectorias)


def simular_trayectorias(p: Parametros, cantidad: int = 30, semilla: int = 99) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(semilla)
    z = rng.standard_normal((cantidad, p.observaciones))
    precios = _precio_trayectoria(p, z)
    tiempos = np.r_[0.0, p.tiempos()]
    precios = np.column_stack([np.full(cantidad, p.s0), precios])
    return tiempos, precios


if __name__ == "__main__":
    p = Parametros()
    print("Ejemplo basico: S0=100, K=100, r=5%, sigma=20%, T=1, n=12")
    for tipo in ("call", "put"):
        tw = turnbull_wakeman(p, tipo)
        mc = monte_carlo(p, tipo)
        print(f"{tipo.upper():4s} | TW={tw.precio:.6f} | MC={mc.precio:.6f} "
              f"| IC95%=[{mc.ic_inferior:.6f}, {mc.ic_superior:.6f}]")
