import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def encode(x) -> np.ndarray:
    # Niveles de cuantización
    niveles = [2, 4, 8, 16, 32, 64, 128, 256]

    # Diccionario para guardar resultados por cada M
    resultados = {}

    for M in niveles:
        delta = Rango / M
        d = np.zeros_like(x)
        dq = np.zeros_like(x)
        xq = np.zeros_like(x)

        # Inicialización
        d[0] = x[0]
        dq[0] = np.floor(d[0] / delta) * delta + delta / 2
        xq[0] = dq[0]

        for i in range(1, len(x)):
            d[i] = x[i] - xq[i - 1]
            dq[i] = np.floor(d[i] / delta) * delta + delta / 2
            xq[i] = dq[i] + xq[i - 1]

        # Guardamos la reconstrucción y el error para este M
        resultados[M] = {
            'xq': xq,
            'error': x - xq
        }
    return Rango

def decode(x, dq, Rango, M):
    # Elegí un M para graficar
    
    delta = Rango / M

    # Reconstrucción
    d = np.zeros_like(x)
    dq = np.zeros_like(x)
    xqr = np.zeros_like(x)

    d[0] = x[0]
    dq[0] = np.floor(d[0] / delta) * delta + delta / 2
    xqr[0] = dq[0]

    for i in range(1, len(x)):
        d[i] = x[i] - xqr[i - 1]
        dq[i] = np.floor(d[i] / delta) * delta + delta / 2
        xqr[i] = dq[i] + xqr[i - 1]

    # Cálculo de SNR y distorsión
    SNR = 10 * np.log10(np.sum(x**2) / np.sum((x - xqr)**2))
    distorsion = np.sum((x - xqr)**2) / len(x)
    return SNR


def graph(t, x, xqr, SNR, M) -> None:
    # Gráfico
    plt.plot(t, x, label='Original', color='red')
    plt.plot(t, xqr, label='Reconstruida', color='blue', linestyle='--')
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud')
    plt.title(f'SNR: {SNR:.3f} dB')
    plt.suptitle(f'Intervalos de cuantificación: {M}', fontsize=10)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def main():
    # Parámetros iniciales
    fm = 4
    fs = 20 * fm
    am = 2
    t = np.arange(0, 1 + 1/fs, 1/fs)
    x = am * np.cos(2 * np.pi * fm * t)
    # Calcular diferencias
    diferencia = x - np.insert(x[:-1], 0, x[0])
    Rango = np.max(diferencia) - np.min(diferencia)
    dq = encode(x)
    M = 16  # Cambiá este valor si querés ver otro nivel
    xqr = decode(x, dq, Rango, M)
    graph(t, x, xqr, SNR, M)


if __name__ == "__main__":
    main()
