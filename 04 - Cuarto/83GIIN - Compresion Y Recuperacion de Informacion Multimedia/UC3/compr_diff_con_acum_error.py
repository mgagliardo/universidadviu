import numpy as np
import matplotlib.pyplot as plt


def encode(x) -> np.ndarray:
    # Encoder
    d = np.zeros_like(x)
    dq = np.zeros_like(x, dtype=int)

    d[0] = x[0] - 0
    dq[0] = int(round(d[0]))

    for i in range(1, len(x)):
        d[i] = x[i] - x[i - 1]
        dq[i] = int(round(d[i]))
    return dq


def decode(x, dq) -> np.ndarray:
    # Decoder (reconstrucción de la señal)
    xqr = np.zeros_like(x)
    xqr[0] = dq[0]
    for i in range(1, len(x)):
        xqr[i] = dq[i] + xqr[i - 1]
    return xqr


def graph(t, x, xqr) -> None:
    # Gráfico
    plt.plot(t, x, label="Original", color="red")
    plt.plot(t, xqr, label="Reconstruida", color="blue", linestyle="--")
    plt.xlabel("Tiempo")
    plt.ylabel("Amplitud")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()


def main():
    fm = 4
    fs = 20 * fm
    am = 2
    t = np.arange(0, 1 + 1 / fs, 1 / fs)  # Incluye el último punto como en R
    x = am * np.cos(2 * np.pi * fm * t)
    dq = encode(x)
    xqr = decode(x, dq)
    graph(t, x, xqr)


if __name__ == "__main__":
    main()
