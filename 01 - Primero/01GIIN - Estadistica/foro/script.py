import statistics
import pandas as pd
import matplotlib.pyplot as plt

from tabulate import tabulate
from operator import itemgetter

def leer_archivo(archivo):
  return pd.read_csv(archivo, encoding='latin-1', sep=';')

def gasto_por_sector(archivo):
  val_per_com = []
  df = leer_archivo(archivo)
  for elem in range(len(df)):
    # Si el valor no se reporta (..) se reemplaza por 0.0
    val = 0.0 if '..' in df['Total'][elem] else round(float(df['Total'][elem].replace(',', '.')), 2)
    # Armado de tuplas (Valor, Nombre de Comunidad)
    val_per_com.append((val, df['Comunidades autónomas'][elem]))
  return val_per_com


gasto_pub_por_com = gasto_por_sector('gasto_publico.csv')
gasto_priv_por_com = gasto_por_sector('gasto_empresas.csv')

max_inv_privada = max(gasto_pub_por_com, key=itemgetter(0))[0]
max_inv_publica = max(gasto_priv_por_com, key=itemgetter(0))[0]

min_inv_privada = min(gasto_pub_por_com, key=itemgetter(0))[0]
min_inv_publica = min(gasto_priv_por_com, key=itemgetter(0))[0]

gasto_per_pib_privado = [elem[0] for elem in gasto_priv_por_com]
gasto_per_pib_publico = [elem[0] for elem in gasto_pub_por_com]

# Mediana
mediana_inv_privada = statistics.median(gasto_per_pib_privado)
mediana_inv_publica = statistics.median(gasto_per_pib_publico)

# Media
media_inv_privada = round(statistics.mean(gasto_per_pib_privado), 2)
media_inv_publica = round(statistics.mean(gasto_per_pib_publico), 2)

# Moda
moda_inv_privada = round(statistics.mode(gasto_per_pib_privado), 2)
moda_inv_publica = round(statistics.mode(gasto_per_pib_publico), 2)

# Desviacion estandar
stdev_inv_privada = round(statistics.stdev(gasto_per_pib_privado), 2)
stdev_inv_publica = round(statistics.stdev(gasto_per_pib_publico), 2)

# Varianza
var_inv_privada = round(statistics.variance(gasto_per_pib_privado), 2)
var_inv_publica = round(statistics.variance(gasto_per_pib_publico), 2)

tabla = [
  ['Tipo', 'Valor'],
  ['Media inversion privada', media_inv_privada],
  ['Media inversion publica', media_inv_publica],
  ['Mediana inversion privada', mediana_inv_privada],
  ['Mediana inversion publica', mediana_inv_publica],
  ['Moda inversion privada', moda_inv_privada],
  ['Moda inversion publica', moda_inv_publica],
  ['Desviacion Estandar inversion privada', stdev_inv_privada],
  ['Desviacion Estandar inversion publica', stdev_inv_publica],
  ['Varianza inversion privada', var_inv_privada],
  ['Varianza inversion publica', var_inv_publica],
]
print(tabulate(tabla, tablefmt='html'))

# Scatter
plt.scatter(*zip(*gasto_pub_por_com))
plt.scatter(*zip(*gasto_priv_por_com))
plt.title("Gasto en I+D por comunidad Autonoma")
plt.xlabel('% Gasto en I+D sobre PIB')
plt.ylabel('Comunidad Autonoma')
plt.legend(['Gasto Publico', 'Gasto Privado'])
plt.show()

# Plot
plt.plot(*zip(*gasto_pub_por_com))
plt.plot(*zip(*gasto_priv_por_com))
plt.title("Gasto en I+D por comunidad Autonoma")
plt.xlabel('% Gasto en I+D sobre PIB')
plt.ylabel('Comunidad Autonoma')
plt.legend(['Gasto Publico', 'Gasto Privado'])
plt.show()
