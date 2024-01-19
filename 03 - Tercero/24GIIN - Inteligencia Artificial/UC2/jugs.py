# El problema nos da 3 jarros con capacidades de 12, 8 y 3 galones, y un grifo de agua
# Nos pide llenar los jarros o vaciarlos de uno a otro o en el suelo y obtener exactamente un galon
# Obviamente dado que es un algoritmo deberia ser en la menor cantidad de pasos posibles
# La ecuacion a cumplir es 1 < max(A, B, C): Donde A, B y C son los jarros de 12, 8 y 3 galones, respectivamente.

# Las capacidades estan en galones


def transferir(estado, desde, hacia):
    s = estado[:]  # clone
    if s[desde] == 0 or s[hacia] == capacidad[hacia]:
        return s
    libre = capacidad[hacia] - s[hacia]
    if s[desde] <= libre:
        s[hacia] += s[desde]
        s[desde] = 0
    else:
        # `hacia` esta lleno
        s[hacia] = capacidad[hacia]
        s[desde] -= libre
    return s


def get_vecinos(nodo):
    # Devuelve la lista de vecinos de un nodo
    vecinos = []
    t = transferir(nodo, 0, 1)  # de 0 a 1
    if t not in vecinos:
        vecinos.append(t)
    t = transferir(nodo, 0, 2)  # de 0 a 2
    if t not in vecinos:
        vecinos.append(t)
    t = transferir(nodo, 1, 0)  # de 1 a 0
    if t not in vecinos:
        vecinos.append(t)
    t = transferir(nodo, 1, 2)  # de 1 a 2
    if t not in vecinos:
        vecinos.append(t)
    t = transferir(nodo, 2, 0)  # de 2 a 0
    if t not in vecinos:
        vecinos.append(t)
    t = transferir(nodo, 2, 1)  # de 2 a 1
    if t not in vecinos:
        vecinos.append(t)
    return vecinos


def buscar(nodo):
    global solucion
    visitado.append(nodo)

    # Chequeo la solucion
    if nodo == hacia_nodo:
        solucion += 1
        print(f"{solucion} - Ruta: {visitado} - Largo: {len(visitado)}")

    for s in get_vecinos(nodo):
        if s not in visitado:
            buscar(s)
    visitado.pop()


# Defino las capacidades de las jarras
capacidad = [8, 5, 3]
nodo_inicial = [8, 0, 0]
hacia_nodo = [4, 4, 0]
solucion = 0
visitado = []
buscar(nodo_inicial)
print("Finalizado. Se ha encontrado la mejor solucion")
