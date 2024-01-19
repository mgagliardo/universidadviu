import sys
from graphviz import Digraph
from copy import deepcopy


class Automata:
    """
    Clase automata
    """
    def __init__(self, estados=None, estados_finales=None, trans_cero=None, trans_uno=None):
        self.estados = estados
        self.estados_finales = estados_finales
        self.trans_cero = trans_cero
        self.trans_uno = trans_uno

    def visualizar(self):
        """
        Funcion para graficar todos los estados y transiciones de un automata
        :return: Un grafico
        """
        grafico = Digraph()

        for estado in self.estados:
            if estado in self.estados_finales:
                grafico.node(name=str(estado), label=str(estado), shape='doublecircle')
            else:
                grafico.node(name=str(estado), label=str(estado), shape='circle')

        for tail, head in zip(self.estados, self.trans_cero):
            grafico.edge(tail_name=str(tail), head_name=str(head), label='0')

        for tail, head in zip(self.estados, self.trans_uno):
            grafico.edge(tail_name=str(tail), head_name=str(head), label='1')

        return grafico


def minimizar_automata(automata, estados_equivalentes) -> Automata:
    """
    Funcion que recibe un automata y una lista de sus estados equivalentes y devuelve un automata minimizado
    :input: Un automata y una lista de sus estados equivalentes
    :return: Un automata minimizado
    """

    # Se copia enteramente el automata a fin de no modificarlo
    minimo_automata = deepcopy(automata)

    # Loop donde obtenemos un nuevo set de transferencias equivalentes utilizando los estados equivalentes
    for estado_equivalente in estados_equivalentes:
        for estado in estado_equivalente:
            
            if minimo_automata.trans_cero[estado] in estado_equivalente:
                minimo_automata.trans_cero[estado] = estado
            else:
                minimo_automata.trans_cero[estado] = min(estados_equivalentes[[estados_equivalentes.index(estado_equivalente) for estado_equivalente in estados_equivalentes if minimo_automata.trans_cero[estado] in estado_equivalente][0]])

            if minimo_automata.trans_uno[estado] in estado_equivalente:
                minimo_automata.trans_uno[estado] = estado
            else:
                minimo_automata.trans_uno[estado] = min(estados_equivalentes[[estados_equivalentes.index(estado_equivalente) for estado_equivalente in estados_equivalentes if minimo_automata.trans_uno[estado] in estado_equivalente][0]])

    # Se marcan los estados "sobrantes" con un -1
    for estado_equivalente in estados_equivalentes:
        for estado in estado_equivalente[1:]:
            minimo_automata.trans_cero[estado] = -1
            minimo_automata.trans_uno[estado] = -1
            # Si el estado es un `sobrante`` lo removemos de los estados finales
            if estado in minimo_automata.estados_finales:
                minimo_automata.estados_finales.remove(estado)
            minimo_automata.estados[minimo_automata.estados.index(estado)] = -1


    # Filtramos los estados (de la lista de estados y de transacciones) que no estan marcados con un `-1`.
    # Aquellos que estan marcados con "-1" los vamos a remover.
    # En este paso nos quedamos con una lista definitiva de estados pero todavia no los reemplazamos
    not_deleted = lambda x: x != -1
    minimo_automata.estados = list(filter(not_deleted, minimo_automata.estados))
    minimo_automata.trans_cero = list(filter(not_deleted, minimo_automata.trans_cero))
    minimo_automata.trans_uno = list(filter(not_deleted, minimo_automata.trans_uno))

    # Finalmente, reemplazamos cada estado por otro usando los indices de la lista
    # Asi por ejemplo a partir de una lista de estados que queremos de nuestro antiguo automata, por ejemplo: [0, 2, 4, 5, 7]
    # Nos quedaremos con una nueva lista reemplazada por sus indices: [0, 1, 2, 3, 4]
    for i, estado in enumerate(minimo_automata.estados):
        if i < len(minimo_automata.estados_finales):
            minimo_automata.estados_finales[i] = minimo_automata.estados.index(minimo_automata.estados_finales[i])
        minimo_automata.trans_cero[i] = minimo_automata.estados.index(minimo_automata.trans_cero[i])
        minimo_automata.trans_uno[i] = minimo_automata.estados.index(minimo_automata.trans_uno[i])
    minimo_automata.estados = list(range(len(minimo_automata.estados)))

    return minimo_automata


def get_estados_equivalentes(automata) -> list:
    """
    Funcion que recibe un automata calcula los estados equivalentes (reemplazables) utilizando el algoritmo clasico
    :input: Un automata
    :return: Lista de Union de tuplas y una lista
    """

    estados_equivalentes = []

    estados_marcados = []

    # Recorro los estados del automata
    for estado in automata.estados:
        
        # Si el estado es final lo agrego a una lista marcada con un `1`
        if estado in automata.estados_finales:
            estados_marcados.append((1, (None, None)))
        
        # Si el estado no es final lo agrego a una lista marcada con un `0`
        else:
            estados_marcados.append((0, (None, None)))

    cant_estados_antiguos = len(set(estados_marcados))

    while True:
        for estado, ind in enumerate(estados_marcados):
            estados_marcados[estado] = (ind[0], (estados_marcados[automata.trans_cero[estado]][0], estados_marcados[automata.trans_cero[estado]][0]))

        marcas_unicas = list(set(estados_marcados))
        for estado, ind in enumerate(estados_marcados):
            estados_marcados[estado] = (marcas_unicas.index(ind), (None, None))

        if len(marcas_unicas) == cant_estados_antiguos:
            for clase_equivalente in range(len(marcas_unicas)):
                estados_equivalentes.append([estado for estado, ind in enumerate(estados_marcados) if ind[0] == clase_equivalente])
            break

        cant_estados_antiguos = len(marcas_unicas)

    return estados_equivalentes


def leer_archivo(archivo: str) -> Automata:
    """
    Funcion que lee un archivo y devuelve un Automata
    :input: La ruta de un archivo como string
    :return: Un Automata
    """
    automata = Automata()

    entrada = None
    aut_transiciones = []
    lineas = [linea.rstrip() for linea in open(archivo, "r").readlines()]
    for linea in lineas:
        if "#" in linea:
            entrada = linea
        else:
            match entrada:
                case "#estados":
                    estados = [int(estado) for estado in linea.split(",")]
                case "#terminales":
                    terminales = [int(terminal) for terminal in linea.split(",")]
                case "#transiciones":
                    transicion = tuple(linea.split(","))
                    aut_transiciones.append(transicion)

    # Ordeno las transiciones por el primer elemento (estado de partida de la transicion)
    aut_transiciones = sorted(aut_transiciones, key=lambda x: x[0])
    automata.estados = estados
    automata.estados_finales = terminales
    automata.trans_cero = [int(trans[2]) for trans in aut_transiciones if "0" == trans[1]]
    automata.trans_uno = [int(trans[2]) for trans in aut_transiciones if "1" == trans[1]]

    return automata


def main():
    archivo = input("Ingrese un archivo a leer: ")
    if not archivo:
        print("Debe ingresar un archivo")
        sys.exit(1)

    automata_inicial = leer_archivo(archivo)
    estados_equivalentes = get_estados_equivalentes(automata_inicial)
    automata_minimo = minimizar_automata(automata_inicial, estados_equivalentes)
    automata_inicial.visualizar().render('./inicial.gv', view=True)
    automata_minimo.visualizar().render('./minimo.gv', view=True)


if __name__ == "__main__":
    main()
