import sys
from typing import Union
from dataclasses import dataclass, field


def leer_archivo(archivo: str) -> Union[tuple, tuple, tuple, tuple, list[tuple]]:
    """
    Funcion que lee un archivo y devuelve la totalidad de: estados, estado inicial, estados terminales, alfabeto y posibles transiciones
    :input: La ruta de un archivo como string
    :return: Union de tuplas y una lista
    """
    entrada = None
    aut_transiciones = []
    lineas = [linea.rstrip() for linea in open(archivo, "r").readlines()]
    for linea in lineas:
        if "#" in linea:
            entrada = linea
        else:
            match entrada:
                case "#alfabeto":
                    aut_alfabeto = tuple(linea.split(","))
                case "#inicial":
                    aut_estado_inicial = linea
                case "#terminales":
                    aut_terminales = tuple(linea.split(","))
                case "#transiciones":
                    transicion = tuple(linea.split(","))
                    aut_transiciones.append(transicion)
    return aut_alfabeto, aut_estado_inicial, aut_terminales, aut_transiciones


# Clase base para el AFN
@dataclass
class AFN:
    estado_actual: str
    aut_alfabeto: tuple = field(default_factory=tuple)
    aut_terminales: tuple = field(default_factory=tuple)
    aut_transiciones: list = field(default_factory=list)

    def _name(self):
        return "AFN"

    def _trans_estado(self, estado: str, transiciones: list) -> list:
        """
        Recupera las transiciones salidas de un estado
        :inputs:
            Un estado (por ejemplo "s1")
            Una cantidad de transiciones disponibles
        :return:
            Una lista de transiciones posibles desde el estado ingresado:
            Ejemplo: [('s1', 'a', 's1'), ('s1', 'b', 's1'), ('s1', 'a', 's2')]
        """
        return [trans for trans in transiciones if trans[0] == estado]

    def _disparable(self, simbolo: str, transiciones) -> list:
        """
        Funcion para determinar si en una lista de transiciones, se reconoce un simbolo
        :input:
            Un simbolo aceptado por un alfabeto
            Una lista de transiciones en la forma [Estado Inicial, Simbolo, Estado Final]
        :return:
            Una lista de estados de llegada posibles con ese simbolo
        """
        if not transiciones:
            return []
        return [trans[2] for trans in transiciones if simbolo in trans]

    def _calcular_sig_estados(self, sig_estados: list) -> list:
        """
        Funcion que calcula y devuelve las siguientes transiciones a partir de una lista de estados
        :input: Una lista de estados
        :output: Una lista de transiciones a las cuales llegar desde dichos estados como input
        """
        transiciones_posibles = []
        for sig_trans in sig_estados:
            transf = self._trans_estado(sig_trans, self.aut_transiciones)
            for t in transf:
                transiciones_posibles.append(t)
        return transiciones_posibles

    def _validar(self, char: str):
        return char in self.aut_alfabeto

    def procesar(self, cadena: str):
        # Motor del Automata
        reconocida = True
        reconocida_parcial = (False, None)
        transiciones_posibles = self._trans_estado(
            self.estado_actual, self.aut_transiciones
        )
        # Lectura y loop sobre la cadena ingresada
        for ind, char in enumerate(cadena):
            # Reconoce si el simbolo se encuentra en el alfabeto
            if self._validar(char):

                # Chequea si utilizando el simbolo actual, hay transiciones disponibles
                # Obtiene los proximos estados posibles a partir del caracter actual y las transiciones disponibles
                sig_estados = self._disparable(char, transiciones_posibles)

                # Si no hay estados proximos disponibles a partir del simbolo actual
                # Quiere decir que no es reconocida por el automata
                if not sig_estados:
                    reconocida = False
                    break

                # Si `alguno` de los proximos estados disponibles a partir del simbolo actual es una terminal
                # Y el caracter analizado es el ultimo, entonces la cadena es reconocida
                if any(
                    sig_trans in self.aut_terminales for sig_trans in sig_estados
                ) and (ind == len(cadena) - 1):
                    reconocida = True

                # Si `alguno` de los proximos estados disponibles a partir del simbolo actual es una terminal
                # PERO el caracter analizado NO es el ultimo, entonces la cadena es reconocida parcialmente
                elif any(
                    sig_trans in self.aut_terminales for sig_trans in sig_estados
                ) and (ind != len(cadena) - 1):
                    reconocida_parcial = (True, cadena[: ind + 1])

                # En ultimo caso, la cadena NO es reconocida
                else:
                    reconocida = False

                # A partir de los proximos estados disponibles, se crea una nueva lista
                # Con las proximas transiciones que estaran disponibles a partir de dichos estados
                transiciones_posibles = self._calcular_sig_estados(sig_estados)

            else:
                # El simbolo ingresado NO es reconocido en el alfabeto
                # Pero una parte de la cadena ingresada SI lo fue
                if reconocida_parcial[0]:
                    print(
                        f"La palabra `{cadena}` no ha sido reconocida en su totalidad porque tiene un simbolo que no pertenece al alfabeto {self.aut_alfabeto}, pero `{reconocida_parcial[1]}` ha sido reconocida parcialmente por el automata {self._name()}."
                    )

                # La cadena en su totalidad no fue reconocida en el alfabeto
                else:
                    print(
                        f"El simbolo: `{char}`, no se encuentra en el alfabeto: {self.aut_alfabeto}. Intentelo nuevamente con una palabra diferente"
                    )
                sys.exit(1)

        # La cadena ha sido reconocida en su totalidad
        if reconocida:
            print(
                f"La palabra `{cadena}` ha sido reconocida por el automata {self._name()}."
            )

        # La cadena ha sido reconocida parcialmente
        elif not reconocida and reconocida_parcial[0]:
            print(
                f"La palabra `{cadena}` no ha sido reconocida en su totalidad, pero `{reconocida_parcial[1]}` ha sido reconocida parcialmente por el automata {self._name()}."
            )

        # La cadena no ha sido reconocida ni parcial ni totalmente por el AFD
        else:
            print(
                f"La palabra `{cadena}` NO ha sido reconocida por el automata {self._name()}."
            )


# Clase base para el AFN-ε
@dataclass
class AFNE(AFN):

    _epsilon_symbol: str = "E"

    def _name(self):
        return "AFN-ε"

    def _validar(self, char: str):
        return char in self.aut_alfabeto or None

    def _cerradura_epsilon(self, transiciones: list) -> list:
        """
        Funcion para la cerradura Epsilon
        :input: Una lista de transiciones
        Ejemplo: [('s3', 'c', 's3'), ('s3', 'a', 's4'), ('s3', 'c', 's5')]
        :output: Una lista de estados alcanzables a traves de Epsilon sin consumir de la cadena de caracteres
        """
        if not transiciones:
            return []

        transiciones_epsilon = []
        for trans in transiciones:
            if self._epsilon_symbol in trans:
                transiciones_epsilon.append(trans[2])

        nuevas_trans = []
        if transiciones_epsilon:
            for trans in transiciones_epsilon:
                nuevas_trans = self._trans_estado(trans, self.aut_transiciones)
                return self._cerradura_epsilon(nuevas_trans)

        return transiciones

    def _disparable(self, simbolo: str, transiciones: list) -> list:
        """
        Funcion para determinar si en una lista de transiciones, se reconoce un simbolo
        :input:
            Un simbolo aceptado por un alfabeto
            Una lista de transiciones en la forma [Estado Inicial, Simbolo, Estado Final]
        :return:
            Una lista de estados de llegada posibles con ese simbolo
        """
        if not transiciones:
            return []

        transiciones = transiciones + self._cerradura_epsilon(transiciones)
        trans_disp = []
        if transiciones:
            for trans in transiciones:
                if simbolo == trans[1]:
                    trans_disp.append(trans[2])

        return trans_disp


def main():
    afn = None
    archivo = input("Ingrese un archivo a leer: ")
    if not archivo:
        print("Debe ingresar un archivo")
        sys.exit(1)

    aut_alfabeto, estado_actual, aut_terminales, aut_transiciones = leer_archivo(
        archivo
    )

    if "E" in aut_alfabeto:
        afn = AFNE(estado_actual, aut_alfabeto, aut_terminales, aut_transiciones)
    else:
        afn = AFN(estado_actual, aut_alfabeto, aut_terminales, aut_transiciones)

    cadena = input("Ingrese la cadena a ser reconocida por el automata: ")
    afn.procesar(cadena)


if __name__ == "__main__":
    main()
