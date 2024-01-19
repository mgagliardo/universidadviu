
# coding: utf-8
## Este código pertence a Coursera 
## Curso Resolución de problemas de búsqueda
## https://www.coursera.org/learn/resolucion-busqueda/home/welcome
## de La Universidad Nacional Autónoma de México
## 
## Enseñado por:Stalin Muñoz Gutiérrez, 
## Maestro en Ciencias de la Complejidad
## Centro de Ciencias de la Complejidad
## Esta actividad es una de las actividades del curso, en las que se debia modificar
## este ejercicio es el ejercicio resuelto por Alexandra La Cruz 
## para el curso de Inteligencia Aritificial de la Universidad Internacional de Valencia
## Contiene otros algoritmos de búsqueda que no necesariamente tocaremos en el curso

# # Algoritmos de búsqueda ciega

# Los algoritmos de búsqueda ciega o también denominados como de búsqueda no informada no utilizan información del dominio del problema para guiar la búsqueda. Sus decisiones se basan únicamente en los estados descubiertos desde el inicio de la exploración hasta el momento en que toman una decisión.
# 
# Antes de comenzar con los algoritmos te presentaremos una implementación en Python del rompecabezas del 15 que puede usarse para probar tus algoritmos.

# ## Resolver un problema

# Para resolver un problema vamos a requerir abstraerlo como un grafo de _estados-acciones_.
# Para los algoritmos que veremos en el curso es conveniente definir una clase en python que
# represente un estado o configuración del mundo de nuestro problema a resolver.
# La clase se denominará __EstadoProblema__.
# 

# In[1]:


from abc import ABC, abstractmethod

class EstadoProblema:
    """
    La clase EstadoProblema es abstracta.
    Representa un estado o configuración del problema a resolver.
    
    Es una interfaz simplificada para utilizarse
    en los algoritmos de búsqueda del curso.
    
    Al definir un problema particular hay que implementar los métodos
    abstractos
    """
    
    @abstractmethod
    def expand():
        """
        :return: el conjunto de estados sucesores
        """
        pass
    
    @abstractmethod
    def get_depth():
        """
        :return: la profundidad del estado
        """
        pass
        
    @abstractmethod
    def get_parent():
        """
        :return: referencia al estado predecesor o padre
        """
        pass


# ## El rompecabezas del 15

# Para ilustrar varios de los algoritmos utilizaremos el juego del rompecabezas del 15.
# 
# Para ello hemos preparado una implementación simple en Python, la clase __Puzzle__.
# 
# El juego extenderá de la clase __EstadoProblema__.

# In[2]:


from functools import reduce
import random

# La secuencia del 0 al 15
# 0 representará el espacio en blanco
seq = list(range(0,16))

# Cuatro posibles acciones para nuestro agente
# Mover una ficha en dirección: 
# izquierda (E), derecha (W), arriba (N), o abajo (S)
actions = ['E','W','N','S']

# Representaremos las configuraciones con bits
# Definimos algunas funciones útiles
# Recorre un bloque de 4 bits de unos a la posición i
x_mask = lambda i: 15<<(4*i)

# Extrae los cuatro bits que están en la posción i
# en la configuración c
# El rompecabezas tiene 16 posiciones (16X4 = 64 bits)
extract = lambda i,c: (c&(x_mask(i)))>>(4*i)

# Verifica si la posición z es la última columna
e_most = lambda z: (z%4)==3

# Verifica si la posición z es la primera columna
w_most = lambda z: (z%4)==0

# Verifica si la posición z es el primer renglón
n_most = lambda z: z<=3

# Verifica si la posición z es el último renglón
s_most = lambda z:z>=12

# Establecemos un diccionario con las acciones posibles
# para cada posición del rompecabezas
valid_moves = {i:list(filter(lambda action:((not action=='E') or (not e_most(i))) and ((not action=='W') or (not w_most(i))) and ((not action=='S') or (not s_most(i))) and ((not action=='N') or (not n_most(i))),actions)) for i in seq}

# Realiza el movimiento hacía la izquierda
def move_east(puzzle):
    """
    :param puzzle: el rompecabezas
    """
    if(not e_most(puzzle.zero)):
        puzzle.zero += 1;
        mask = x_mask(puzzle.zero)
        puzzle.configuration =         (puzzle.configuration&mask)>>4 |         (puzzle.configuration&~mask)

# Realiza el movimiento hacía la derecha
def move_west(puzzle):
    if(not w_most(puzzle.zero)):
        puzzle.zero -= 1;
        mask = x_mask(puzzle.zero)
        puzzle.configuration =         (puzzle.configuration&mask)<<4 |         (puzzle.configuration&~mask)

# Realiza el movimiento hacía arriba
def move_north(puzzle):
    if(not n_most(puzzle.zero)):
        puzzle.zero -= 4;
        mask = x_mask(puzzle.zero)
        puzzle.configuration =         (puzzle.configuration&mask)<<16 |         (puzzle.configuration&~mask)

# Realiza el movimiento hacía abajo
def move_south(puzzle):
    if(not s_most(puzzle.zero)):
        puzzle.zero += 4;
        mask = x_mask(puzzle.zero)
        puzzle.configuration =         (puzzle.configuration&mask)>>16 |         (puzzle.configuration&~mask)

class Puzzle(EstadoProblema):
    """
    Rompecabezas del 15
    """
    
    
    def __init__(self, parent=None, action =None, depth=0):
        """
        Puede crearse un rompecabezas ordenado al no especificar
        parámetros del constructor.
        También se puede crear una nueva configuración a 
        partir de una configuración dada en parent.
        :param parent: configuración de referencia.
        :param action: la acción que se aplica a parent para
        generar la configuración sucesora.
        :depth la profundidad del estado a crear
        """
        self.parent = parent
        self.depth = depth
        if(parent == None):
            self.configuration =                  reduce(lambda x,y: x | (y << 4*(y-1)), seq)
            # posición del cero
            self.zero = 15
        else:
            self.configuration = parent.configuration
            self.zero = parent.zero
            if(action != None):
                self.move(action)

    def __str__(self):
        """
        :return: un string que representa 
        la configuración del rompecabezas
        """
        return '\n'+''.join(list(map(lambda i:        format(extract(i,self.configuration)," x")+        ('\n' if (i+1)%4==0 else ''),seq)))+'\n'

    def __repr__(self):
        """
        :return: representación texto de la configuración
        """
        return self.__str__()

    def __eq__(self,other):
        """
        :param other: la otra configuración con la que se comparará
        el objeto
        :return: verdadero cuando el objeto y el parámetro
        tienen la misma configuración.
        """
        return (isinstance(other, self.__class__)) and         (self.configuration==other.configuration)

    def __ne__(self,other):
        """
        :param other: la otra configuración con la que se comparará
        el objeto
        :return: verdadero cuando el objeto y el parámetro
        no tienen la misma configuración
        """
        return not self.__eq__(other)
        
    def __lt__(self,other):
        """
        :param other: la otra configuración con la que se comparará
        el objeto
        :return: verdadero cuando la profundidad del objeto
        es menor que la del argumento
        """
        return self.depth < other.depth

    def __hash__(self):
        """
        :return: un número hash para poder usar la configuración en 
        un diccionario, delegamos al hash de un entero
        """
        return hash(self.configuration)

    def move(self,action):
        """
        Realiza un movimiento de ficha.
        Debemos imaginar que el espacio se mueve en la dirección
        especificada por acción
        :param action: la acción a realizar
        """
        if(action =='E'):
            move_east(self)
        if(action =='W'):
            move_west(self)
        if(action =='N'):
            move_north(self)
        if(action =='S'):
            move_south(self)
        return self


    @staticmethod
    def to_list(puzzle):
        """
        Convertimos la configuración a una lista de números
        :param puzzle: la configuración a convertir
        :return la lista con enteros
        """
        return [extract(i,puzzle.configuration) for i in seq]

    def shuffle(self,n):
        """
        Desordena de manera aleatoria el rompecabezas.
        :param n: el número de movimientos aleatorios a aplicar
        """
        for i in range(0,n):
            self.move(random.choice(valid_moves[self.zero]))
        return self

    def expand(self):
        """
        Los sucesores del estado, quitamos el estado padre
        """
        #filtering the path back to parent
        return list(filter(lambda x:         (x!=self.parent),         [Puzzle(self,action,self.depth+1)         for action in valid_moves[self.zero]]))
    
    def get_depth(self):
        """
        :return: la profundidad del estado
        """
        return self.depth
    
    def get_parent(self):
        """
        :return: el nodo predecesor (padre) del estado 
        """
        return self.parent


# Vamos a crear una instancia de la clase.

# In[3]:


# No indicamos un padre, 
# el rompecabezas estará ordenado
# y su profundidad será cero
puzzle = Puzzle()
print("Configuración:\n",puzzle)
print("Profundidad:\n",puzzle.get_depth())
print("Estado predecesor:\n",puzzle.get_parent())


# Para revolver el estado de manera aleatoria podemos invocar al método _shuffle_.

# In[4]:


puzzle.shuffle(10)
print("Rompecabezas revuelto:\n",puzzle)


# ## Estructuras de datos

# A continuación te presentamos algunas estructuras de datos que servirán para la implementación de los algoritmos de búsqueda.

# ### Pilas

# Para implementar las pilas se puede utilizar la clase __deque__ definida en el paquete _collections_.

# In[5]:


from collections import deque
# Vamos a usar deque como una pila
pila = deque()
# Agreguemos tres estados
pila.append(1)
pila.append(2)
pila.append(3)
# imprimimos el estado de la pila
print(pila)


# Ahora vamos a sacar el elemento en el tope.

# In[6]:


# sacamos el tope
tope = pila.pop()
print(tope)


# In[7]:


# volvemos a sacar del tope
tope = pila.pop()
print(tope)


# In[8]:


# imprimimos el estado de la pila
print(pila)


# In[9]:


# una operación importante de las pilas es 
# consultar el tope sin sacarlo
# esto es la operacion peek
# para ello solo consultamos el elemento
# con posición -1
tope = pila[-1]
print("el tope:",tope)
print("la pila:",pila)


# ### Colas

# Para implementar la cola usamos __deque__ también.

# In[10]:


# creamos la cola
cola = deque()
cola.append(1)
cola.append(2)
cola.append(3)
print("cola:",cola)


# Para sacar el frente de la cola usamos el método _popleft_

# In[11]:


# sacamos el elemento al frente
frente = cola.popleft()
print("frente:",frente)
print("cola:",cola)


# ### Conjuntos (Hash sets)

# Los conjuntos en python se crean con la clase __set__.

# In[12]:


# Creamos un conjunto vacío
conjunto = set()
# agregamos algunos elementos al conjunto
conjunto.add(1)
conjunto.add(2)
conjunto.add(3)
print("conjunto:",conjunto)


# Para verificar pertenencia usamos la palabra reservada _in_.

# In[13]:


print("¿está 4 en el conjunto?",4 in conjunto)
print("¿está 3 en el conjunto?", 3 in conjunto)


# Para remover un elemento del conjunto usamos la función _remove_.

# In[14]:


# Eliminamos al 2 del conjunto
conjunto.remove(2)
print("conjunto:",conjunto)


# Podemos unir conjuntos con el método _union_ en __set__.

# In[15]:


A = {1,3,5} 
B = {5,8,9}
C = A.union(B)
print("A:",A)
print("B:",B)
print("Union de A y B:",C)


# Podemos intersectar conjuntos con el método _intersection_ de __set__.

# In[16]:


D = A.intersection(B)
print("Intersección de A y B:",D)


# ### Tablas de dispersión o diccionarios

# Los diccionarios de python nos permiten asociar parejas de objetos.
# El primer elemento de una pareja es la llave, el segundo elemento es el valor.

# In[17]:


# Crear un diccionario vacío
diccionario = {}
print("diccionario vacío:",diccionario)


# In[18]:


# vamos a asociar el dígito 1 con su nombre
diccionario[1] = "uno"
print("diccionario:",diccionario)


# In[19]:


# agreguemos algunas otras asociaciones
diccionario[2] = "dos"
diccionario[5] = "cinco"
diccionario[9] = "nueve"
print("diccionario:",diccionario)


# Podemos verificar si una llave esta en el diccionario con la palabra reservada _in_.

# In[20]:


print("¿está el número 2 como llave en el diccionario?",2 in diccionario)
print("¿está el número 7 como llave en el diccionario?",7 in diccionario)


# Para extraer el valor asociado a una llave, usamos los corchetes.

# In[21]:


print("El valor asocidado a la llave 2 es:",diccionario[2])


# ### Colas de prioridad

# Las colas de prioridad son muy eficientes para obtener el elemento de mayor prioridad.
# En python usamos la clase __heapq__.
# 

# In[22]:


from heapq import heappush as push
from heapq import heappop as pop

# creamos la cola vacía
colap = []
# agregamos un elemento indicando la prioridad (primer elemento de la tupla)
push(colap,(3,"hola"))
# agregamos un segundo elemento
push(colap,(5,"mundo"))
# uno más
push(colap,(1,"adios"))
# imprimimos la cola
print("la cola tras las inserciones:",colap)
# extraemos el elemento de mayor prioridad (menor valor)
# en este caso el de prioridad 1
p = pop(colap)
print("Elemento de mayor prioridad:",p)


# In[23]:


# El siguiente elemento:
p = pop(colap)
print("Elemento de mayor prioridad:",p)


# In[24]:


print(colap)


# ## Algoritmo BFS

# Ilustraremos como implementar el algoritmo __BFS__ para resolver el rompecabezas del 15.
# 
# Comenzaremos por definir la función _trajectory_ para recuperar la ruta a partir del nodo meta.

# In[25]:


from collections import deque

# trajectory nos regresará la trayectoria a partir de un estado
def trajectory(end):
    # nos valemos de un deque para almacenar la ruta
    sequence = deque()
    # agregamos el estado final o meta
    sequence.append(end)
    # nos vamos regresando al estado predecesor mientras este exista
    while end.get_parent():
        # nos movemos al predecesor
        end = end.get_parent()
        # lo agregamos a la lista
        sequence.append(end)
    # invertimos el orden de la secuencia
    sequence.reverse()
    # lo regresamos como una lista
    return list(sequence)


# Por ejemplo vamos a crear un nuevo rompecabezas del 15 ordenado.

# In[26]:


ordenado = Puzzle()
print("ordenado:",ordenado)
print("la profundidad del estado ordenado:",ordenado.get_depth())


# Ahora vamos a expandir la configuración y tomamos el primer elemento de la lista de sucesores.

# In[27]:


#Obtenemos los sucesores del estado ordenado
sucesores = ordenado.expand() 
# imprimimos los sucesores
print("sucesores del estado ordenado:",sucesores)


# Podemos ver que hay dos sucesores.
# Tomemos el primero y calculemos sus sucesores.

# In[28]:


# el primer sucesor
primer_sucesor = sucesores[0]
print("el primer sucesor del estado ordenado:",primer_sucesor)
# imprimimos su profundidad
print("su profundidad:",primer_sucesor.get_depth())
# sucesores del primer sucesor del estado ordenado
sucesores_primer_sucesor = primer_sucesor.expand()
# imprimimos los sucesores a profundidad 
print("los sucesores del primer sucesor:",sucesores_primer_sucesor)


# Tomemos el primer sucesor del primer sucesor.

# In[29]:


primer_sucesor_primer_sucesor = sucesores_primer_sucesor[0]
print("primer sucesor del primer sucesor:",primer_sucesor_primer_sucesor)


# Con la función trayectory podemos extraer la ruta desde el estado ordenado.

# In[30]:


ruta = trajectory(primer_sucesor_primer_sucesor)
print("ruta desde el estado ordenado: ",ruta)


# Ahora damos la implementación del algoritmo BFS.

# In[48]:


class DFS:

    @staticmethod    
    def search(start,stop):
        """
        Realiza la búsqueda primero en anchura
        :param start: el estado inicial
        :param stop: una función de paro
        """
        # usamos deque para la agenda que será una pila
        agenda = deque()
        # un conjunto basado en tabla de dispersión para
        # registrar los estados expandidos
        explored = set()
        # verificamos la condición trivial
        if(stop(start)):
            # regresamos la ruta trivial
            return trajectory(start)
        # agregamos el primer estado a la agenda
        agenda.append(start)
        # mientras la agenda tenga elementos
        while(agenda):
            # sacamos el elemento al frente de la cola
            nodo = agenda.pop()
            # lo agregamos a los expandidos
            explored.add(nodo)
            # para cada sucesor del nodo
            for child in nodo.expand():
                # si el sucesor es la meta 
                if stop(child):
                    # recuperamos la ruta y la regresamos
                    return trajectory(child)
                # si el nodo no se ha previamente expandido
                elif not child in explored:
                    # agregamos los sucesores a la agenda
                    agenda.append(child)
        # en caso de que no haya ruta
        # (instrucción redundante)
        return None


class BFS:

    @staticmethod    
    def search(start,stop):
        """
        Realiza la búsqueda primero en anchura
        :param start: el estado inicial
        :param stop: una función de paro
        """
        # usamos deque para la agenda que será una cola
        agenda = deque()
        # un conjunto basado en tabla de dispersión para
        # registrar los estados expandidos
        explored = set()
        # verificamos la condición trivial
        if(stop(start)):
            # regresamos la ruta trivial
            return trajectory(start)
        # agregamos el primer estado a la agenda
        agenda.append(start)
        # mientras la agenda tenga elementos
        while(agenda):
            # sacamos el elemento al frente de la cola
            nodo = agenda.popleft()
            # lo agregamos a los expandidos
            explored.add(nodo)
            # para cada sucesor del nodo
            for child in nodo.expand():
                # si el sucesor es la meta 
                if stop(child):
                    # recuperamos la ruta y la regresamos
                    return trajectory(child)
                # si el nodo no se ha previamente expandido
                elif not child in explored:
                    # agregamos los sucesores a la agenda
                    agenda.append(child)
        # en caso de que no haya ruta
        # (instrucción redundante)
        return None


# Vamos a probar el algoritmo BFS.
# Para ello revolvemos 5 movimientos aleatorios.

# In[49]:


# Un nuevo rompecabezas
puzzle = Puzzle()
# 20 movimeintos aleatorios
puzzle.shuffle(5)
print("rompecabezas desordenado:",puzzle)


# Invocamos al metodo _search_ de nuestra clase BFS.
# La condición de paro es que el rompecabezas esté ordenado.

# In[50]:


# la función de paro evalua a cierto cuando el estado es igual al rompecabezas ordenado
ruta = BFS.search(puzzle,lambda s:s==Puzzle())
# imprimimos la ruta
print(ruta)


# ## Algoritmo DFS

# Para implementar el algoritmo DFS solo habría que cambiar una línea de código.
# En tu tarea de programación tendrás que identificar dicha línea y proponer la nueva.

# ## Algoritmo DLS

# En el algoritmo DLS vamos a acotar la profundidad de los estados visitados. 
# No podremos exandir nodos más allá de la cota establecida.
# A continuación una implementación recursiva del algoritmo.

# In[34]:


class DLS:
    """
    Implementación del algoritmo de profundidad limitada
    """
    @staticmethod
    def search(origen,stop,prof_max):
        """
        Método de búsqueda
        :param origen: el estado inicial
        :param stop: la función de paro
        :param prof_max: la cota de profundidad
        """
        # condición base si el origen es la meta nos detenemos
        # recuperando la ruta
        if(stop(origen)):
            return trajectory(origen)
        # si se alcanzo la profundidad de la cota 
        # podemos concluir que no encontramos la meta
        # (los sucesores superarían la cota)
        if(origen.depth == prof_max):
            # regresamos None
            return None
        # hacemos la expansión
        for hijo in origen.expand():
            # para cada sucesor (hijo)
            # establecemos una nueva búsqueda,
            # donde el sucesor es el nuevo estado inicial
            r = DLS.search(hijo,stop,prof_max)
            # si encontramos una ruta la regresamos
            if r :
                return r


# ¿Dónde están las estructuras de datos?
# La agenda es una pila. Cuando hacemos una invocación recursiva Python genera una pila de invocaciones, por lo que la agenda es implícita.
# En DLS no tenemos conjunto de expandidos.

# Antes de probar el algoritmo.
# Vamos a establecer la semilla del generador de números aleatorios con un valor determinado.
# De esta manera podremos hacer repetible los experimentos.

# In[35]:


from random import seed
import random
# Inicializamos el generador
seed(1)

# Creamos un rompecabezas ordenado
puzzle  = Puzzle()

# Desordenamos 5 movimientos aleatorios
puzzle.shuffle(5)
print("rompecabezas revuelto:",puzzle)


# Procedemos a ordenarlo usando el algoritmo BFS, de esa manera sabemos la profundidad de la solución.

# In[36]:


# Encontramos la profundidad de la solución usando BFS
# restamos 1 por que la profundidad es el número de acciones
prof = len(BFS.search(puzzle,lambda s:s==Puzzle())) - 1
print("profundidad de la solución: ",prof)


# Observamos que la profundidad de la solución es 5.
# Si resolvemos con DLS indicando una profundidad de 5 deberíamos encontrar la solución.

# In[37]:


# la cota de DLS se establece a 5 y se invoca
ruta = DLS.search(puzzle,lambda s:s==Puzzle(),prof_max=5)
print(ruta)
print("profundidad de la solución: ",len(ruta)-1)


# Observamos que DLS encuentra la ruta.
# Si la cota es menor que la profundidad de la ruta DLS no la podrá encontrar.

# In[38]:


# la cota de DLS se establece a 4 y se invoca
ruta = DLS.search(puzzle,lambda s:s==Puzzle(),prof_max=4)
print(ruta)


# Nos damos cuenta que DLS es un algoritmo incompleto.
# ¿Ahora que pasa si establecemos una cota superior a la profundidad de la solución?

# In[39]:


# la cota de DLS se establece a 15 y se invoca
ruta = DLS.search(puzzle,lambda s:s==Puzzle(),prof_max=15)
print(ruta)
print("profundidad de la solución: ",len(ruta)-1)


# Concluimos que DLS __NO GARANTIZA__ la solución óptima en número de pasos si la cota es superior a la profundidad de la solución.

# ## Algoritmo ID

# En el algoritmo de profundidad iterada hacemos invocaciones a DLS incrementando la cota de uno en uno hasta encontrar la meta.

# In[40]:


class ID:
    """
    Implementación del algoritmo de profundidad limitada
    """
    @staticmethod
    def search(origen,stop):
        """
        Método de búsqueda
        :param origen: el estado inicial
        :param stop: la función de paro
        :param prof_max: la cota de profundidad
        """
        # condición base si el origen es la meta nos detenemos
        # recuperando la ruta
        if(stop(origen)):
            return trajectory(origen)
        # establecemos la cota de profundidad
        cota = 1
        # no tenemos el resultado
        resultado = None
        while not resultado:
            resultado = DLS.search(origen,stop,cota)
            cota +=1
        return resultado


# In[41]:


#probemos si ID puede encontrar la solución óptima en nuestro ejemplo
ruta = ID.search(puzzle,lambda s:s==Puzzle())
print(ruta)
print("profundidad de la solución:",len(ruta)-1)


# Concluimos que ID puede encontrar la solución óptima.

# ## Algoritmo DFBB

# El algoritmo DFBB se basa en una búsqueda DLS que poda el arbol de búsqueda al encontrar una solución.
# A continuación te presentamos un esqueleto que deberás completar como parte de tu tarea de programación.

# In[59]:


class DFBB:
    
    @staticmethod
    def search(start,stop, prof_max):
        """
        Búsqueda primero en profundidad con arborescencia y poda
        :param start: estado inicial
        :param stop: función de paro evalúa a verdadero en un estado meta
        :param prof_max: la profundidad máxima
        """
        # no tenemos la solución
        solucion = None
        # usamos deque para la agenda que será una pila
        agenda = deque()
        # un conjunto basado en tabla de dispersión para
        # registrar los estados expandidos
        explored = set()
        # verificamos la condición trivial
        if(stop(start)):
            # regresamos la ruta trivial
            return trajectory(start)
        # agregamos el primer estado a la agenda
        agenda.append(start)
        # mientras la agenda tenga elementos
        while(agenda):
            # sacamos el elemento en el tope de la pila
            nodo = agenda.pop()
            # siempre que no se haya alcanzado la cota de profundidad
            if nodo.get_depth()<prof_max:
                # para cada sucesor del nodo
                for child in nodo.expand():
                    # si el sucesor es la meta 
                    if stop(child):
                        # ESCRIBE AQUÍ TU CÓDIGO
                        # EMPIEZA TU CÓDIGO
                        solucion = DLS.search(child,stop,prof_max)
                        prof_max = len(solucion)-2
                        # TERMINA TU CÓDIGO
                        # conserva esta línea es parte de la salida
                        # para la evaluación automática
                        print("ruta de %d pasos"%(len(solucion)-1))
                    else:
                        # agregamos los sucesores a la agenda
                        agenda.append(child)
        # regresamos la solución
        return solucion        


# In[60]:


# Descomenta las líneas siguientes ejecuta la celda y pega la salida en un
# archivo que someterás como tarea de programación
seed(2019)
puzzle  = Puzzle()
puzzle.shuffle(5)
ruta = DFBB.search(puzzle,lambda s:s==Puzzle(),15)
print("".join(str(i.configuration) for i in ruta))
print("longitud de ruta:",len(ruta)-1)


# ## Algoritmo Bidireccional

# Ahora trataremos el algoritmo Bidireccional.
# Tendrás que completar el código para resolver tu tarea.

# In[77]:


class Bidireccional:
    #Método de búsqueda bidireccional
    @staticmethod
    def search(start,end):
        """
        Búsqueda bidireccional
        :param start: estado inicial
        :param end: estado meta
        """
        # condición trivial
        if start == end:
            return trajectory(start)
        # frontera hacia adelante
        # colocamos el estado inicial
        Df = {start:start}
        # frontera hacia atrás
        # colocamos el estado meta
        Db = {end:end}
        # nuestro conjunto de expandidos
        E = {}
        # la cadena siguiente sólo sirve para calificar tu tarea
        # no es parte del algoritmo
        s = ''
        while Df and Db:
            # fronteras temporales vacías
            Dfp = {}
            Dbp = {}
            # expandir frontera hacia adelante
            for n in Df:
                E[n]=n
                for h in n.expand():
                    if h in Db:
                        print(s)
                        r = trajectory(Db[h])
                        r.reverse()
                        return trajectory(h)+r[1:]
                    if h not in E:
                        # conserva esta línea es parte de la salida
                        # para la evaluación automática
                        s+='.'
                        Dfp[h]=h
            Df = Dfp
            #expandir frontera hacia atrás
            for n in Db:
                E[n] = n
                for h in n.expand():
                    if h in Df:
                        print(s)
                        # ESCRIBE AQUÍ TU CÓDIGO
                        # EMPIEZA TU CÓDIGO
                        r = trajectory(Df[h])
                        r.reverse()
                        return trajectory(h)+r[1:]
                        # TERMINA TU CÓDIGO
                    if h not in E:
                        # conserva esta línea es parte de la salida
                        # para la evaluación automática
                        s+='-'
                        Dbp[h]=h
            Db = Dbp


# In[80]:


# Descomenta las líneas siguientes ejecuta la celda y pega la salida en un
# archivo que someterás como tarea de programación
seed(20190131125)
puzzle  = Puzzle()
puzzle.shuffle(5)
ruta = Bidireccional.search(puzzle,Puzzle())
print("".join(str(i.configuration) for i in ruta))
print("longitud de ruta:",len(ruta)-1)

