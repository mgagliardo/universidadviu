package com.universidadviu.tp1;

import java.util.ArrayList;

/*
 * La clase Conjunto<E> generica implementa
 * Herencia de ArrayList<E>
 */
class Conjunto<E> extends ArrayList<E> {

    /*
     * En vez de crer un nuevo array
     * Llamando a new ArrayList<>()
     * Directamente llamo al constructor de la clase padre
     * Usando super()
     */
    Conjunto() {
        super();
    }

    // Funcion que devuelve True si el objeto se encuentra en el conjunto
    boolean pertenece(E o) {
        return this.contains(o);
    }

    // Agrega un elemento al conjunto
    void agregar(E o)  {
        // Chequea primero si el elemento NO pertenece al Conjunto
        if (!this.pertenece(o)) {
            // Si no pertenece, lo agrega
            this.add(o);
        }
    }

    // Elimina el elemento o del conjunto
    void eliminar(E o) {
        this.remove(o);
    }

    /*
     * Funcion de tipo estatica que devuelve la union de 2 conjuntos
     * Aunque no se especifica en el enunciado, se decidio que NO se creen elementos duplicados
     * Por ejemplo con dos conjuntos:
     * a = {1,2,3}
     * b = {1,2,3}
     * El conjunto a devolver sera {1,2,3}
     */
    static <T> Conjunto<T> union(Conjunto conjunto1, Conjunto conjunto2) {
        // Se crea un nuevo conjunto a devolver
        Conjunto conjuntoCopia = new Conjunto<>();
        // Se agregan todos los elementos del conjunto1 utilizando la funcion nativa `agregar`
        for (Object o : conjunto1) {
            conjuntoCopia.agregar(o);
        }
        // Se agregan todos los elementos del conjunto2 utilizando la funcion nativa `agregar`
        for (Object o : conjunto2) {
            conjuntoCopia.agregar(o);
        }
        return conjuntoCopia;
    }

    /*
     * Funcion de tipo estatica que devuelve la diferencia del conjunto1 - conjunto2
     * Por ejemplo con dos conjuntos:
     * a = {1,2,3}
     * b = {3,4,5}
     * Se devuelve un nuevo conjunto {1,2,4,5}
     */
    static <T> Conjunto<T> diferencia(Conjunto conjunto1, Conjunto conjunto2) {
        Conjunto conjuntoCopia = new Conjunto();
        // Verifica primero que ambos conjuntos no esten vacios para no entrar innecesariamente en el loop
        if (!(conjunto1.isEmpty() && conjunto2.isEmpty())) {
            int i;
            /*
             * Recorre todos los elementos y del conjunto1 y
             * Verifica si el elemento i del conjunto1 NO se encuentra en conjunto2
             * Si es asi, lo agrega, dado que solo queremos
             * Los elementos del conjunto1 que NO estan en conjunto2
             */
            for (i = 0; i < conjunto1.size(); i++) {
                if (!conjunto2.pertenece(conjunto1.get(i))) {
                    conjuntoCopia.agregar(conjunto1.get(i));
                }
            }
        }
        // Devuelve conjuntoCopia
        // Si ambas conjunto1 y conjunto2 son iguales, se devolvera un conjunto vacio
        return conjuntoCopia;
    }

    // Por ultimo, el metodo toArray() no esta implementado dado que se hereda de ArrayList
    // Si se quiere imprimir por pantalla (ver lineas 120, 124 y 127) simplemente se usa el metodo de clase toString
}

// Clase de prueba del conjunto
class PruebaConjunto {
    public static void main(String[] args) {
        // Creo 2 conjuntos de tipo integer
        Conjunto<Integer> conjunto1 = new Conjunto();
        Conjunto<Integer> conjunto2 = new Conjunto();

        // Agrego un elemento al conjunto1
        conjunto1.agregar(1);
        System.out.println("Elementos del Conjunto 1 luego de agregar 1: " + conjunto1.toString());

        // Verifico que remover funciona
        conjunto1.eliminar(1);
        System.out.println("Elementos del Conjunto 1 luego de remover 1: " + conjunto1.toString());

        // Agrego objetos al conjunto1
        conjunto1.agregar(1);
        conjunto1.agregar(2);
        conjunto1.agregar(3);
        conjunto1.agregar(4);

        // Agrego objetos al conjunto2
        conjunto2.agregar(4);
        conjunto2.agregar(5);
        conjunto2.agregar(6);
        conjunto2.agregar(7);

        // Imprimo los elementos de los conjuntos
        System.out.println("Elementos del Conjunto 1: " + conjunto1.toString());
        System.out.println("Elementos del Conjunto 2: " + conjunto2.toString());

        // Chequeo que devuelva la diferencia de conjunto1 - conjunto2
        Conjunto diferenciaConjunto1Conjunto2 = Conjunto.diferencia(conjunto1, conjunto2);
        System.out.println("Diferencia de conjunto1 - conjunto2: " + diferenciaConjunto1Conjunto2.toString());

        // Chequeo que devuelva la diferencia de conjunto2 - conjunto1
        Conjunto diferenciaConjunto2Conjunto1 = Conjunto.diferencia(conjunto2, conjunto1);
        System.out.println("Diferencia de conjunto2 - conjunto1: " + diferenciaConjunto2Conjunto1.toString());

        Conjunto union = Conjunto.union(conjunto1, conjunto2);
        System.out.println("Union de conjuntos: " + union.toString());
    }
}
