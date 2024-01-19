package com.universidadviu.tp3;

import java.util.ArrayList;
import java.util.Scanner;


public interface Cola<AnyType> {
    void encolar(AnyType x);

    void desencolar();
    
    // Devolver el frente de la cola
    AnyType frente();

    boolean estaVacia();

    void convertirVacia();

    int numElem();

    // Para poder imprimir la cola comenzando con el frente de la cola
    String toString(); 
}



public class ColaConDosPilas<AnyType> implements Cola<AnyType> {

    // La clase contendra 2 colas
    private ArrayList<AnyType> pila1;
    private ArrayList<AnyType> pila2;

    // Builder de la clase, inicializa las colas
    public ColaConDosPilas() {
        pila1 = new ArrayList<>();
        pila2 = new ArrayList<>();
    }

    public void encolar(Object x) {
        // Orden del algoritmo: O(N*2)
        // Con N = Tamaño de la pila1
        // Dado que para el peor caso (donde la pila1 tiene elementos)
        // Primero vaciamos la pila1 (en la pila 2), llamando al metodo desencolar
        // Y luego hacemos la inversa, desencolando pila2 (que a fines practicos tiene size = pila1)
        // Y agregamos cada elemento nuevamente a pila1


        // Si la pila1 esta vacia, directamente agrega el elemento a esta
        if (this.estaVacia()) {
            pila1.add((AnyType) x);
        } else {
            // Si la pila1 no esta vacia, entonces primero la desencola
            this.desencolar();
            
            // Agrega el elemento luego de desencolar
            pila1.add((AnyType) x);

            while (!pila2.isEmpty()) {
                // Luego de agregar el elemento a pila1
                // Desencola pila2
                int ultimo = pila2.size() - 1;
                AnyType topeCola2 = pila2.get(ultimo);
                pila2.remove(ultimo);

                
                // Agrega el ultimo elemento que sacamos de pila2 a pila1
                pila1.add(topeCola2);
                
                // System.out.println("Estado de pila1: " + pila1.toString());
                // System.out.println("Estado de pila2: " + pila2.toString());
            }
        }

        // Descomentar para imprimir lo que en cada caso tendra pila1
        // System.out.println(pila1);
    }

    public void desencolar() {
        // Orden del algoritmo: O(N)
        // Con N = Tamaño de la pila1
        // O bien: pila1.size()

        // Si la pila1 esta vacia, no hay nada que desencolar
        if (this.estaVacia()) {
            return;
        }
        // Mientras pila1 NO este vacia:
        while (!pila1.isEmpty()) {
            // Tomo el ultimo elemento de pila1, lo saco
            int ultimo = pila1.size() - 1;
            AnyType tope = pila1.get(ultimo);
            pila1.remove(ultimo);

            // Agrego el elemento a pila2
            pila2.add(tope);

            // System.out.println("Estado de pila1: " + pila1.toString());
            // System.out.println("Estado de pila2: " + pila2.toString());
        }

        // Descomentar para imprimir lo que en cada caso tendra pila2
        // System.out.println(pila2);
    }

    // El frente es el elemento 0 de la pila1
    public AnyType frente() {
        // Orden del algoritmo: O(1). Sin importar el tamaño de la cola
        return pila1.get(0);
    }

    // Esta vacia quiere decir que no hay elementos en la pila1
    public boolean estaVacia() {
        // Orden del algoritmo: O(1). Sin importar el tamaño de la cola
        return pila1.isEmpty();
    }

    // convertirVacia vacia la Pila, no se utiliza pero se deja implementado
    public void convertirVacia() {
        // Orden del algoritmo: O(2). Sin importar el tamaño de las colas
        pila1 = new ArrayList<>();
        pila2 = new ArrayList<>();
    }

    // numElem devuelva la cantidad de elementos que hay en la pila
    // O sea, la cantidad de elementos en pila1
    public int numElem() {
        // Orden del algoritmo: O(1). Sin importar el tamaño de la cola
        return pila1.size();
    } 

    // Devuelve todos los elementos de la Pila separados cada uno por un espacio
    public String toString() {
        // Orden del algoritmo: O(N)
        // Siendo N el tamaño de la pila - 1
        // O bien: pila1.size() - 1


        String caracteres = "";
        // Si la pila1 esta vacia vamos a devolver un string vacio
        if (!this.estaVacia()) {
            // Recorro la pila1 desde el ultimo (o sea, la cola) al primer indice
            for(int i = pila1.size() - 1; i >= 0; i--) {
                // Encadeno los strings a la cadena y agrego un espacio a cada uno
                caracteres += pila1.get(i).toString() + " ";
            }
        }
        return caracteres;
    }

    public static void main(String args[]) {
        
        // Leer una palabra (con espacios si se desea) por teclado
        System.out.print("Ingrese una palabra: ");
        Scanner input = new Scanner(System.in);
        String palabra = "";
        palabra += input.nextLine();
        input.close();

        // Creacion de la cola
        Cola<String> cola = new ColaConDosPilas<String>();

        // Encolamos el string
        cola.encolar(palabra);

        // Imprimimos el texto
        System.out.println(cola.toString());

        // Vacio la cola y pruebo agregando varios elementos
        cola.convertirVacia();
        cola.encolar("Esto");
        cola.encolar("es");
        cola.encolar("un");
        cola.encolar("texto");
        cola.encolar("de");
        cola.encolar("prueba");
        System.out.println(cola.toString());

        // Prueba con enteros
        Cola<Integer> colaInt = new ColaConDosPilas<Integer>();
        colaInt.encolar(1);
        colaInt.encolar(2);
        colaInt.encolar(3);
        colaInt.encolar(4);
        colaInt.encolar(5);
        colaInt.encolar(6);
        System.out.println(colaInt.toString());
    }
}
