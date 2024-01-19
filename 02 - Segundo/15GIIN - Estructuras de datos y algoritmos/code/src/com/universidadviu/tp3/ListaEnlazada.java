package com.universidadviu.tp3;


// Implementacion de una Lista enlazada simple
public class ListaEnlazada<AnyType> {
    // Una referencia a un Nodo es suficiente para definir la lista

    private Nodo<AnyType> lista = null;
    private int tam = 0;

    private class Nodo<T> { // clase interna para 
    	                               // definir nodos
    	                               // de tipo generico
        private T elemento;
        private Nodo<T> siguiente;
        
        public Nodo(T o, Nodo<T> n) {
            establecerElemento(o);
            establecerSiguiente(n);
        }

        public void establecerElemento(T elemento) {
            this.elemento = elemento;
 //         System.out.println(tam); //tenemos acceso a los campos 
                                   // privados de ListaEnlazada          
        }

        public void establecerSiguiente(Nodo<T> sig) {
            siguiente = sig;
        }

        public T obtenerElemento() {
            return elemento;
        }

        public Nodo<T> obtenerSiguiente() {
            return siguiente;
        }
    }
   
    public ListaEnlazada() {
        // Constructor que crea una lista vacía
        lista = null; tam = 0;
    }

    public int tam() {
        // devolver el numero de elementos de la lista
        return tam;
    }

    public void agregar(AnyType o) {
        // el nuevo elemento o es colocado de primero en la lista
        Nodo<AnyType> tmp = new Nodo<>(o, null);

        // la instruccion siguiente no tiene problemas 
        // si lista es null (vacia)
        tmp.establecerSiguiente(lista);
        // actualizamos donde apunta la lista
        lista = tmp;
        tam++;
    }

    public boolean estaVacia() {
        return lista == null;
    }
    
    public boolean eliminar(int  i) {
        // Elimina el elemento en la posicion i de la lista
        // Devuelve falso si y solo si  i no est� entre 0 y tam-1, 

        if ((i<0) || (i>=tam)) return false;
        else 
            if (i==0) {
        	    lista = lista.obtenerSiguiente();     
                return true;
            }
            else  {
                Nodo<AnyType> anterior, actual;
                anterior = lista;
                actual = lista.obtenerSiguiente();
                int j = 1;
                while (j < i) {
                    anterior = actual;
                    actual = actual.obtenerSiguiente();
                    j ++ ;
                }
                anterior.establecerSiguiente(actual.obtenerSiguiente());
                // podria haber colocado "actual.siguiente"
                // el objeto lista tiene acceso a los campos
                // privados de Nodo, aunque no deberiamos
                return true;
           }
    }
    
    public AnyType obtener(int i) {
        // obtener el elemento en la posicion i
        if ((i>=0)&&(i<tam)) {
            Nodo<AnyType> tmp ;

            int j = 0; tmp = lista;
            while (j < i) {
                j ++ ;
                tmp = tmp.obtenerSiguiente(); //tmp será el elemento 
                                              //en la posición j
                          // el objeto lista tiene acceso a los datos 
                          // privados de la clase Nodo
                          // podria haber colocado: tmp.siguiente
                          // aunque no se deberia
                          //
            }
            return tmp.obtenerElemento();
        }
        return null;
    }
    
    public void print() {
        // Imprime la lista. Si la lista esta vacia no haga nada
        System.out.println(lista);
        if (!estaVacia()) {
            Nodo<AnyType> tmp = lista;
            // iterar sobre los nodos hasta llegar al final de la lista
            while (tmp != null) {
                System.out.print(tmp.obtenerElemento() + " ");
                tmp = tmp.obtenerSiguiente();
            }
            System.out.println();
        }
    }

    // Metodo para invertir la lista
    public void invertir() {
        // Inicializo los elementos necesarios SIN crear nuevos nodos
        Nodo<AnyType> anterior = null;
        Nodo<AnyType> siguiente = null;
        Nodo<AnyType> actual = lista;

        // Mientras que el nodo actual no sea null, recorro el loop
        // No hay necesidad de validar si esta vacia o no
        // dado que si actual == null nunca entraremos en el loop
        while (actual != null) {
            siguiente = actual.obtenerSiguiente();
            actual.establecerSiguiente(anterior);
            anterior = actual;
            actual = siguiente;
        }
        lista = anterior;
    }

    public static void main(String args[]) {
	    // Creamos (instanciamos) un objeto tipo ListaEnlazada<String>, es decir, una lista
	    // cuyos elementos son Strings (cadena de caracteres) y luego
	    // la imprimimos mediante el metodo print() de la clase ListaEnlazada
	    // y luego la imprimimos en el programa principal (main) utilizando los m�todos de la lista
	    // (recuerde que los nodos estan escondidos, la clase interna Nodo es  private)

        ListaEnlazada<String> listaEnlazada1 = new ListaEnlazada<>();   
        listaEnlazada1.agregar("hola3");
        listaEnlazada1.agregar("hola2");
        listaEnlazada1.agregar("hola4");
        listaEnlazada1.agregar("hola5");
        listaEnlazada1.agregar("hola4");
        listaEnlazada1.print();

        // Aplicar el metodo invertir y luego imprimir la lista invertida
        System.out.println("Invirtiendo lista..");        
        listaEnlazada1.invertir();
        listaEnlazada1.print();
  }
}
