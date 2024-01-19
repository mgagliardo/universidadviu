package com.universidadviu.clase4;

public class BusquedaBinaria {

    // Calculo del n-esimo numero de Fibonacci
    public static boolean busquedaBinaria(int[] a, int x) {
        // Metodo `driver` que prepara la llamada recursiva
        return busquedaBinariaRecursiva(a, 0, a.length - 1, x);
    }

    public static boolean busquedaBinariaRecursiva(int[] a, int ini, int fin, int x) {
        //Caso base
        if(ini > fin) {
            return false;
        }

        if(ini == fin) {
            return x == a[ini];
        }

        int medio = (ini+fin)/2, y = a[medio];
        if (y == x) {
            return true;
        }

        if (x > y) {
            return busquedaBinariaRecursiva(a, 1 + medio, fin, x);
        } else {
            return busquedaBinariaRecursiva(a, ini, medio - 1, x);
        }
    }

    public final static void main(String[] args) {
        int[] coleccion = {-3, 0, 5, 12, 20, 22, 23, 30};
        if (busquedaBinaria(coleccion, 20)) {
            System.out.println("Se encontro el numero en la coleccion");
        }
    }
}
