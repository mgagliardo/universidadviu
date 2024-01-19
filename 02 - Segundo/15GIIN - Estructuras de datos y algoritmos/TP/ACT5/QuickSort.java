package com.universidadviu.tp5;

import java.util.Arrays;

/**
 * Clase QuickSort que contiene el metodo estatico QuickSort
 */
public final class QuickSort {

    private static final int CUTOFF = 2;

    /**
     * Metodo interno para ordenar subarrays usan el metodo de ordenamiento InsertionSort
     * Usada internamente por QuickSort
     * @param array un array de items comparables
     * @param izq el indice del lado izquierdo del subarray
     * @param cant la cantidad de items a ordenar
     */
    private static void insertionSort(Comparable[] array, int izq, int cant) {
        for(int p = izq + 1; p <= cant; p++) {
            Comparable tmp = array[p];
            int j;
            for(j = p; j > izq && tmp.compareTo(array[j - 1]) < 0; j--) {
                array[j] = array[j - 1];
            }
            array[j] = tmp;
        }
    }

    /**
     * Ordenamiento por Quicksort
     * @param array un array de items Comparables
     */
    public static void quicksort(Comparable[] array) {
        quicksort(array, 0, array.length - 1);
    }

    /**
     * Metodo de ayuda para intercambiar los elementos de un array
     * @param array un array de objetos
     * @param index1 el indice del primer objeto
     * @param index2 el indice del segundo objeto
     */
    private static final void swapReferences(Object[] array, int index1, int index2) {
        Object tmp = array[index1];
        array[index1] = array[index2];
        array[index2] = tmp;
    }

    /**
     * Metodo QuickSort interno que usa llamadas recursivas
     * Usa la mediona tal como se pide en el ejercicio
     * CUTOFF = 2 definido como variable de clase
     * @param array un array de items comparables
     * @param bajo el indice del lado izquierdo del subarray
     * @param alto el indice del lado derecho del subarray
     */
    private static void quicksort(Comparable[] array, int bajo, int alto) {
        if(bajo + CUTOFF > alto) {
            insertionSort(array, bajo, alto);
        } else {

            // Ordeno bajo, medio, alto
            int medio = (bajo + alto) / 2;

            // Si el elemento del medio del array es mas bajo que el indice del lado izquierdo, los intercambio
            if(array[medio].compareTo(array[bajo]) < 0) {
                swapReferences(array, bajo, medio);
            }

            // Si el ultimo elemento del array es mas bajo que el indice del lado izquierdo, los intercambio
            if(array[alto].compareTo(array[bajo]) < 0) {
                swapReferences(array, bajo, alto);
            }

            // Si el ultimo elemento del array es mas bajo que el indice del medio, los intercambio
            if(array[alto].compareTo(array[medio]) < 0) {
                swapReferences(array, medio, alto);
            }

            // Pongo el pivot en la posicion [alto - 1]
            swapReferences(array, medio, alto - 1);
            Comparable pivot = array[alto - 1];

            // Particiono
            int i, j;
            for(i = bajo, j = alto - 1; ;) {
                while(array[++i].compareTo(pivot) < 0)
                    ;
                while(pivot.compareTo(array[--j]) < 0)
                    ;
                if(i >= j) {
                    break;
                }
                swapReferences(array, i, j);
            }

            // Restauro el pivot
            swapReferences(array, i, alto - 1);

            // Ordeno los los elementos mas chicos
            quicksort(array, bajo, i - 1);

            // Ordeno los los elementos mas grandes
            quicksort(array, i + 1, alto);
        }
    }

    public static void main(String [] args) {
        // Creo el array de numeros solicitados en el ejercicio
        Integer[] numeros = {10, 9, 8, 6, 5, 9 ,8 ,6 ,5, 1};

        // // Descomentar para leer los numeros por linea de comandos, separados por un espacio
        // Scanner scanner = new Scanner(System.in);
        // System.out.print("\nInserte numeros en una misma linea, separados por un espacio: ");
        // String n = scanner.nextLine();
        // String[] no = n.split(" ");
        // Integer[] numeros = new Integer[10];;
   
        // // Agregamos todos los elementos al Arbol
        // for (int i = 0; i < no.length; i++) {
        //     numeros[i] = Integer.parseInt(no[i]);
        // }

        // Imprimo los numeros antes de aplicarles QuickSort  version de Weiss
        System.out.println("Numeros sin ordenar: " + Arrays.toString(numeros));

        // Aplico QuickSort al array de numeros
        quicksort(numeros);

        // Imprimo los numeros luego de aplicarles QuickSort (version de Weiss)
        System.out.println("Numeros ordenados: " + Arrays.toString(numeros));
    }
}
