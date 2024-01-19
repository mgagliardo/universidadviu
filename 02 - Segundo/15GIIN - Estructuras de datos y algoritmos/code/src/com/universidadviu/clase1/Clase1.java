package main.com.universidadviu;

import java.util.Scanner;

public class Clase1 {
    public static void main(String args[]) {
        Scanner lector = new Scanner(System.in);
        int N = 0, i = 0, suma = 0;
        System.out.println("Coloque el numero N: ");
        N = lector.nextInt();
        do {
            suma = suma + i;
            i++;
        } while (i < N);
        System.out.println("La suma de los primeros " + N + " numeros naturales es igual a: " + suma);
    }

    public static void imprimirTexto(String text) {
        System.out.println("This is a static public class method: " + text);
    }
}
