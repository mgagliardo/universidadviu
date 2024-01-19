package main.com.universidadviu;

import java.util.Random;
import java.util.Scanner;

public class GenerarNumerosAleatorios {

    public static void main(String[] args) {

        int cantNumeros = leerNumeroPorTeclado("Introduca la cantidad de numeros aleatorios a generar: ");
        int desde = leerNumeroPorTeclado("Introduca el rango desde: ");
        int hasta = leerNumeroPorTeclado("Introduca el rango hasta: ");

        if (desde > hasta) {
            System.out.println("El rango `desde`: " + desde + ", no puede ser mayor al rango `hasta`: " + hasta + ". Por favor introduzca el rango `desde` nuevamente.");
            desde = leerNumeroPorTeclado("Introduca el rango desde: ");
        }

        final int rango = hasta - desde + 1;
        final int[] numeros = new int[rango];

        Random rand = new Random();
        int numAleatorio;
        for (int i = 0; i < cantNumeros; i++) {
            numAleatorio = rand.nextInt(rango) + desde;
            numeros[numAleatorio - desde]++;
        }

        System.out.println("Imprimiendo la cantidad de ocurrencias..");
        for (int i = desde; i <= hasta; i++) {
            System.out.println(i + ": " + numeros[i - desde]);
        }
    }

    private static int leerNumeroPorTeclado(String mensaje) {
        Scanner reader = new Scanner(System.in);
        System.out.print(mensaje);
        int num = reader.nextInt();
        while (!chequearSiEsNatural(num)) {
            System.out.print("El numero ingresado debe ser natural (mayor a 0), ingreselo nuevamente: ");
            num = reader.nextInt();
        }
        return num;
    }

    private static boolean chequearSiEsNatural(int num) {
        return num > 0;
    }

}
