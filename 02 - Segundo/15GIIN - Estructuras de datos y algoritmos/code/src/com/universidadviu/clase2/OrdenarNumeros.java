package main.com.universidadviu;

import java.util.Arrays;
import java.util.Scanner;

public class OrdenarNumeros {
    public static void main(String args[]) {
        int i;
        int cantNum = 3;
        int[] arrayNum = new int[cantNum];
        Scanner leerNum = new Scanner(System.in);
        // Lee los 3 numeros y los inserta en el `arrayNum`
        for (i = 0; i < cantNum; i++) {
            System.out.println("Inserte el numero " + (i+1) + ": ");
            arrayNum[i] = leerNum.nextInt();
        }
        leerNum.close(); // Cierro el scanner
        Arrays.sort(arrayNum); // Arrays Sort
        System.out.println(Arrays.toString(arrayNum));
    }

}
