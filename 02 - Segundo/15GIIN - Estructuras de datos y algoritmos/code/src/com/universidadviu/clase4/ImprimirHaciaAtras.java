package com.universidadviu.clase4;

public class ImprimirHaciaAtras {
    public static void imprimirHaciaAtras(int[] nums, int len) {
        if (len == 0) {
            return;
        }

        System.out.println(nums[len-1]);
        imprimirHaciaAtras(nums, len-1);
    }

    public final static void main(String[] args) {
        int[] coleccion = {-3, 0, 5, 12, 20, 22, 23, 30};
        imprimirHaciaAtras(coleccion, coleccion.length);
    }
    
}
