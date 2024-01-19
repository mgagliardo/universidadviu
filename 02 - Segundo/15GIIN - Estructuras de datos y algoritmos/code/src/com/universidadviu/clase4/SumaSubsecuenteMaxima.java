package com.universidadviu.clase4;

public class SumaSubsecuenteMaxima {

    public static int sumaMaximaCruzada(int[] nums, int len, int mitad, int h) {
		int sum = 0;

        // Suma del lado izquierdo de la mitad
		int sumaIzquierda = Integer.MIN_VALUE;
		for (int i = mitad; i >= len; i--) {
			sum = sum + nums[i];
			if (sum > sumaIzquierda) {
                sumaIzquierda = sum;
            }
		}

		sum = 0;
        // Suma del lado derecho de la mitad
		int sumaDerecha = Integer.MIN_VALUE;
		for (int i = mitad; i <= h; i++) {
			sum = sum + nums[i];
			if (sum > sumaDerecha)
				sumaDerecha = sum;
		}

		// Devuelve la suma de los elementos a la izquierda y derecha de la mitad
		// devuelve solamente sumaIzquierda + sumaDerecha 
		return Math.max(sumaIzquierda + sumaDerecha - nums[mitad],
						Math.max(sumaIzquierda, sumaDerecha));
    }

    public static int sumaMaximaSubArray(int[] nums, int l, int len) {
        int mitadArray = (l + len) / 2;

        // Devuelve el maximo de los 3
        return Math.max(
            Math.max(sumaMaximaSubArray(nums, l, mitadArray-1),
                    sumaMaximaSubArray(nums, mitadArray + 1, len)),
                    sumaMaximaCruzada(nums, l, mitadArray, len));
    }

    public static void main(String[] args) {
        int[] nums = { 4, -3, 5, -2 };
        int n = nums.length - 1;
        int posInicial = 0;
        int maxSum = sumaMaximaSubArray(nums, posInicial, n);
        System.out.println("Maximum contiguous sum is " + maxSum);
    }
}
