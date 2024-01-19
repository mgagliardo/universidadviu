package com.universidadviu.tp2;

import java.util.Scanner;

/* Evaluacion de un polinomio por tres programas distintos */

public class EvaluarPolinomio {

    private static double calculaHorner(double [ ] pol, double x, int n) {
    	/* Aplica la formula de Horner para evaluar el polinomio
    	pol de grado n en el valor de la variable x */
    	
        double resultado = 0;
        for (int i = 0; i <= n; i++) {
            resultado = (resultado * x) + pol[i]; 
        }
        return resultado;
    }
    
    private static double calculaConPotencia(double [ ] pol, 
    		double x, int n) { 
    	/* Aplica la formula del polinomio para evaluar
    	pol de grado n en el valor de la variable x 
    	las potencias i-esimas de x se calculan multiplicando
    	 i veces x */
        
    	int i=n; double suma=0.0;
        while (i >= 0) {
            suma += pol[n-i]*potencia(x, i);
            i--;
         }
        return suma;
    }
    private static double calculaConPotencia1(double [ ] pol, 
    		double x, int n) {
    	/*   Aplica la formula del polinomio para evaluar 
    	   * el polinomio pol de grado n en el valor de 
    	   * la variable x 
    	   * las potencias i-esimas de x se calculan con 
    	   * un algoritmo mejorado potencia1*/
 
        int i=n; double suma=0.0;
        while (i >= 0) {
            suma += pol[n-i]*potencia1(x, i);
            System.out.println("suma: " + suma);
            i--;
         }
        return suma;
    }

    public static double potencia(double x,int i) {
    	double resultado = 1.0;
    	for(int j=0;j<i;j++)
    		resultado*=x;
    	return resultado;
    }
    
    public static double potencia1(double x,int i) {
    	System.out.println("i: " + i);
    	if( i == 0 ) return 1;
    	if( i == 1 ) return x;
    	if( i%2 == 0 )return potencia1( x * x, i / 2 );
    	else return potencia1( x * x, i / 2 ) * x;
    }
    
    public final static void main(String[] args) {
        try (Scanner reader = new Scanner(System.in)) {
            System.out.println("Introduce el grado del polinomio: ");
            int n = reader.nextInt();
            double[ ] pol= new double[n+1];
            System.out.println("Introduce coeficientes "
            		+ "polinomio de mayor a menor grado: ");  
            for(int i=0;i<=n;i++) {
            	pol[i] = reader.nextDouble();            
            }
            System.out.println("Introduce el valor x: ");            
            double x = reader.nextDouble();
            
            double resultado;
            // resultado = calculaHorner(pol, x, n);
            // System.out.println(String.format("Resultado: %10.2f", 
            // 		resultado));
            // resultado = calculaConPotencia(pol, x, n);
            // System.out.println(String.format("Resultado: %10.2f", 
            // 		resultado));
            resultado = calculaConPotencia1(pol, x, n);
            System.out.println(String.format("Resultado: %10.2f", 
            		resultado));
        } catch(Exception e) {
            System.out.println("ERROR: " + e.getMessage());
        }
    }    
}
