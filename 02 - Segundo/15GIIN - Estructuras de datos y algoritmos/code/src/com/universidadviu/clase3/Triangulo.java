package com.universidadviu.clase3;

public class Triangulo implements Shape {
    Double lado1;
    Double lado2;

    Triangulo(Double l1, Double l2) {
        lado1 = l1;
        lado2 = l2;
    }

    public Double area() {
        return (lado1 * lado2) / 2;
    }
}
