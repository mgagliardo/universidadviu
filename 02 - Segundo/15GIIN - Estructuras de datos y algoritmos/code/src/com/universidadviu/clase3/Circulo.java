package com.universidadviu.clase3;

public class Circulo implements Shape {
    Double radio;

    Circulo(Double r) {
        radio = r;
    }

    public Double area() {
        return Math.pow(Math.PI * radio, 2);
    }
}
