package com.universidadviu.clase2;

public class Estudiante extends Persona {

    private double promedio; // Promedio de calificaciones

    public Estudiante(int ed, double prom) {
        super(ed);
        promedio = prom;
    }

    public double getPromedio() {
        return promedio;
    }
}
