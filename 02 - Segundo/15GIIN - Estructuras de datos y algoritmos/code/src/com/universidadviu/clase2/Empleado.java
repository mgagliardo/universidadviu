package com.universidadviu.clase2;

public class Empleado extends Persona {
    private double salary;

    public Empleado(int ed, double sal) {
        super(ed);
        salary = sal;
    }

    public double getSalary() {
        return salary;
    }
}
