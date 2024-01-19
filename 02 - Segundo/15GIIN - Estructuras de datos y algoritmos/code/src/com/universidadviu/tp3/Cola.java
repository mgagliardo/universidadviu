package com.universidadviu.tp3;

public interface Cola<AnyType> {
    void encolar(AnyType x);

    void desencolar();
    
    // Devolver el frente de la cola
    AnyType frente();

    boolean estaVacia();

    void convertirVacia();

    int numElem( );

    // Para poder imprimir la cola comenzando con el frente de la cola
    String toString(); 
}
