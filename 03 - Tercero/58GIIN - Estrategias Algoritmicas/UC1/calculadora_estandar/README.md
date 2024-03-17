# Calculadora con numeros flotantes

Calculadora estandar pero que acepta numeros enteros positivos, operadores aritméticos (+, -, *, /) y paréntesis e imprime el valor calculado

## Para compilar

```shell
flex calc.l && bison -d calc.y && gcc lex.yy.c calc.tab.c -lfl -o programa
```

## Para ejecutar

```shell
./programa 
Ingrese el cálculo (ESC para salir): 2 + 3
Resultado: 5.00
```
