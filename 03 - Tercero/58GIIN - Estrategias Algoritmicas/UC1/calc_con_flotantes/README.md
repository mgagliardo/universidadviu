# Calculadora con numeros flotantes

Calculadora similar a la calculadora estandar pero que acepta numeros con punto flotante (1.2, 3.1, etc.)


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
