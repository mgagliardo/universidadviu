# UC1

Ejercicio de construcción de una calculadora simple utilizando Flex (para el análisis léxico, archivo .l) y Bison (para el análisis sintáctico, archivo .y). Este tipo de calculadora soportará operaciones básicas como suma, resta, multiplicación y división.

Desarrollar una calculadora simple que acepte expresiones aritméticas básicas incluyendo suma (+), resta (-), multiplicación (*) y división (/). La calculadora debe soportar el uso de paréntesis para indicar el orden de las operaciones y debe poder manejar números enteros positivos. Además, debe imprimir el resultado de evaluar la expresión ingresada.

**Requisitos**:
- Análisis Léxico (.l): Identificar los tokens para números (enteros positivos), operadores aritméticos (+, -, *, /) y paréntesis.
- Análisis Sintáctico (.y): Definir la gramática para las expresiones aritméticas, asegurando que se respete la precedencia de los operadores y el uso de paréntesis.
- Evaluación: Calcular el resultado de las expresiones aritméticas e imprimir el valor calculado.
