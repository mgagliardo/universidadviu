%{
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
int yylex();
void yyerror(const char *s);
bool imprimirResultado = true; // Variable global para controlar la impresión del resultado
%}

%token NUMERO
%token SUMA RESTA MULT DIV L_PAREN R_PAREN FIN_LINEA

%%

linea: expresion FIN_LINEA {
        if (imprimirResultado) {
            printf("Resultado: %d\n", $1);
        }
}
     | FIN_LINEA
     ;

expresion: factor
         | expresion SUMA factor { $$ = $1 + $3; }
         | expresion RESTA factor { $$ = $1 - $3; }
         ;

factor: termino
      | factor MULT termino { $$ = $1 * $3; }
      | factor DIV termino { 
          if ($3 == 0) {
              yyerror("No se puede realizar una división por cero");
          } else {
              $$ = $1 / $3;
          }
      }
      ;

termino: NUMERO
       | L_PAREN expresion R_PAREN { $$ = $2; }
       ;

%%

// Funciones

// Manejo de errores
void yyerror(const char *s) {
    fprintf(stderr, "Error sintáctico: %s\n", s);
    imprimirResultado = false; // No imprimir resultado si hay error
}

// Main
int main() {
    printf("Ingrese el cálculo ($ para salir): ");
    yyparse();
    return 0;
}
