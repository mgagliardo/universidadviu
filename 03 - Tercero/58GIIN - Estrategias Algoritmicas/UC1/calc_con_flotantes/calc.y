%{
#include <stdio.h>
#include <stdbool.h>
int yylex();
void yyerror(const char *s);
%}

%union{
    float dval;
}

%token <dval> NUMERO
%token SUMA RESTA MULT DIV L_PAREN R_PAREN FIN_LINEA
%token ESCAPE

%type <dval> expresion factor termino

%%

linea: expresion FIN_LINEA { printf("Resultado: %.2f\n", $1); }
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

termino: NUMERO   { $$ = $1; }
       | L_PAREN expresion R_PAREN { $$ = $2; }
       ;

%%

// Funciones

// Manejo de errores
void yyerror(const char *s) {
    fprintf(stderr, "Error sintáctico: %s\n", s);
}


// Main
int main() {
    printf("Ingrese el cálculo (ESC para salir): ");
    
    // Ejecutar el cálculo inicial
    while (yyparse()) {
        if (yychar == ESCAPE) {
            printf("Saliendo del programa...\n");
            return 0; // Salir del programa si se presiona la tecla Escape
        }
        printf("Ingrese otro cálculo (ESC para salir): ");
    }
    
    return 0;
}
