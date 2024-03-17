%{
#include <stdio.h>
#include <stdlib.h>
#include "calc.tab.h"
%}

%option noyywrap

DIGIT       [0-9]

%% 

{DIGIT}+    { yylval = atoi(yytext); return NUMERO; }
"+"         { return SUMA; }
"-"         { return RESTA; }
"*"         { return MULT; }
"/"         { return DIV; }
"("         { return L_PAREN; }
")"         { return R_PAREN; }
\n          { return FIN_LINEA; }
[\t ]+      ; // Ignora espacios en blanco y tabulaciones
"$"         {printf("Saliendo del programa..\n"); exit(1);}


%%
