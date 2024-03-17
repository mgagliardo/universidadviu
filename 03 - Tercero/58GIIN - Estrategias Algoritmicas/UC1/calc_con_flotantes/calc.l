%{
#include <stdio.h>
#include <stdlib.h>
#include "calc.tab.h"
%}

DIGIT               [0-9]

%% 

(\.{DIGIT}+)|({DIGIT}+(\.{DIGIT}*)?([eE][+-]?[0-9]+)?)   {yylval.dval = atof(yytext); return NUMERO;}   // Regex para numeros flotantes

"+" {return SUMA;}
"-" {return RESTA;}
"*" {return MULT;}
"/" {return DIV;}
"(" {return L_PAREN;}
")" {return R_PAREN;}


"\n"                { return FIN_LINEA; }
[\t ]+              ;                                                                                   // Ignora espacios en blanco y tabulaciones
[^\n\t ]            { return yytext[0]; } // Caracteres no reconocidos



"^[^[:alnum:]]$"    { return ESCAPE; } // Reconoce la tecla "ESC"

%%
