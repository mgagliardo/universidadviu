%token <dval> NUMBER
%start input
%left PLUS MINUS
%left MULT DIV

%%
// Expresiones
lines:  exp EQUAL END           {printf("t%f\n", $1);}      // Imprimir los resultados o mostrar
        ;
exp  :  NUMBER                  {$$ = $1;}                  // Para numeros y asignacion de valor
        | exp PLUS exp          {$$ = $1 + $3;}             // Suma
        | exp MINUS exp         {$$ = $1 - $3;}             // Resta
        | exp MULT exp          {$$ = $1 * $3;}             // Multiplicacion
        | exp DIV exp           {$$ = $1 / $3;}             // Division
        ;

sen  :  l_paren input r_paren   {$$ = $2;}                  // Parentesis
sen  :  exp + exp;
%%

// Funciones

void yyerror(const char* msg) {
    printf("%s\n", msg);
};

// Como estructurar para realizar la calculadora
