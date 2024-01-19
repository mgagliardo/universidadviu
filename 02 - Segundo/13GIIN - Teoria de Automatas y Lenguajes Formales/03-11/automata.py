#funciones auxiliares
#recuperar las transciones salidas de un estado
def trans_estado(estado,transiciones):
    res=[]
    for trans in transiciones:
        if(trans[0] == estado):
            tran_est = []
            res.append(trans)
    return res
#funcion para determinar si en un estado se reconoce un simbolo
def disparable(s,transiciones):
    val = -1
    i=0
    for trans in transiciones:
        if(s == trans[1]):
            val = i
        i=i+1
    return val 


#lectura del automata
# abrir archivo a leer
fichero = open('automata1.txt','r')
lineas = fichero.readlines()
aut_transiciones=[]
for linea in lineas:
    linea=linea.rstrip('\n')
    if(linea[0] == '#'):
        entrada=linea
    else:
        match entrada:
            case '#estados':
                aut_estados = linea.split(',')
            case '#alfabeto':
                aut_alfabeto = linea.split(',')
            case '#inicial':
                aut_estado_inicial = linea
            case '#terminales':
                aut_terminales = linea.split(',')
            case '#transiciones':
                transicion = linea.split(',')
                aut_transiciones.append(transicion)




#motor para los automatas

estado_actual = aut_estado_inicial
res = trans_estado(estado_actual,aut_transiciones)
cadena = input()
reconocida = True
for c in cadena:
    nro_trans = disparable(c,res)
    if nro_trans == -1:
        reconocida = False
        break
    estado_actual = res[nro_trans][2]
    res = trans_estado(estado_actual,aut_transiciones)
    print(estado_actual)
print('reconocida ', reconocida)



        




