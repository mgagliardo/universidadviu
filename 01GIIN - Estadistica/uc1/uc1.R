# Se trabajará únicamente con 3 columnas (variables), a saber, ‘mpg’, ‘hp’, 
# ‘wt’. Se puede crear una variable con únicamente las tres columnas con la 
# sentencia misDatos<-mtcars[,c('mpg','hp','wt')]

datos <- mtcars[, c('mpg', 'hp', 'wt')]

datosMpg <- datos$mpg
hist(datosMpg, 15)

datosHp <- datos$hp
hist(datosHp, 20)

datosWt <- datos$wt
hist(datosWt)

# b. Realizar una única tabla que contenga la siguiente información para las tres 
# variables: rango, media, mediana, desviación estándar, primer cuartil, tercer cuartil. 

df <- data.frame(
    "Rango" = c(sapply(datos, IQR)),
    "Media" = c(sapply(datos, mean)),
    "Mediana" = c(sapply(datos, median)),
    "Desviacion Estandar" = c(sapply(datos, sd)),
    "Primer Cuartil" = c(sapply(datos, function(x) quantile(x, 0.25))),
    "Tercer Cuartil" = c(sapply(datos, function(x) quantile(x, 0.75))))

View(df)

# c. Pintar ‘mpg’ frente a ‘hp’, y ‘hp’ frente a ‘wt’ 

plot(datosMpg, datosHp)
plot(datosHp, datosWt)
