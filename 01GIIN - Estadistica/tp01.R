# Se trabajará únicamente con 3 columnas (variables), a saber, ‘mpg’, ‘hp’, 
# ‘wt’. Se puede crear una variable con únicamente las tres columnas con la 
# sentencia misDatos<-mtcars[,c('mpg','hp','wt')]

datos <- mtcars[, c('mpg', 'hp', 'wt')]

# Responder a las siguientes cuestiones de manera razonada (se valorará cada una 
# con 2 puntos sobre 10) 
# a. Realizar un histograma de cada una de las tres variables 

datosMpg <- mtcars$mpg
datosHp <- mtcars$hp
datosWt <- mtcars$wt

hist(datosMpg)
hist(datosHp)
hist(datosWt)

# b. Realizar una única tabla que contenga la siguiente información para las tres 
# variables: rango, media, mediana, desviación estándar, primer cuartil, tercer cuartil. 

### WIP
columnas <- c("mpg", "hp", "wt")
filas    <- c("rango", "media", "mediana", "desviacion estandar", "primer cuartil", "tercer cuartil")
dat      <- as.data.frame(cbind(columnas, filas))

# test data frame 2 (this one has fewer rows than the first data frame)
c <- c(IQR(datosMpg), IQR(datosHp), IQR(datosWt))
dat.new <- as.data.frame(c)

# add column to each data frame with row number
dat$number <- row.names(dat)
dat.new$number <- row.names(dat.new)

# merge data frames
# "all = TRUE" will mean that NA values will be added whenever there is no match
finaldata <- merge(dat, dat.new, by = "number", all = TRUE)


tab <- matrix(rep(times = 6), ncol = 3, byrow = TRUE)
colnames(tab) <- c("mpg", "hp", "wt")
rownames(tab) <- c("rango", "media", "mediana", "desviacion estandar", "primer cuartil", "tercer cuartil")
tab <- as.table(tab)


# c. Pintar ‘mpg’ frente a ‘hp’, y ‘hp’ frente a ‘wt’ 


# d. En base a la información de los tres apartados anteriores (gráficos y 
# estadísticos) e inspeccionando los datos o realizando cálculos adicionales 
# en caso necesario, indicar razonadamente la presencia de outliers, y 
# describirlos. 
# e. Utilizando la información de los apartados anteriores e inspeccionando los 
# datos o realizando cálculos adicionales en caso necesario, describir 
# razonadamente las relaciones entre las tres variables propuestas.
