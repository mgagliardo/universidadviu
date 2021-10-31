dat1 <- read.csv("datosLBD/data01.dat", comment = "#", sep = "")
dat2 <- read.csv("datosLBD/data02.dat", comment = "#", sep = "")

# summary: Recuento de los valores
# Min, Max, 1st Quartil, Mediana, 3er Quartil, Media
summary(dat1)
summary(dat2)

# Media:
mean(dat1$Diametro)
mean(dat2$Diam1)
mean(dat2$Diam2)

# Mediana:
median(dat2$Diam1)

# Quantiles (El del 50% es la mediana)
quantile(dat1$Diametro)
# Para pedir el p99
quantile(dat1$Diametro, p=c(0.576, 0.99))

# 
summary(dat1)

# Varianza: Media de las distancias de los datos a la media, al cuadrado
var(dat1$Diametro)

# Desviacion Estandar
sd(dat1$Diametro)

sd(dat1$Diametro) ^ 2 # Idem varianza

# Rango interquartilico
IQR(dat1$Diametro)

quantile(dat1$Diametro, 0.75) - quantile(dat1$Diametro, 0.25)

# Plot
plot(dat1$Diametro)

# Histograma
hist(dat1$Diametro)

histoMio <- hist(dat1$Diametro)

histoMio <- hist(dat1$Diametro, breaks = 20)

# Boxplot
boxMio <- boxplot(dat1$Diametro)


###### Dat2

plot(dat2$Diam1, dat2$Diam2)

# Covarianza
cov(dat2)
var(dat2$Diam1)

# Coeficiente de Correlacion
# Se define a partir de la covarianza dividiendo por las desviaciones tipicas
cor(dat2)

boxplot(dat2)


media <- mean(dat2$Diam1)

desv <- sd(dat2$Diam1)

plot(duration ~ waiting, data = faithful)
abline(lm(duration ~ waiting, data = faithful))



plot(media ~ desv, data = dat2)
abline(v=c(media - desv, media, media + desv), data = dat2col=2)
abline(v=median(dat2$Diam1), col=3)
