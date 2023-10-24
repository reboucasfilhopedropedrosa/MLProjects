# Criando um árvore de decisão a partir do dataset titanic

# Obs: Caso tenha problemas com a acentuação, consulte este link:
# https://support.rstudio.com/hc/en-us/articles/200532197-Character-Encoding

# Definindo o diretório de trabalho
getwd()
setwd("~/Dropbox/DSA/MachineLearning2.0/Cap08/R")

# Gerando o dataset
data(Titanic, package = "datasets") 

# Criando o dataframe
dataset <- as.data.frame(Titanic) 
View(dataset)

# Carregando o pacote
install.packages("rpart")
library(rpart) 

# Criando o modelo
titanic_tree <- rpart(Survived ~ Class + Sex + Age, 
                      data = dataset, 
                      weights = Freq, 
                      method = "class", 
                      parms = list(split = "information"), 
                      control = rpart.control(minsplit = 5)) 

titanic_tree

# Aplicando o Prune
?prune
pruned_titanic_tree <- prune(titanic_tree, cp = 0.02)
pruned_titanic_tree

# Carregando o pacote rpart.plot
install.packages("rpart.plot")
library(rpart.plot) 

# Imprimindo a árvore antes e depois do Prune

# Antes do Pruning
prp(titanic_tree, type = 0, extra = 1, under = TRUE, compress = TRUE)

# Depois do Pruning
prp(pruned_titanic_tree, type = 0, extra = 1, under = TRUE, compress = TRUE)


