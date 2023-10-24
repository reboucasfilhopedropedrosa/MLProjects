# eXtreme Gradient Boosting (XGBoost)

# https://xgboost.readthedocs.io/en/latest/

# O XGBoost é uma biblioteca otimizada de aumento de gradiente distribuído, projetada 
# para ser altamente eficiente, flexível e portátil. Ele implementa algoritmos de 
# aprendizado de máquina sob a estrutura Gradient Boosting. O XGBoost fornece um 
# aumento de árvore paralelo (também conhecido como GBDT, GBM) que resolve muitos 
# problemas de Ciência de Dados de maneira rápida e precisa. O mesmo código é executado 
# no ambiente distribuído (Hadoop, SGE, MPI) e pode resolver problemas com dados de 
# bilhões de registros.

# Amplamente usado nas competições do Kaggle.

# Obs: Caso tenha problemas com a acentuação, consulte este link:
# https://support.rstudio.com/hc/en-us/articles/200532197-Character-Encoding

# Definindo o diretório de trabalho
getwd()
setwd("~/Dropbox/DSA/MachineLearning2.0/Cap09/R")

# Neste exemplo, pretendemos prever se um cogumelo pode ser comido ou não!

# Pacotes
install.packages("xgboost")
install.packages("Ckmeans.1d.dp")
install.packages("DiagrammeR")
require(xgboost)
require(Ckmeans.1d.dp)
require(DiagrammeR)

# Datasets
# https://archive.ics.uci.edu/ml/datasets/mushroom
?agaricus.train
data(agaricus.train, package = 'xgboost')
data(agaricus.test, package = 'xgboost')

# Coletando subsets de treino e de teste
dados_treino <- agaricus.train
dados_teste <- agaricus.test

# Resumo dos dados
str(dados_treino)

# Visualizando as dimensões
dim(dados_treino$data)
dim(dados_teste$data)

# Visualizando os dados
View(dados_treino)
View(dados_teste)

# Classes a serem previstas
class(dados_treino$data)[1]
class(dados_treino$label)

# Construindo o modelo
?xgboost
modelo_v1 <- xgboost(data = dados_treino$data, 
                     label = dados_treino$label, 
                     max.depth = 2, 
                     eta = 1, 
                     nthread = 2, 
                     nround = 2, 
                     objective = "binary:logistic")

# Imprimindo o modelo
modelo_v1

# Matriz Densa
?xgb.DMatrix
dtrain <- xgb.DMatrix(data = dados_treino$data, label = dados_treino$label)
dtrain
class(dtrain)

# Modelo baseado em matriz densa
modelo_v2 <- xgboost(data = dtrain, 
                     max.depth = 2, 
                     eta = 1, 
                     nthread = 2, 
                     nround = 2, 
                     objective = "binary:logistic")

# Imprimindo o modelo
modelo_v2

# Criando um modelo e imprimindo as etapas realizadas
modelo_v3 <- xgboost(data = dtrain, 
                     max.depth = 2, 
                     eta = 1, 
                     nthread = 2, 
                     nround = 2, 
                     objective = "binary:logistic", 
                     verbose = 2)

# Imprimindo o modelo
modelo_v3

# Fazendo previsões
pred <- predict(modelo_v3, dados_teste$data)

# Tamanho do vetor de previsões
print(length(pred))

# Previsões
print(head(pred))

# Transformando as previsões em classificação
prediction <- as.numeric(pred > 0.5)
print(head(prediction))

# Erros
err <- mean(as.numeric(pred > 0.5) != dados_teste$label)
print(paste("test-error = ", err))

# Criando 2 matrizes
dtrain <- xgb.DMatrix(data = dados_treino$data, label = dados_treino$label)
dtest <- xgb.DMatrix(data = dados_teste$data, label = dados_teste$label)

# Criando uma watchlist
watchlist <- list(train = dtrain, test = dtest)
watchlist

# Criando um modeo
?xgb.train
modelo_v4 <- xgb.train(data = dtrain, 
                       max.depth = 2, 
                       eta = 1, 
                       nthread = 2, 
                       nround = 2, 
                       watchlist = watchlist, 
                       objective = "binary:logistic")

# Obtendo o label
label = getinfo(dtest, "label")

# Fazendo previsões
pred <- predict(modelo_v4, dtest)

# Erro
err <- as.numeric(sum(as.integer(pred > 0.5) != label))/length(label)
print(paste("test-error = ", err))

# Criando a Matriz de Importância de Atributos
importance_matrix <- xgb.importance(model = modelo_v4)
print(importance_matrix)

# Plot
xgb.plot.importance(importance_matrix = importance_matrix)

# Dump
xgb.dump(modelo_v4, with_stats = T)

# Plot do modelo
xgb.plot.tree(model = modelo_v4)

# Salvando o modelo
xgb.save(modelo_v4, "xgboost.model")

# Carregando o modelo treinado
bst2 <- xgb.load("xgboost.model")

# Fazendo previsões
pred2 <- predict(bst2, dados_teste$data)
pred2





