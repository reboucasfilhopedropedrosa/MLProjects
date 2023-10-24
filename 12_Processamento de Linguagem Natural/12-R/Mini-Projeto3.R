# Machine Learning

# Mini-Projeto 3 - Text Analytics em Recursos Humanos

# Detalhes sobre o projeto no manual em pdf do Capítulo 12.

# Diretório de trabalho
setwd("~/Dropbox/DSA/MachineLearning2.0/Cap12/R")
getwd()

# Pacotes
install.packages("qdap")
install.packages("RWeka")
install.packages("ggthemes")
library(readr)
library(qdap)
library(tm)
library(RWeka)
library(wordcloud)
library(plotrix)
library(ggthemes)
library(ggplot2)

# Carrega os dados
df_amazon <- read_csv("dados/amazon.csv")
df_google <- read_csv("dados/google.csv")

# Visualiza os dados
View(df_amazon)
View(df_google)

# Tipos de dados
str(df_amazon)
str(df_google)

# Dimensões
dim(df_amazon)
dim(df_google)

# Prós e contras da Amazon
amazon_pros <- df_amazon$pros
amazon_cons <- df_amazon$cons

# Prós e contras do Google
google_pros <- df_google$pros
google_cons <- df_google$cons

# Organização do texto

# Função para limpeza do texto
func_limpa_texto <- function(x){
  
  x <- na.omit(x)
  x <- replace_abbreviation(x)
  x <- replace_contraction(x)
  x <- replace_number(x)
  x <- replace_ordinal(x)
  x <- replace_symbol(x)
  x <- tolower(x)

  return(x)
}

# Aplicando limpeza aos dados
amazon_pros <- func_limpa_texto(amazon_pros)
amazon_cons <- func_limpa_texto(amazon_cons)
google_pros <- func_limpa_texto(google_pros)
google_cons <- func_limpa_texto(google_cons)

# O próximo passo é converter o vetor contendo os dados de texto em um corpus. 
# Corpus é uma coleção de documentos, mas também é importante saber que no pacote tm, R o reconhece 
# como um tipo de dado.

# Usaremos o corpus volátil, que é mantido na RAM do computador em vez de salvo no disco, apenas para 
# ter mais eficiência de memória.

# Para criar um corpus volátil, R precisa interpretar cada elemento em nosso vetor de texto como um documento.
# O pacote tm fornece funções para fazer exatamente isso! 
# Usaremos uma função Source chamada VectorSource() porque nossos dados de texto estão contidos em um vetor. 
?VCorpus
amazon_p_corp <- VCorpus(VectorSource(amazon_pros))
amazon_c_corp <- VCorpus(VectorSource(amazon_cons))
google_p_corp <- VCorpus(VectorSource(google_pros))
google_c_corp <- VCorpus(VectorSource(google_cons))

# Agora aplicamos limpeza ao Corpus

# Função de limpeza do Corpus
func_limpa_corpus <- function(x){
  
  x <- tm_map(x,removePunctuation)
  x <- tm_map(x,stripWhitespace)
  x <- tm_map(x,removeWords, c(stopwords("en"), "Amazon", "Google", "Company"))
  
  return(x)
}

# Aplicando a função
amazon_pros_corp <- func_limpa_corpus(amazon_p_corp)
amazon_cons_corp <- func_limpa_corpus(amazon_c_corp)
google_pros_corp <- func_limpa_corpus(google_p_corp)
google_cons_corp <- func_limpa_corpus(google_c_corp)

# Feature Extraction 

# Como amzn_pros_corp, amzn_cons_corp, goog_pros_corp e goog_cons_corp foram pré-processados, 
# agora podemos extrair os recursos que desejamos examinar. 

# Como estamos usando a abordagem do saco de palavras (bag of words), podemos criar um bigrama TermDocumentMatrix 
# para o corpus de avaliações positivas e negativas da Amazon.

# A partir disso, podemos criar rapidamente uma nuvem de palavras para entender quais frases as pessoas 
# associam positivamente e negativamente ao trabalhar na Amazon.

# Tokenização
tokenizer <- function(x) 
?NGramTokenizer
NGramTokenizer(x, Weka_control(min = 2, max = 2))

# Feature extraction e análise de avaliações positivas
amazon_p_tdm    <- TermDocumentMatrix(amazon_pros_corp)
amazon_p_tdm_m  <- as.matrix(amazon_p_tdm)
amazon_p_freq   <- rowSums(amazon_p_tdm_m)
amazon_p_f.sort <- sort(amazon_p_freq, decreasing = TRUE)

# Plot
barplot(amazon_p_freq[1:5])

# Prepara os dados para a wordcloud
amazon_p_tdm    <- TermDocumentMatrix(amazon_pros_corp, control = list(tokenize=tokenizer))
amazon_p_tdm_m  <- as.matrix(amazon_p_tdm)
amazon_p_freq   <- rowSums(amazon_p_tdm_m)
amazon_p_f.sort <- sort(amazon_p_freq,decreasing = TRUE)

# Cria o dataframe de comentários positivos
df_amazon_p <- data.frame(term = names(amazon_p_f.sort), num = amazon_p_f.sort)
View(df_amazon_p)

# Nuvem de palavras
wordcloud(df_amazon_p$term, 
          df_amazon_p$num, 
          max.words = 100, 
          color = "tomato4")

# Feature extraction e análise de avaliações negativas
amazon_c_tdm    <- TermDocumentMatrix(amazon_cons_corp, control = list(tokenize = tokenizer))
amazon_c_tdm_m  <- as.matrix(amazon_c_tdm)
amazon_c_freq   <- rowSums(amazon_c_tdm_m)
amazon_c_f.sort <- sort(amazon_c_freq, decreasing = TRUE)

# Cria o dataframe de comentários negativos
df_amazon_c <- data.frame(term = names(amazon_c_f.sort), num = amazon_c_f.sort)
View(df_amazon_c)

# Nuvem de palavras
wordcloud(df_amazon_c$term,
          df_amazon_c$num,
          max.words = 100,
          color = "palevioletred")

# Parece que há uma forte indicação de longas horas de trabalho e equilíbrio entre trabalho e 
# vida pessoal nas avaliações. Como uma técnica de agrupamento simples, vamos realizar um 
# agrupamento hierárquico e criar um dendrograma para ver como essas frases estão conectadas.
amazon_c_tdm <- TermDocumentMatrix(amazon_cons_corp,control = list(tokenize = tokenizer))
amazon_c_tdm <- removeSparseTerms(amazon_c_tdm, 0.993)

# Cria o dendograma
amazon_c_hclust <- hclust(dist(amazon_c_tdm, method = "euclidean"), method = "complete")

# Plot
plot(amazon_c_hclust)

# Voltando aos comentários positivos, vamos examinar as principais frases que apareceram 
# nas nuvens de palavras. Esperamos agora encontrar termos associados usando a função findAssocs() do pacote tm.
amazon_p_tdm    <- TermDocumentMatrix(amazon_pros_corp, control = list(tokenize=tokenizer))
amazon_p_m      <- as.matrix(amazon_p_tdm)
amazon_p_freq   <- rowSums(amazon_p_m)
token_frequency <- sort(amazon_p_freq,decreasing = TRUE)
token_frequency[1:5]

# Encontramos as associações
findAssocs(amazon_p_tdm, "fast paced", 0.2)

# Vamos criar uma nuvem de palavras comparativa das avaliações positivas e negativas do Google para comparação 
# com a Amazon. Isso dará uma compreensão rápida dos principais termos.
all_google_pros   <- paste(df_google$pros, collapse = "")
all_google_cons   <- paste(df_google$cons, collapse = "")
all_google        <- c(all_google_pros,all_google_cons)
all_google_clean  <- func_limpa_texto(all_google)
all_google_vs     <- VectorSource(all_google_clean) 
all_google_vc     <- VCorpus(all_google_vs)
all_google_clean2 <- func_limpa_corpus(all_google_vc)
all_google_tdm    <- TermDocumentMatrix(all_google_clean2)

# Colnames
colnames(all_google_tdm) <- c("Google Pros", "Google Cons")

# Converte para matriz
all_google_tdm_m <- as.matrix(all_google_tdm)

# Nuvem de comparação
?comparison.cloud
comparison.cloud(all_google_tdm_m, colors = c("orange", "blue"), max.words = 50)

# As críticas positivas da Amazon parecem mencionar bigramas como "bons benefícios", enquanto suas 
# críticas negativas se concentram em bigramas, como questões de "equilíbrio trabalho-vida".

# Em contraste, as análises positivas do Google mencionam "regalias", "pessoas inteligentes", "boa comida" 
# e "cultura divertida", entre outras coisas. As avaliações negativas do Google discutem "política", "crescer", 
# "burocracia" e "média gerência".

# Agora faremos um gráfico de pirâmide alinhando comentários positivos para a Amazon e o Google para que você 
# possa ver adequadamente as diferenças entre quaisquer bigramas compartilhados.
amazon_pro    <- paste(df_amazon$pros, collapse = "")
google_pro    <- paste(df_google$pros, collapse = "")
all_pro       <- c(amazon_pro, google_pro)
all_pro_clean <- func_limpa_texto(all_pro)
all_pro_vs    <- VectorSource(all_pro)
all_pro_vc    <- VCorpus(all_pro_vs)
all_pro_corp  <- func_limpa_corpus(all_pro_vc)

# Matriz termo-documento
tdm.bigram = TermDocumentMatrix(all_pro_corp,control = list(tokenize = tokenizer))

# Colnames
colnames(tdm.bigram) <- c("Amazon", "Google")

# Converte para matriz
tdm.bigram <- as.matrix(tdm.bigram)

# Palavras comuns
common_words <- subset(tdm.bigram, tdm.bigram[,1] > 0 & tdm.bigram[,2] > 0 )

# Diferença
difference <- abs(common_words[, 1] - common_words[,2])

# Vetor final
common_words <- cbind(common_words, difference)
common_words <- common_words[order(common_words[,3],decreasing = TRUE),]

# Dataframe
top25_df <- data.frame(x = common_words[1:25,1], 
                       y = common_words[1:25,2], 
                       labels = rownames(common_words[1:25,]))

# Plot
pyramid.plot(top25_df$x,
             top25_df$y,
             labels=top25_df$labels,
             gap=15,
             top.labels=c("Amazon Pros", "Vs", "Google Pros"),
             unit = NULL,
             main = "Palavras em Comum")

# Os funcionários da Amazon mencionaram o "equilíbrio entre vida pessoal e profissional" como um aspecto positivo. 
# Em ambas as organizações, as pessoas mencionaram "cultura" e "pessoas inteligentes", portanto, há alguns 
# aspectos positivos semelhantes entre as duas empresas.

# Agora vamos voltar a atenção para as avaliações negativas e criar os mesmos recursos visuais.
amazon_cons    <- paste(df_amazon$cons, collapse = "")
google_cons    <- paste(df_google$cons, collapse = "")
all_cons       <- c(amazon_cons,google_cons)
all_cons_clean <- func_limpa_texto(all_cons)
all_cons_vs    <- VectorSource(all_cons)
all_cons_vc    <- VCorpus(all_cons_vs)
all_cons_corp  <- func_limpa_corpus(all_cons_vc)

# Matriz termo-documento
tdm.cons_bigram = TermDocumentMatrix(all_cons_corp,control=list(tokenize =tokenizer))

# Preparação dos dados conforme feito anteriormente
colnames(tdm.cons_bigram) <- c("Amazon", "Google")
tdm.cons_bigram <- as.matrix(tdm.cons_bigram)
common_words <- subset(tdm.cons_bigram, tdm.cons_bigram[,1] > 0 & tdm.cons_bigram[,2] > 0 )
difference <- abs(common_words[, 1] - common_words[,2])
common_words <- cbind(common_words, difference)
common_words <- common_words[order(common_words[,3], decreasing = TRUE),]

# Dataframe
top25_df <- data.frame(x = common_words[1:25,1],
                       y = common_words[1:25,2],
                       labels = rownames(common_words[1:25,]))

# Plot
pyramid.plot(top25_df$x,
             top25_df$y,
             labels=top25_df$labels,
             gap=10,
             top.labels = c("Amazon Cons","Vs","Google Cons"),
             unit = NULL,
             main = "Palavras em Comum")

# Usaremos nuvem de comunalidade para mostrar o que é comum entre Amazon e Google 
# com o tokenizer Unigram, Bigram e Trigram para identificar mais insights.

# Unigram
tdm.unigram <- TermDocumentMatrix(all_pro_corp)
colnames(tdm.unigram) <- c("Amazon","Google")
tdm.unigram <- as.matrix(tdm.unigram)

?commonality.cloud
commonality.cloud(tdm.unigram, colors = c("tomato2", "yellow2"), max.words = 100)

# Bigram
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
tdm.bigram <- TermDocumentMatrix(all_pro_corp,control = list(tokenize=BigramTokenizer))
colnames(tdm.bigram) <- c("Amazon", "Google")
tdm.bigram <- as.matrix(tdm.bigram)

commonality.cloud(tdm.bigram, colors = c("tomato2", "yellow2"), max.words = 100)

# Trigram
TrigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))
tdm.trigram <- TermDocumentMatrix(all_pro_corp,control = list(tokenize=TrigramTokenizer))
colnames(tdm.trigram) <- c("Amazon","Google")
tdm.trigram <- as.matrix(tdm.trigram)

commonality.cloud(tdm.trigram, colors = c("tomato2", "yellow2"), max.words = 100)

# Palavras mais frequentes nos comentários dos funcionários
amazon_tdm <- TermDocumentMatrix(amazon_p_corp)
associations <- findAssocs(amazon_tdm,"fast",0.2)
associations_df <- list_vect2df(associations)[,2:3] 

ggplot(associations_df, aes(y = associations_df[,1])) +
  geom_point(aes(x = associations_df[,2]),
             data = associations_df, size = 3) + 
  theme_gdocs()

google_tdm <- TermDocumentMatrix(google_c_corp)
associations <- findAssocs(google_tdm,"fast",0.2)
associations_df <- list_vect2df(associations)[,2:3] 

ggplot(associations_df,aes(y=associations_df[,1])) +
  geom_point(aes(x = associations_df[,2]),
             data = associations_df, size = 3) + 
  theme_gdocs()


# Conclusão

# O Google tem um melhor equilíbrio entre vida profissional e pessoal de acordo com as avaliações dos funcionários.

# Fim

