# Estudo de Caso 1 - Features Engineering com Variáveis Categóricas na Prática

# Definindo o diretório de trabalho
# setwd("˜/Dropbox/DSA/EstudoCaso1")
# getwd()

# Os modelos de aprendizado de máquina têm dificuldade em interpretar dados categóricos; 
# Mas a engenharia de recursos nos permite re-contextualizar nossos dados categóricos para 
# melhorar o rigor de nossos modelos de Machine Learning. A engenharia de recursos também 
# fornece camadas adicionais de perspectiva para a análise de dados. A grande questão que as 
# abordagens de engenharia de recursos resolvem é: como utilizar meus dados de maneiras 
# interessantes e inteligentes para torná-los muito mais úteis? 

# A engenharia de recursos não trata de limpar dados, remover valores nulos ou outras tarefas 
# semelhantes (isso é Data Wrangling); a engenharia de recursos tem a ver com a alteração de 
# variáveis para melhorar a história que elas contam. 

# Vejamos alguns exemplos de tarefas de engenharia de atributos!

# Para este estudo de caso usaremos este dataset com dados bancários de usuários:

# Dataset: http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip

# Carregando os dados
dataset_bank <- read.table("bank/bank-full.csv", header = TRUE, sep = ";")
View(dataset_bank)

# Exemplo 1 - Criação de Nova Coluna

# Muitas vezes, quando você usa dados categóricos como preditores, pode achar que alguns dos 
# níveis dessa variável têm uma ocorrência muito escassa ou que os níveis das variáveis são 
# seriamente redundantes.

# Qualquer decisão que você tome para começar a agrupar os níveis de variáveis deve ser 
# estrategicamente orientada. Um bom começo aqui para ambas as abordagens é a função table() em R.
table(dataset_bank$job)

# A ideia seria identificar a ocorrência de um nível com poucos registros ou alternativamente 
# compartimentos que parecem mais indicativos do que os dados estão tentando informar.

# Às vezes, uma tabela é um pouco mais difícil de ingerir; portanto, jogar isso em um gráfico 
# de barras pode ser mais fácil.
library(dplyr)
library(ggplot2)

dataset_bank %>%
  group_by(job)%>%
  summarise(n = n())%>%
  ggplot(aes(x = job, y = n))+
  geom_bar(stat = "identity")+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Para esse estudo de caso, digamos que realmente queremos entender a profissão (job) de acordo 
# com o uso da tecnologia em uma determinada função. Nesse caso, começaríamos a classificar 
# cada uma das profissões em nível médio, alto e baixo em termos de uso de tecnologia.

# Uma função que você pode usar é a mutate do dplyr é muito útil quando você está reatribuindo 
# muitos níveis diferentes de uma variável, em vez de usar alguma função ifelse aninhada. 
# Essa função também é muito útil ao converter variáveis numéricas em dados categóricos. 

dataset_bank <- dataset_bank %>%
  mutate(technology_use = 
           case_when(job == 'admin' ~ "medio",
                     job == 'blue-collar' ~ "baixo",
                     job == 'entrepreneur' ~ "alto",
                     job == 'housemaid' ~ "baixo",
                     job == 'management' ~ "medio",
                     job == 'retired' ~ "baixo",
                     job == 'self-employed' ~ "baixo",
                     job == 'services' ~ "medio",
                     job == 'student' ~ "alto",
                     job == 'technician' ~ "alto",
                     job == 'unemployed' ~ "baixo",
                     job == 'unknown' ~ "baixo"))

View(dataset_bank)

# Como você pode ver acima, criamos um novo campo chamado technology_use e atribuímos a cada 
# um valor de acordo com seu uso de tecnologia. Tenho certeza de que você poderia argumentar 
# tarefas diferentes para cada uma delas, mas para este estudo de caso é suficiente.

# Agora vamos revisar rapidamente esse novo campo.
table(dataset_bank$technology_use)

# Vamos colocar isso em percentual
round(prop.table(table(dataset_bank$technology_use)),2)


# A distribuição deve depender do que você está tentando entender. Digamos que a granularidade 
# do trabalho foi muito maior e tivemos vários jobs relacionados a marketing, analista de marketing, 
# gerente de marketing digital etc.
# Aproveite a tabela e os gráficos de barras para obter uma melhor 
# classificação de níveis das variáveis.


# Exemplo 2 - Variáveis Dummies 

# A coluna default representa se um usuário entreou ou não no cheque especial.
# Em vez de deixar os níveis da variável padrão como "sim" e "não", 
# codificaremos como uma variável fictícia (dummy). 

# Uma variável dummy é a representação numérica de uma variável categórica. 
# Sempre que o valor padrão for sim, codificaremos para 1 e 0 caso contrário. 
# Para duas variáveis de nível mutuamente exclusivas, isso elimina a necessidade de uma 
# coluna adicional, pois está implícito na primeira coluna.
dataset_bank <- dataset_bank %>%    
  mutate(defaulted = ifelse(default  == "yes", 1, 0))

View(dataset_bank)


# Exemplo 3 - One-Hot Encoding

# Falamos sobre a criação de uma única coluna como uma variável dummy, mas devemos falar 
# sobre a codificação one-hot. 

# Uma codificação One-Hot é efetivamente a mesma coisa que fizemos no item anterior, mas para variáveis 
# de muitos níveis em que a coluna possui 0s em todas as linhas, exceto onde o valor corresponde 
# à nova coluna, que seria 1.

# Seria algo assim:

# 0000000001 - indica um valor
# 0000000010 - indica outro valor

library(caret)
?dummyVars
dmy <- dummyVars(" ~ .", data = dataset_bank)
bank.dummies <- data.frame(predict(dmy, newdata = dataset_bank))
View(bank.dummies)

# Acima, carregamos o pacote caret, executamos a função dummyVars para todas as variáveis e, 
# em seguida, criamos um novo dataframe, dependendo das variáveis codificadas identificadas.
# Vamos dar uma olhada na nova tabela:
View(dataset_bank)
str(bank.dummies)
View(bank.dummies)


# Não incluímos todas as colunas, e você pode ver que isso deixou a coluna idade (age) no formato original.


# Exemplo 4 - Combinando Recursos ou Cruzamento de Recursos

# O cruzamento de recursos é onde você combina diversas variáveis. 
# Às vezes, a combinação de variáveis pode produzir um desempenho preditivo que executa o 
# que eles poderiam fazer isoladamente.

# Assim, podemos fazer um agrupamento por duas variáveis por exemplo, com a devida contagem:

dataset_bank %>% 
  group_by(job, marital) %>%
  summarise(n = n())


# Uma visualização disso geralmente é muito mais fácil de interpretar
dataset_bank %>% 
  group_by(job, marital) %>%
  summarise(n = n()) %>%
  ggplot(aes(x = job, y = n, fill = marital))+
  geom_bar(stat = "identity", position = "dodge") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


# Uma avaliação que geralmente é muito mais fácil de interpretar
dmy <- dummyVars( ~ job:marital, data = dataset_bank)
bank.cross <- predict(dmy, newdata = dataset_bank)
View(bank.cross)

# Lembre-se de que, ao combinar diversas variáveis, você pode ter alguns desses novos valores 
# muito esparsos. Revise as saídas e se necessário aplique alguma outra técnica mencionada anteriormente.


# Conclusão

# Existem muitos métodos adicionais que podem ser usados para variáveis numéricas e combinações de 
# numérico e categórico; podemos usar o PCA, entre outras coisas, para melhorar o poder preditivo das 
# variáveis explicativas.




