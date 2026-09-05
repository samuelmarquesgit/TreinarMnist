WEBVTT

1
00:00:07.716 --> 00:00:11.720
Hoje a gente vai atacar uma das
partes que é muito negligenciada

2
00:00:11.720 --> 00:00:14.889
no desenvolvimento de
software, que é a documentação.

3
00:00:15.223 --> 00:00:19.102
Não porque ela não seja importante,
pelo contrário, mas porque quase ninguém

4
00:00:19.102 --> 00:00:23.231
tem tempo ou paciência para fazer isso
direito.

5
00:00:23.732 --> 00:00:29.070
Além disso, quanto mais nós usamos IA no
processo de desenvolvimento, ter uma

6
00:00:29.070 --> 00:00:33.783
documentação preparada para IAs e para
agentes se torna mais importante ainda.

7
00:00:34.034 --> 00:00:38.788
Nessa aula você vai aprender como usar IA
para fazer o trabalho pesado por você.

8
00:00:38.997 --> 00:00:44.127
Gerar READMEs profissionais, documentos de
requisitos, especificações de API,

9
00:00:44.419 --> 00:00:46.713
tudo aquilo que a gente evita fazer.

10
00:00:47.172 --> 00:00:49.966
A ideia aqui é bem
simples, se o código muda

11
00:00:49.966 --> 00:00:53.094
rápido, a documentação
também precisa mudar.

12
00:00:53.386 --> 00:00:57.432
E a IA é a peça que faltava para
simplificar esse processo como um todo.

13
00:01:00.894 --> 00:01:05.482
Então antes de ir para a prática, deixa
eu alinhar um conceito muito importante.

14
00:01:05.607 --> 00:01:08.651
Documentação não é um produto estático.

15
00:01:09.277 --> 00:01:14.157
Documentação é um sistema vivo de
comunicação entre desenvolvedores,

16
00:01:14.157 --> 00:01:19.412
times, clientes... e, hoje, até entre
ferramentas que são os próprios agentes da IA.

17
00:01:19.537 --> 00:01:23.708
O problema é que, historicamente,
documentar sempre foi demorado,

18
00:01:23.750 --> 00:01:28.213
repetitivo e ficava muito desatualizado
rapidamente.

19
00:01:28.338 --> 00:01:31.466
Com a IA, isso muda completamente.

20
00:01:31.633 --> 00:01:34.677
A IA consegue ler código,
entender a estrutura,

21
00:01:34.677 --> 00:01:38.139
organizar a informação
e escrever de forma clara.

22
00:01:38.181 --> 00:01:40.391
Essa é a razão de ser de uma LLM.

23
00:01:40.517 --> 00:01:43.353
E hoje a gente vai explorar exatamente
isso.

24
00:01:46.815 --> 00:01:50.110
Então vamos começar pelo básico,
que é o README.

25
00:01:50.193 --> 00:01:51.903
Todo projeto precisa ter um README.

26
00:01:52.112 --> 00:01:56.449
Um bom arquivo README precisa responder
três perguntas imediatamente.

27
00:01:56.449 --> 00:01:57.700
O que esse projeto faz?

28
00:01:57.909 --> 00:01:59.786
Como eu rodo isso localmente?

29
00:01:59.828 --> 00:02:02.080
E como eu uso isso?

30
00:02:02.455 --> 00:02:06.626
Por incrível que pareça, muitos READMES
falham já nessa parte do processo.

31
00:02:07.168 --> 00:02:12.048
Você pode seguir por um prompt básico,
como escreva o README do projeto.

32
00:02:12.340 --> 00:02:16.427
Ou também pode ser bem específico no que
você considera importante ou a sua equipe.

33
00:02:16.469 --> 00:02:20.140
Quais sessões e explicações você quer
incluir.

34
00:02:20.265 --> 00:02:22.350
Eu pessoalmente vou fazer isso agora.

35
00:02:22.433 --> 00:02:25.770
Eu sempre peço para a
IA avaliar o projeto como

36
00:02:25.770 --> 00:02:28.314
um todo antes e me
dizer o que ela entendeu.

37
00:02:28.982 --> 00:02:35.029
Eu faço isso como uma técnica para trazer
à janela de contexto todas as informações

38
00:02:35.029 --> 00:02:39.617
que ela precisa, aumentando a
probabilidade da LLM não deixar passar nada

39
00:02:39.617 --> 00:02:42.954
de importante, conectar todas as
informações necessárias.

40
00:02:43.246 --> 00:02:45.707
Então vamos fazer isso na prática agora.

41
00:02:46.624 --> 00:02:51.546
Aqui eu estou usando o cursor com
integração com o Claude Code.

42
00:02:51.754 --> 00:02:57.427
Então, em vez de usar os agentes do
cursor, eu vou criar isso tudo utilizando

43
00:02:57.427 --> 00:03:01.848
o Claude Code para a gente ter uma ideia da
diferença entre as ferramentas e você

44
00:03:01.848 --> 00:03:05.393
poder escolher o que você considera melhor
no seu dia a dia.

45
00:03:06.311 --> 00:03:07.645
Então eu vou dizer o seguinte.

46
00:03:07.687 --> 00:03:15.278
Preciso gerar um README para esse projeto.

47
00:03:18.156 --> 00:03:32.045
Mas antes, me explique tudo o que você
entende tecnicamente sobre essa aplicação.

48
00:03:32.045 --> 00:03:34.005
Mas antes, me explique tudo o que você
entende tecnicamente sobre essa aplicação.

49
00:03:34.047 --> 00:03:37.091
Então, usando as técnicas
de prompt que a gente já

50
00:03:37.091 --> 00:03:39.761
viu, a gente tenta fazer
ele pensar passo a passo.

51
00:03:40.094 --> 00:03:43.890
Nesse caso específico, eu estou dizendo
para ele primeiro conecte todas as

52
00:03:43.890 --> 00:03:48.353
informações possíveis e depois você pensa
em gerar um arquivo.

53
00:03:48.978 --> 00:03:55.944
Eu poderia usar o modo de planejamento
para ele pensar e só me explicar o que ele

54
00:03:55.944 --> 00:03:59.364
está pensando, mas eu já
estou usando diretamente

55
00:03:59.364 --> 00:04:02.408
o modo agente aqui para
ganhar um certo tempo.

56
00:04:02.450 --> 00:04:08.206
Também tem outras opções aqui como você
pode perguntar antes de deixar ele fazer

57
00:04:08.206 --> 00:04:12.085
qualquer edição ou você deixar
ele editar automaticamente,

58
00:04:12.085 --> 00:04:14.337
além do modo de
planejamento que eu mencionei.

59
00:04:14.462 --> 00:04:17.090
Então eu vou com essa
opção de que ele me pergunte

60
00:04:17.090 --> 00:04:21.094
sempre que ele precisar
fazer alguma alteração.

61
00:04:22.011 --> 00:04:23.972
Então vamos ver o que ele está explicando
aqui.

62
00:04:24.555 --> 00:04:27.558
Ele disse que vai explorar o
projeto de forma abrangente

63
00:04:27.558 --> 00:04:31.062
para entender a arquitetura
e as funcionalidades técnicas.

64
00:04:31.062 --> 00:04:35.149
E aí você pode ver que essa
tarefa está em andamento,

65
00:04:35.149 --> 00:04:38.486
ele está analisando a
arquitetura do projeto.

66
00:04:38.861 --> 00:04:44.325
Você pode abrir aqui para ver exatamente a
linha de raciocínio que ele está seguindo.

67
00:04:45.034 --> 00:04:48.705
Nesse projeto ele está tentando descobrir
quais são as features principais e a

68
00:04:48.746 --> 00:04:54.168
funcionalidade, o banco de dados,
modelos, as rotas da API, autenticação,

69
00:04:54.585 --> 00:04:58.965
arquitetura do front end, quais
componentes que existem, a lógica de

70
00:04:58.965 --> 00:05:03.052
negócios e de serviços, e todos os padrões
que ele consegue perceber.

71
00:05:03.511 --> 00:05:09.726
Então ele vai ser o mais completo possível
nessa análise.

72
00:05:10.351 --> 00:05:14.355
Aí a gente pode ver que ele está rodando
de fato comandos aqui, comandos bash,

73
00:05:14.564 --> 00:05:21.279
buscando por arquivos de dependências,
por configurações do typescript,

74
00:05:21.446 --> 00:05:23.573
nesse projeto específico aqui.

75
00:05:24.532 --> 00:05:31.748
Está procurando todos os arquivos que tem
aqui, por exemplo, dentro do projeto.

76
00:05:32.290 --> 00:05:34.375
Ele está tentando fazer uma leitura geral.

77
00:05:34.709 --> 00:05:38.421
Tentou ler um arquivo de
README, não existe ainda, e

78
00:05:38.421 --> 00:05:41.382
ele começou a ler vários
arquivos dentro do projeto.

79
00:05:41.799 --> 00:05:48.639
Claro, aqui a gente está fazendo um gasto
considerável de tokens, digamos assim,

80
00:05:48.723 --> 00:05:53.186
porque eu estou fazendo ele ler todo o
projeto, mas ao mesmo tempo a gente vai

81
00:05:53.186 --> 00:05:58.691
aumentar muito a capacidade dele de
entender o projeto como um todo,

82
00:05:58.691 --> 00:06:01.944
que é o que é necessário quando alguém vai
escrever a documentação.

83
00:06:02.236 --> 00:06:04.989
Então aqui ele deu o resultado,
a análise técnica completa.

84
00:06:07.075 --> 00:06:11.788
Ele vem trazendo aqui toda a stack
tecnológica para o front end.

85
00:06:11.829 --> 00:06:17.668
Aqui é um projeto que eu iniciei usando o
framework 10stack, o react, tem o wind,

86
00:06:17.919 --> 00:06:20.838
tem a biblioteca de
visualização de dados, que

87
00:06:20.838 --> 00:06:23.800
é o recharts, tudo
que já foi implementado.

88
00:06:23.925 --> 00:06:27.720
Back end em node, com
banco de dados postgres, e

89
00:06:27.762 --> 00:06:31.099
o drizzle como ORM,
tem outras bibliotecas aqui.

90
00:06:31.641 --> 00:06:34.602
DevTools, o que a gente está usando para
rodar o projeto.

91
00:06:34.977 --> 00:06:39.607
Ele consegue encontrar as funcionalidades
principais como o gerenciamento

92
00:06:39.607 --> 00:06:42.819
financeiro, o dashboard que foi criado,
segurança.

93
00:06:43.444 --> 00:06:46.614
Ele vem trazendo o
esquema do banco de dados,

94
00:06:46.614 --> 00:06:50.660
todas as tabelas, campos,
tudo como isso funciona.

95
00:06:51.661 --> 00:06:56.457
Os endpoints da API também estão todos
aqui, quais as funções que aceitam.

96
00:06:56.791 --> 00:07:02.839
Por exemplo, este aqui só tem GET,
por exemplo, para buscar as settings,

97
00:07:02.964 --> 00:07:08.302
as configurações do projeto, enquanto
outras, por exemplo, podem ser editadas.

98
00:07:09.345 --> 00:07:13.266
Aí ele parte para o front end,
começa a falar sobre todos os componentes,

99
00:07:13.266 --> 00:07:15.351
depois ele vai para a lógica de negócio
principal.

100
00:07:17.395 --> 00:07:19.939
Resumindo, ele é extremamente completo.

101
00:07:19.981 --> 00:07:25.403
Ele avaliou o projeto como um todo,
deu toda a opinião dele sobre front end,

102
00:07:25.528 --> 00:07:27.280
sobre back end, sobre a lógica de negócio.

103
00:07:27.572 --> 00:07:30.908
Aqui ele trouxe também
padrões arquiteturais

104
00:07:30.908 --> 00:07:33.161
notáveis, tudo aquilo
que a gente está usando.

105
00:07:33.411 --> 00:07:36.080
Um exemplo bem específico:

106
00:07:36.122 --> 00:07:40.585
A gente está fazendo todo o cálculo de
moeda em centavos.

107
00:07:40.585 --> 00:07:46.424
Existe uma discussão toda se quando você
vai usar valores financeiros, se você

108
00:07:46.424 --> 00:07:51.762
mantém isso como números
com vírgula, decimais, ou se você

109
00:07:51.762 --> 00:07:55.308
mantém isso inteiros em
centavos, que facilita no cálculo.

110
00:07:55.391 --> 00:07:57.977
Aqui ele já está mostrando que a gente fez
essa decisão.

111
00:07:58.394 --> 00:07:59.812
E assim por diante.

112
00:08:00.438 --> 00:08:04.692
Então, ele tem uma visão
geral da estrutura inteira

113
00:08:04.692 --> 00:08:07.528
do projeto, como tudo está
ligado, como tudo funciona.

114
00:08:07.862 --> 00:08:12.658
E agora sim, eu posso pedir para ele
escrever a documentação para o nosso projeto.

115
00:08:12.867 --> 00:08:14.160
Ele até sugeriu:

116
00:08:14.160 --> 00:08:15.828
Agora estou pronto para gerar o README,

117
00:08:15.953 --> 00:08:18.539
quer que eu crie o README completo para o
seu projeto?

118
00:08:18.706 --> 00:08:20.666
Eu vou simplesmente dizer sim.

119
00:08:20.958 --> 00:08:23.669
Porque ele já tem todo o contexto de tudo
aquilo que a gente pediu.

120
00:08:28.716 --> 00:08:33.596
Neste momento, ele começou a de fato gerar
o arquivo README.

121
00:08:33.763 --> 00:08:36.140
Ele tentou acessar o arquivo, viu que não
existe.

122
00:08:37.391 --> 00:08:41.020
Encontrou a página, o diretório onde ele
está.

123
00:08:41.187 --> 00:08:42.855
Está checando para ver se o arquivo
existe

124
00:08:43.356 --> 00:08:46.943
e começou a ler os documentos que a gente
tem aqui.

125
00:08:47.443 --> 00:08:50.363
Então, ele vai pegar tudo
aquilo que ele já tem na

126
00:08:50.363 --> 00:08:53.366
janela de contexto por causa
da nossa conversa anterior.

127
00:08:54.408 --> 00:08:56.911
Vai juntar isso com a estrutura do
projeto.

128
00:08:57.370 --> 00:09:03.751
E disse aqui que ele está criando tudo
aquilo que é necessário.

129
00:09:03.834 --> 00:09:08.422
O Claude Code tem essa particularidade de
inventar palavras ou usar palavras que não

130
00:09:08.422 --> 00:09:11.259
são comumente usadas para dizer que ele
está trabalhando.

131
00:09:11.467 --> 00:09:15.846
Como uma espécie de piada interna dos
desenvolvedores.

132
00:09:16.430 --> 00:09:17.848
Então, agora ele está gerando isso.

133
00:09:17.890 --> 00:09:20.434
E a gente vai esperar para ver o
resultado.

134
00:09:24.438 --> 00:09:24.939
Pronto.

135
00:09:24.939 --> 00:09:27.024
Ele acabou de gerar o arquivo.

136
00:09:27.358 --> 00:09:32.196
E ele está me pedindo permissão se ele
pode, de fato, escrever nesse arquivo.

137
00:09:32.280 --> 00:09:32.989
Eu vou permitir.

138
00:09:33.197 --> 00:09:34.198
Vou dizer que sim.

139
00:09:34.615 --> 00:09:38.995
E aí, a gente pode abrir para ver o que
ele está fazendo.

140
00:09:40.746 --> 00:09:43.207
Bom, aqui, deixa eu primeiro ler a
descrição.

141
00:09:43.499 --> 00:09:46.669
Ele disse: ó, criei um README completo e
profissional para o projeto.

142
00:09:47.044 --> 00:09:51.549
O arquivo inclui a visão geral,
stack tecnológico, arquitetura,

143
00:09:51.632 --> 00:09:56.846
guia de instalação, scripts e comandos,
documentação da API, funcionalidades,

144
00:09:56.846 --> 00:10:02.893
segurança, testes, deploy, roadmap com as
funcionalidades futuras, baseado num

145
00:10:02.893 --> 00:10:05.980
arquivo que a gente já tem de PRD,
que eu vou falar mais à frente,

146
00:10:06.147 --> 00:10:08.316
e ainda, sessões extras.

147
00:10:09.066 --> 00:10:15.239
Como ele foi treinado com muitos arquivos
de código, com muitos projetos disponíveis

148
00:10:15.239 --> 00:10:18.951
na Internet, no GitHub
e outras plataformas, ele

149
00:10:18.951 --> 00:10:22.163
já tem uma ótima noção do
que é um README completo.

150
00:10:22.538 --> 00:10:23.039
Né?

151
00:10:23.039 --> 00:10:29.337
Como eu pedi para ele olhar o projeto em
detalhe, ver tudo que tem aqui para depois

152
00:10:29.337 --> 00:10:33.591
gerar esse arquivo, ele automaticamente
assumiu que, bom, ele está procurando a

153
00:10:33.591 --> 00:10:36.719
versão mais completa possível do arquivo
README.

154
00:10:37.261 --> 00:10:44.268
Você pode, por outro lado, pedir para ele
ser bem conciso, ser bem explícito com

155
00:10:44.268 --> 00:10:48.397
quais sessões você quer incluir,
só me ensina a rodar o projeto local e

156
00:10:48.397 --> 00:10:50.608
como fazer deploy e é só isso que eu
preciso.

157
00:10:50.608 --> 00:10:53.861
O projeto é seu, a
documentação é sua, você vai

158
00:10:53.861 --> 00:10:57.073
ver aquilo que faz sentido
dentro da sua realidade.

159
00:10:57.531 --> 00:11:00.576
Aqui nesse caso, eu estou
procurando justamente

160
00:11:00.576 --> 00:11:05.956
mostrar como ele pode ser
completo na sua abordagem.

161
00:11:06.290 --> 00:11:13.297
Por exemplo, algo que a gente vê muito em
open source é essas tags mostrando quais

162
00:11:13.297 --> 00:11:15.966
são as tecnologias que são
usadas nesse projeto logo

163
00:11:16.008 --> 00:11:17.843
quando você abre, por
exemplo, a página no GitHub.

164
00:11:17.843 --> 00:11:22.890
Ele tem aqui qual a licença do projeto,
como eu estou conectando o banco de dados,

165
00:11:23.140 --> 00:11:25.851
qual o framework que eu
estou usando, a biblioteca de

166
00:11:25.851 --> 00:11:30.481
front end e typescript como
a linguagem deste projeto.

167
00:11:31.273 --> 00:11:35.111
E aí ele traz todas aquelas funções que
ele mencionou, ele inclui, por exemplo,

168
00:11:35.194 --> 00:11:39.657
links para as dependências, para as
bibliotecas, ele está sendo extremamente

169
00:11:39.657 --> 00:11:44.412
completo para mostrar tudo aquilo que ele
pode fazer.

170
00:11:45.371 --> 00:11:47.498
Então ele dá exemplos de código.

171
00:11:47.832 --> 00:11:50.459
Ele mostra como funciona o sistema de
transferências.

172
00:11:50.626 --> 00:11:54.004
E aí ele vem para a parte que,
se você é novo no projeto, é a parte

173
00:11:54.004 --> 00:11:57.717
talvez mais importante, que é
como eu instalo isso localmente,

174
00:11:57.717 --> 00:12:00.845
como eu configuro isso
para rodar na minha máquina.

175
00:12:01.470 --> 00:12:04.849
Então ele tem todos os
scripts, todas as funções,

176
00:12:04.849 --> 00:12:08.936
mostra tudo como funciona
de forma bem completa.

177
00:12:09.478 --> 00:12:12.148
Aí ele entra na parte
mais de produto, as

178
00:12:12.148 --> 00:12:14.859
funcionalidades principais
que a gente tem aqui.

179
00:12:15.276 --> 00:12:17.611
Ele fala sobre a segurança da aplicação.

180
00:12:17.611 --> 00:12:22.825
O que já tem, o que deveria fazer e o que
já existe de documentação.

181
00:12:23.492 --> 00:12:27.413
Ele vai ensinar a fazer
deploy e vai mostrar

182
00:12:27.413 --> 00:12:31.459
ainda um roadmap de
funcionalidades futuras.

183
00:12:33.461 --> 00:12:36.756
Então, se a gente der um passo atrás e
olhar o que ele está fazendo aqui,

184
00:12:36.839 --> 00:12:40.384
ele não é uma ferramenta
que simplesmente vai

185
00:12:40.384 --> 00:12:43.804
pegar o seu código e vai
gerar um arquivo de texto.

186
00:12:43.846 --> 00:12:47.016
Ele está olhando para o projeto como um
todo.

187
00:12:47.600 --> 00:12:48.684
Com um contexto completo.

188
00:12:48.726 --> 00:12:50.436
Isso aqui é um app financeiro.

189
00:12:50.644 --> 00:12:52.271
Então ele já implementa isso.

190
00:12:52.354 --> 00:12:56.859
Ele precisa de segurança, porque é um app
que lida com dinheiro, com valores.

191
00:12:57.067 --> 00:12:58.944
Então ele traz esse assunto à tona.

192
00:12:59.069 --> 00:13:02.573
Ele olha para frente e diz, bom,
se a gente quiser melhorar a segurança e

193
00:13:02.573 --> 00:13:07.495
performance, a gente pode usar,
por exemplo, validação de força de senha,

194
00:13:07.578 --> 00:13:14.251
ou limitações para proteger contra o abuso
de alguém tentando invadir o seu sistema,

195
00:13:15.336 --> 00:13:17.004
usando algum software, por exemplo.

196
00:13:17.588 --> 00:13:23.093
Então ele mostra otimizações de segurança,
de performance, funcionalidades que o

197
00:13:23.093 --> 00:13:25.638
projeto ainda poderia
desenvolver, relatórios

198
00:13:25.721 --> 00:13:28.390
que ele poderia incluir,
novas integrações.

199
00:13:28.599 --> 00:13:32.603
Ele é uma ferramenta completa de software.

200
00:13:32.603 --> 00:13:37.274
Ele avalia o projeto como um todo e
entrega, mostra o que você fez,

201
00:13:37.441 --> 00:13:40.694
o que você está fazendo, o
que você pode fazer no futuro

202
00:13:40.694 --> 00:13:44.031
e ainda te ajuda a fazer isso
sem nenhuma dificuldade.

203
00:13:44.198 --> 00:13:47.535
Então a gente pode ver que Claude Code é
uma ferramenta,

204
00:13:47.535 --> 00:13:50.913
extremamente completa para
desenvolvimento.

205
00:13:50.996 --> 00:13:55.668
Ele vai olhar não só para aquilo que você
já fez no código, vai avaliar aquilo,

206
00:13:55.709 --> 00:13:59.338
analisar como um todo, ele vai te ajudar
no que você está fazendo agora,

207
00:13:59.421 --> 00:14:03.717
vai até escrever código por você,
vai te guiar durante todo esse processo e

208
00:14:03.717 --> 00:14:08.639
também consegue avaliar cenários futuros,
o que você ainda precisa implementar nesse

209
00:14:08.639 --> 00:14:11.308
projeto em termos de
segurança, melhorias de

210
00:14:11.308 --> 00:14:14.186
performance, funcionalidades
que ainda não foram feitas.

211
00:14:14.186 --> 00:14:19.024
Então ele é uma ferramenta muito completa
para desenvolvimento e está se tornando

212
00:14:19.024 --> 00:14:21.902
cada vez mais essencial,
assim como as outras ferramentas

213
00:14:21.902 --> 00:14:24.363
de IA que a gente vem
trabalhando ao longo desse curso.

214
00:14:27.700 --> 00:14:30.786
Uma próxima etapa depois
do README seria a gente

215
00:14:30.786 --> 00:14:34.832
incluir um documento,
por exemplo, como o PRD.

216
00:14:35.457 --> 00:14:41.130
PRD é um documento de descrição de
requisitos de produto.

217
00:14:41.422 --> 00:14:45.551
Você vai pensar não apenas na parte
técnica, você não está escrevendo para um

218
00:14:45.551 --> 00:14:49.471
desenvolvedor que quer instalar a
aplicação localmente, mas você quer

219
00:14:49.471 --> 00:14:54.643
trabalhar na parte de produto,
de funcionalidades, o que o software tem,

220
00:14:54.768 --> 00:14:57.813
o que ele poderia ter, quais são as falhas
dentro desse sistema.

221
00:14:58.105 --> 00:15:02.985
Então ele vai descrever o objetivo do
projeto como um todo, o escopo,

222
00:15:02.985 --> 00:15:08.574
as funcionalidades, regras de negócio,
critérios de sucesso, métricas,

223
00:15:08.574 --> 00:15:12.077
enfim, ele pensa o produto de software
como um todo.

224
00:15:12.494 --> 00:15:16.248
Nesse projeto aqui, a gente
já tem um PRD, então o

225
00:15:16.248 --> 00:15:21.295
que eu vou fazer é que eu
vou apagar esse documento.

226
00:15:23.422 --> 00:15:26.508
Eu vou apagar ele aqui, que deve estar em
docs.

227
00:15:26.926 --> 00:15:35.017
Eu vou apagar esse PRD e eu vou pedir para
o Claude Code gerar novamente esse arquivo,

228
00:15:35.059 --> 00:15:38.187
agora tendo todas as informações que ele
já tem no README.

229
00:15:38.979 --> 00:15:41.482
Dessa vez, eu vou pedir para
ele ser um pouco mais conciso

230
00:15:41.482 --> 00:15:45.361
justamente para a gente ver
essa variação que a gente pode ter.

231
00:15:45.861 --> 00:15:48.238
É claro que ele poderia escrever
um documento gigantesco se

232
00:15:48.280 --> 00:15:52.368
a gente quisesse, mas eu vou
pedir para ele ser bem conciso.

233
00:15:54.703 --> 00:16:00.042
Escreva para mim um PRD para esse projeto.

234
00:16:02.127 --> 00:16:10.427
Seja bem simples e
conciso, com poucas sessões,

235
00:16:10.427 --> 00:16:18.435
apenas para ter uma
visão básica do projeto.

236
00:16:19.478 --> 00:16:24.108
Eu vou mandar esse prompt e a gente vai
ver como o README foi extremamente

237
00:16:24.108 --> 00:16:28.070
completo, mas agora o PRD pode ser um
pouco mais simples.

238
00:16:29.154 --> 00:16:30.739
Pronto, documento gerado.

239
00:16:30.739 --> 00:16:34.368
A gente já tem um prd.markdown aqui.

240
00:16:34.493 --> 00:16:39.915
Eu vou permitir ele fazer essa alteração e
a gente já vai poder ver o resultado.

241
00:16:40.249 --> 00:16:47.339
Aqui ele me disse que tem dez sessões
principais seguindo o guideline de um PRD.

242
00:16:47.464 --> 00:16:51.593
Tem a visão geral, funcionalidades,
requisitos, arquitetura de dados,

243
00:16:51.927 --> 00:16:56.515
experiência do usuário, roadmap
em quatro fases, métricas,

244
00:16:56.515 --> 00:17:00.686
riscos, o que está fora do escopo
e, por último, as referências.

245
00:17:02.062 --> 00:17:08.235
Segundo ele, ele foi muito mais enxuto que
o PRD original, que tinha 734 linhas.

246
00:17:08.402 --> 00:17:11.155
Ele fez um agora com 200 linhas.

247
00:17:11.238 --> 00:17:17.202
Claro que o PRD, por si só, é um documento
um pouco maior, mais complexo,

248
00:17:17.202 --> 00:17:20.622
porque ele tem uma
visão geral do projeto, mas

249
00:17:20.622 --> 00:17:24.501
a gente vai manter ele
um pouco mais conciso.

250
00:17:24.793 --> 00:17:31.008
Aqui a gente pode ver que o documento ficou
completo, mas ele está muito mais enxuto.

251
00:17:31.133 --> 00:17:35.429
Então ele vai trazer funcionalidades
principais, ele vai falar sobre os

252
00:17:35.429 --> 00:17:39.933
requisitos técnicos, já usando só bullet
points, por exemplo.

253
00:17:39.975 --> 00:17:41.894
Ele não vai ficar
colocando textos extras,

254
00:17:41.894 --> 00:17:45.064
muitas explicações,
não vai ser muito prolixo.

255
00:17:45.105 --> 00:17:50.694
Então ele traz a arquitetura de dados,
experiência do usuário, ele mostra até os

256
00:17:50.694 --> 00:17:53.989
atalhos de teclado que estão disponíveis
para o usuário final.

257
00:17:54.406 --> 00:17:57.951
Ele faz também aquele
roadmap que a gente

258
00:17:57.951 --> 00:18:01.288
falou, usando os dados
que já estão no README.

259
00:18:01.538 --> 00:18:03.665
Ele criou métricas de sucesso.

260
00:18:03.665 --> 00:18:10.964
Mais uma vez, ele está usando dados comuns
da Internet, então ele imagina um sistema

261
00:18:10.964 --> 00:18:14.718
padrão, a disponibilidade deveria ser
acima de 99,9%.

262
00:18:14.718 --> 00:18:17.262
Isso pode se aplicar ao seu projeto ou
não.

263
00:18:17.471 --> 00:18:22.392
Você é responsável pela documentação que
ele vai gerar, mas com certeza ele já te

264
00:18:22.392 --> 00:18:27.481
dá uma estrutura pronta muito grande para
acelerar qualquer tipo de processo.

265
00:18:31.276 --> 00:18:37.616
Um último exemplo de documentação que a
gente pode fazer, é criar um diagrama no

266
00:18:37.616 --> 00:18:41.787
Mermaid, que é uma ferramenta bastante
utilizada.

267
00:18:41.954 --> 00:18:43.872
Eu vou pedir para ele
fazer isso, e aí eu vou

268
00:18:43.872 --> 00:18:46.333
explicar um pouco
melhor como isso funciona.

269
00:18:47.084 --> 00:18:55.050
Crie um diagrama do projeto no Mermaid.

270
00:18:55.342 --> 00:18:56.885
Vou pedir para ele fazer isso.

271
00:18:58.095 --> 00:19:04.810
O Mermaid é um formato que é bem parecido
com o markdown, mas ele cria um diagrama,

272
00:19:04.810 --> 00:19:09.022
um gráfico da arquitetura do projeto como
um todo.

273
00:19:09.314 --> 00:19:12.693
Então eu tenho aqui no cursor,
pode ser no VS Code, pode ser em qualquer

274
00:19:12.693 --> 00:19:16.780
outra ferramenta, uma
extensão que é capaz de ler

275
00:19:16.780 --> 00:19:20.450
esse arquivo nesse formato
e mostrar já o diagrama.

276
00:19:20.576 --> 00:19:23.620
Eu tenho, acredito que eu estou usando
essa aqui chamada Mermaid Chart,

277
00:19:23.745 --> 00:19:27.583
você pode encontrar
qualquer extensão que estiver

278
00:19:27.583 --> 00:19:30.460
disponível para mostrar
diagramas como esse.

279
00:19:30.669 --> 00:19:37.551
Então o Mermaid vai ser um documento,
estilo markdown, com um formato

280
00:19:37.551 --> 00:19:41.722
específico, que aí pode
ser lido por uma ferramenta

281
00:19:41.722 --> 00:19:46.310
como essa e vai gerar um
gráfico como a gente tem aqui.

282
00:19:46.894 --> 00:19:51.982
Eu vou mostrar esse processo assim que ele
terminar o diagrama.

283
00:19:52.274 --> 00:19:55.444
Então o arquivo gerado,
eu vou dar autorização para

284
00:19:55.444 --> 00:20:01.241
ele escrever nesse arquivo
chamado diagramas.markdown.

285
00:20:01.241 --> 00:20:04.953
Pode ver que o arquivo
Mermaid é um arquivo markdown,

286
00:20:04.953 --> 00:20:09.041
então ele pode seguir a
mesma estrutura do .md no final.

287
00:20:09.249 --> 00:20:11.251
E ele gerou esse arquivo.

288
00:20:11.335 --> 00:20:12.502
Esse aqui é o formato.

289
00:20:12.920 --> 00:20:16.715
Ele tem uma linguagem bem específica.

290
00:20:16.965 --> 00:20:25.015
Ele usa também essa palavra-chave aqui,
Mermaid, para definir o diagrama.

291
00:20:25.098 --> 00:20:30.395
Então é basicamente uma linguagem para
documentação.

292
00:20:31.480 --> 00:20:38.570
Então eu posso usar isso aqui para abrir
agora esse diagrama do Mermaid.

293
00:20:38.779 --> 00:20:39.780
Vamos fechar aqui.

294
00:20:42.824 --> 00:20:44.660
Pronto, o arquivo foi gerado.

295
00:20:45.244 --> 00:20:51.500
E agora a gente vê o seguinte, o arquivo
.md não pode ser lido pela minha extensão.

296
00:20:51.625 --> 00:20:54.836
Então eu alterei ele para .mmd,
que é o padrão do Mermaid.

297
00:20:55.128 --> 00:20:58.340
Embora você possa ler ele como arquivo
.md, eu alterei.

298
00:20:58.507 --> 00:21:05.305
E aí eu abri aqui, essa opção de
visualizar o diagrama.

299
00:21:05.305 --> 00:21:07.683
E a gente vê que um erro foi gerado.

300
00:21:08.016 --> 00:21:11.979
Então a gente pode avisar
isso para o Claude Code e ele

301
00:21:11.979 --> 00:21:14.606
vai tentar dar um jeito de
corrigir isso para a gente.

302
00:21:16.024 --> 00:21:21.530
O diagrama não pode ser visto.

303
00:21:23.865 --> 00:21:24.950
Pode corrigir?

304
00:21:25.117 --> 00:21:28.620
E aí eu vou colar para ele a mensagem de
erro.

305
00:21:28.620 --> 00:21:36.211
Vamos ver se ele vai ser capaz de corrigir
isso ao vivo e permitir que a gente

306
00:21:36.211 --> 00:21:39.172
visualize o diagrama da arquitetura
inteira do projeto.

307
00:21:39.923 --> 00:21:41.550
E aí, levou alguns segundos.

308
00:21:41.800 --> 00:21:42.926
Está tudo pronto.

309
00:21:43.343 --> 00:21:44.845
Eu vou dar acesso a ele.

310
00:21:45.804 --> 00:21:47.097
E ele corrigiu.

311
00:21:47.973 --> 00:21:50.017
Opa, ele corrigiu e mudou de volta.

312
00:21:50.350 --> 00:21:51.435
Ele ainda está trabalhando.

313
00:21:55.022 --> 00:21:58.025
A gente viu rapidamente aqui o gráfico
sendo gerado.

314
00:21:59.192 --> 00:22:01.820
E aí ele voltou a trabalhar.

315
00:22:02.112 --> 00:22:03.739
Então vamos ver quando ele terminar.

316
00:22:03.989 --> 00:22:09.036
Aqui ele está criando o modelo de dados
ERD.MMD.

317
00:22:09.619 --> 00:22:12.164
Agora ele já usou a extensão correta.

318
00:22:12.622 --> 00:22:13.790
Criou o gráfico.

319
00:22:13.790 --> 00:22:18.503
E a gente já pode ver esse gráfico sendo
utilizado aqui na direita.

320
00:22:18.628 --> 00:22:22.466
Então, por exemplo, aqui
a gente vai ter o diagrama

321
00:22:22.466 --> 00:22:24.760
ERD que mostra as
tabelas do banco de dados.

322
00:22:24.885 --> 00:22:28.805
Você provavelmente já viu um diagrama
desses em outras aplicações.

323
00:22:28.972 --> 00:22:31.933
Aqui tem a tabela de categorias com todos
os campos.

324
00:22:31.975 --> 00:22:33.810
O formato delas.

325
00:22:33.810 --> 00:22:35.687
Se é chave primária, se é chave
estrangeira.

326
00:22:36.271 --> 00:22:38.398
Então o diagrama ERD completo

327
00:22:38.774 --> 00:22:41.485
só lendo os arquivos da sua aplicação.

328
00:22:41.902 --> 00:22:43.236
Então eu vou permitir também.

329
00:22:43.779 --> 00:22:46.073
Ele está criando mais de um arquivo,
na verdade.

330
00:22:47.240 --> 00:22:48.617
Arquitetura geral está aqui.

331
00:22:49.034 --> 00:22:51.244
A gente vai colocar o preview também.

332
00:22:53.413 --> 00:22:55.082
E ele ainda não finalizou.

333
00:22:55.082 --> 00:22:58.960
E ele está trabalhando em uma terceira
aplicação.

334
00:22:58.960 --> 00:23:00.921
Que é o fluxo de autenticação.

335
00:23:01.254 --> 00:23:03.965
Ou seja, eu pedi para ele gerar um arquivo
Mermaid.

336
00:23:04.216 --> 00:23:07.010
E ele já foi mais ambicioso.

337
00:23:07.094 --> 00:23:11.431
E está gerando aqui pelo menos quatro
arquivos de documentação.

338
00:23:11.473 --> 00:23:14.559
De diagramas que a gente precisa

339
00:23:14.684 --> 00:23:16.645
para arquitetura geral, modelo de dados,

340
00:23:17.104 --> 00:23:18.313
fluxo de autenticação,

341
00:23:18.855 --> 00:23:20.273
fluxo de transações,

342
00:23:20.649 --> 00:23:22.651
e arquitetura de componentes.

343
00:23:22.651 --> 00:23:24.236
Já está no quinto arquivo.

344
00:23:25.112 --> 00:23:26.780
Sexto é o sistema de transferências.

345
00:23:26.822 --> 00:23:28.698
Ele basicamente pegou o projeto como um
todo.

346
00:23:28.865 --> 00:23:33.787
E está gerando agora da forma como ele
entende que deve ser feito.

347
00:23:34.871 --> 00:23:36.498
Como qualquer interação com a IA.

348
00:23:36.581 --> 00:23:39.960
Se você quiser ser estrito no seu prompt

349
00:23:40.043 --> 00:23:42.754
você vai pedir para ele gerar um documento
específico,

350
00:23:42.754 --> 00:23:43.380
um arquivo.

351
00:23:43.422 --> 00:23:44.423
E ele vai fazer isso.

352
00:23:44.714 --> 00:23:48.927
Ele sempre assume que você...
Se você está pedindo uma coisa.

353
00:23:48.927 --> 00:23:50.429
É porque você vai seguir nessa linha.

354
00:23:50.637 --> 00:23:53.515
Então ele já está criando vários diagramas
aqui.

355
00:23:53.515 --> 00:23:56.017
Talvez não é o que a gente precise para
esse projeto.

356
00:24:01.481 --> 00:24:02.482
Muito bem.

357
00:24:02.524 --> 00:24:03.817
Claude Code se empolgou.

358
00:24:04.276 --> 00:24:07.404
Escreveu aí dez diagramas para a gente
sobre o projeto.

359
00:24:07.529 --> 00:24:08.530
Como um todo.

360
00:24:08.780 --> 00:24:09.614
Além disso.

361
00:24:09.781 --> 00:24:12.951
Ele atualizou o arquivo inicial dos
diagramas.

362
00:24:13.201 --> 00:24:14.786
Colocando onde está cada um deles.

363
00:24:15.537 --> 00:24:17.497
Também atualizou o arquivo README.

364
00:24:17.789 --> 00:24:20.417
Com todas as informações sobre esses
diagramas.

365
00:24:20.584 --> 00:24:24.171
E já atualizou então toda a nossa
documentação.

366
00:24:24.421 --> 00:24:27.174
Aqui a gente pode ver que ele criou um
índice para os diagramas.

367
00:24:27.174 --> 00:24:29.968
E cada diagrama separadamente.

368
00:24:30.177 --> 00:24:31.011
Então vamos dar uma olhada.

369
00:24:31.011 --> 00:24:33.013
Ver como ficou esse resultado.

370
00:24:33.555 --> 00:24:37.392
A gente vai ter aqui nos diagramas

371
00:24:39.936 --> 00:24:42.189
um diagramas.md,

372
00:24:42.230 --> 00:24:43.398
que é um arquivo markdown...

373
00:24:43.732 --> 00:24:45.317
que vai ter a lista.

374
00:24:45.525 --> 00:24:46.485
Opa.

375
00:24:46.943 --> 00:24:52.282
A lista de todos os diagramas que a gente
tem.

376
00:24:52.407 --> 00:24:53.992
Então aqui tem um link para todos eles.

377
00:24:54.117 --> 00:24:55.535
Vamos abrir um deles.

378
00:24:55.702 --> 00:24:56.703
Por exemplo:

379
00:24:57.954 --> 00:25:01.416
arquiteturageral.mmd. Vamos abrir esse

380
00:25:01.583 --> 00:25:03.877
e pedir para visualizar ele

381
00:25:04.461 --> 00:25:05.504
no Mermaid.

382
00:25:05.921 --> 00:25:10.842
E aqui está a visualização da arquitetura
do projeto como um todo.

383
00:25:11.259 --> 00:25:14.137
A gente tem o cliente no browser.

384
00:25:14.346 --> 00:25:17.974
A gente tem o back end e o front end aqui
na aplicação.

385
00:25:18.141 --> 00:25:19.476
E a gente tem o banco de dados.

386
00:25:19.643 --> 00:25:22.229
Então ele já criou um diagrama
automaticamente

387
00:25:22.812 --> 00:25:24.356
só lendo o contexto do seu projeto.

388
00:25:24.606 --> 00:25:25.732
E ele criou vários outros.

389
00:25:25.732 --> 00:25:27.025
Fluxo de deploy...

390
00:25:27.442 --> 00:25:30.445
Ele criou vários diagramas que você pode
abrir

391
00:25:30.570 --> 00:25:33.532
e ver como que funciona cada etapa.

392
00:25:33.823 --> 00:25:35.492
Isso aqui é bem importante.

393
00:25:35.867 --> 00:25:36.743
Bem interessante.

394
00:25:36.785 --> 00:25:38.078
Isso pode ser jogado depois...

395
00:25:38.245 --> 00:25:39.412
Mantido aqui dentro.

396
00:25:39.579 --> 00:25:43.875
ou jogado para um diretório onde outras
pessoas tenham acesso também.

397
00:25:43.959 --> 00:25:46.962
Quando quiserem saber sobre o projeto que
você está criando.

398
00:25:47.337 --> 00:25:49.965
Então a gente criou bastante documentação
aqui já.

399
00:25:50.674 --> 00:25:53.969
Agora é novamente a sua vez de escrever
documentação.

400
00:25:54.761 --> 00:25:57.180
Pegue o projeto que a gente está
construindo no curso,

401
00:25:57.222 --> 00:25:58.807
ou um projeto pessoal, seu,

402
00:25:59.099 --> 00:26:03.186
e peça para a IA gerar um arquivo README
bem detalhado,

403
00:26:03.270 --> 00:26:06.731
ou um arquivo OpenAPI para sua API,

404
00:26:06.815 --> 00:26:07.899
para o Swagger,

405
00:26:08.066 --> 00:26:11.069
ou um diagrama simples de arquitetura,
como a gente fez no Mermaid.

406
00:26:11.611 --> 00:26:15.198
Depois revise o resultado com um olhar
crítico.

407
00:26:15.240 --> 00:26:19.327
Conhecendo o seu projeto você é capaz de
dizer se a documentação está certa ou não.

408
00:26:19.828 --> 00:26:22.289
Veja se as instruções estão claras,

409
00:26:22.289 --> 00:26:23.582
se elas fazem sentido,

410
00:26:23.582 --> 00:26:26.751
e se alguém de fora do
projeto conseguiria usar

411
00:26:26.751 --> 00:26:30.088
só com essa documentação
que você está criando.

412
00:26:30.547 --> 00:26:32.966
Faça também uma modificação no seu
projeto

413
00:26:33.133 --> 00:26:36.219
e peça para ele atualizar a documentação
de acordo,

414
00:26:36.219 --> 00:26:38.555
caso ele já não faça isso automaticamente.

415
00:26:38.763 --> 00:26:39.764
Me agradeça depois.

416
00:26:40.015 --> 00:26:42.434
Seus projetos vão estar muito melhor
documentados.

417
00:26:42.767 --> 00:26:45.604
Hoje você viu que a
documentação não precisa

418
00:26:45.604 --> 00:26:47.897
mais ser um fardo no
desenvolvimento de software.

419
00:26:47.939 --> 00:26:50.317
Com a IA, ela deixa de ser uma tarefa manual

420
00:26:50.317 --> 00:26:54.237
e passa a ser parte natural do fluxo de
desenvolvimento.