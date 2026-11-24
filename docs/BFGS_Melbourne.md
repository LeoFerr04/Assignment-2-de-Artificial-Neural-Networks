# Implementação do BFGS — Previsão das temperaturas de Melbourne

## 1. Objetivo

O objetivo desta componente do projeto é implementar e avaliar o algoritmo de otimização BFGS (*Broyden–Fletcher–Goldfarb–Shanno*) no treino de uma rede neuronal artificial do tipo Multilayer Perceptron (MLP).

A aplicação experimental consiste na previsão da temperatura mínima do dia seguinte em Melbourne, utilizando temperaturas de dias anteriores e características sazonais.

## 2. Preparação dos dados

Foi utilizado o dataset *Daily Minimum Temperatures in Melbourne*, com observações diárias entre 1981 e 1990.

O pré-processamento inclui:

- Carregamento e ordenação cronológica dos dados.
- Identificação de temperaturas inválidas ou ausentes.
- Exclusão de janelas com valores inválidos ou interrupções na sequência diária.
- Construção de janelas temporais com 7 temperaturas anteriores.
- Inclusão de duas características sazonais, calculadas através de seno e cosseno.
- Divisão cronológica dos dados em treino, validação e teste.
- Normalização através da média e do desvio-padrão dos alvos de treino.

Após a preparação, obtiveram-se **3612 amostras válidas**, distribuídas da seguinte forma:

| Conjunto | Amostras | Proporção aproximada |
|---|---:|---:|
| Treino | 2528 | 70% |
| Validação | 541 | 15% |
| Teste | 543 | 15% |

A divisão cronológica evita utilizar observações futuras para treinar o modelo. Os parâmetros de normalização são calculados exclusivamente com o conjunto de treino.

## 3. Arquitetura da rede neuronal

Foi implementada uma MLP com uma camada escondida, ativação `tanh` e saída linear.

Cada amostra possui nove características de entrada: sete temperaturas anteriores e duas características sazonais.

A função objetivo utilizada corresponde a metade do erro quadrático médio:

`L = 0.5 × mean((y_pred - y)^2)`

Os gradientes são calculados analiticamente através da retropropagação (*backpropagation*).

A implementação foi validada por comparação com gradientes numéricos obtidos através de diferenças finitas.

## 4. Implementação do BFGS

O BFGS é um método de otimização quasi-Newton que utiliza informação dos gradientes para atualizar uma aproximação da inversa da matriz Hessiana.

A implementação foi dividida em três componentes principais:

| Ficheiro | Responsabilidade |
|---|---|
| `make_network_objective.m` | Converter os parâmetros da rede e os respetivos gradientes em vetores |
| `bfgs_line_search.m` | Determinar o tamanho do passo através da condição de Armijo |
| `bfgs_optimize.m` | Executar as iterações do algoritmo BFGS |

A pesquisa linear de Armijo procura assegurar uma diminuição suficiente da função objetivo.

O algoritmo inclui ainda verificações da direção de descida, condições de curvatura e critérios de paragem.

Foram realizados testes com funções quadráticas e experiências de integração com a MLP, incluindo a validação dos gradientes e situações de falha controlada.

## 5. Configuração experimental

Para estudar o efeito da arquitetura da rede, foram consideradas três quantidades de neurónios escondidos: **3, 5 e 8**.

Cada arquitetura foi treinada com as sementes aleatórias **7, 42 e 123**, originando nove experiências.

As configurações principais foram:

| Parâmetro | Valor |
|---|---|
| Temperaturas anteriores | 7 dias |
| Características de entrada | 9 |
| Camadas escondidas | 1 |
| Ativação escondida | `tanh` |
| Ativação de saída | Linear |
| Otimizador | BFGS |
| Máximo de iterações | 150 |
| Tolerância do gradiente | `1e-5` |
| Função de perda | Metade do MSE |

A escolha da arquitetura foi efetuada com base nas métricas de validação. O conjunto de teste foi reservado para a avaliação final.

## 6. Resultados experimentais

Os resultados da comparação das arquiteturas foram:

| Neurónios escondidos | RMSE médio de validação (°C) | Desvio-padrão (°C) |
|---|---:|---:|
| **3** | **2,2600** | **0,0050** |
| 5 | 2,2673 | 0,0197 |
| 8 | 2,3015 | 0,0051 |

A arquitetura com três neurónios escondidos apresentou o menor RMSE médio de validação entre as configurações experimentadas.

Para contextualizar os resultados, foram também implementados dois modelos de referência:

| Modelo | RMSE de validação (°C) |
|---|---:|
| Persistência | 2,5925 |
| Regressão linear | 2,3008 |
| MLP com BFGS — 3 neurónios (média de três sementes) | 2,2600 |

A MLP apresentou o menor RMSE médio de validação nesta comparação. Contudo, a diferença relativamente à regressão linear foi pequena e não permite, por si só, concluir que existe uma superioridade estatisticamente significativa.

## 7. Limitações

As nove experiências atingiram o limite de 150 iterações (`max_iter`), sem alcançar o critério de convergência baseado na norma do gradiente.

Assim, os resultados representam o desempenho obtido com o orçamento computacional definido, não necessariamente o mínimo da função objetivo.

O número de arquiteturas e sementes analisadas também é limitado.

A seleção da arquitetura com três neurónios é provisória e deverá ser seguida por uma avaliação final no conjunto de teste, após fixar a configuração do modelo.

## 8. Reprodutibilidade

`Antes da execução, é necessário garantir que o dataset daily-minimum-temperatures-in-me.csv está disponível na pasta data/.`

As experiências podem ser reproduzidas a partir da raiz do projeto, utilizando GNU Octave:

```bash
octave --quiet scripts/run_melbourne_experiments.m
```

O script executa os nove treinos, calcula as métricas de validação e gera automaticamente os ficheiros:

- `results/melbourne/arquiteturas_validacao_reproduzido.csv`
- `results/melbourne/comparacao_arquiteturas_reproduzido.png`

A execução reproduzida confirmou os valores apresentados nas experiências originais.

## 9. Trabalho futuro

As próximas etapas incluem a integração com os restantes componentes desenvolvidos pela equipa, a comparação final dos métodos de treino e a avaliação no conjunto de teste.

Esta documentação descreve apenas a implementação experimental do BFGS e os componentes desenvolvidos e validados nesta fase.