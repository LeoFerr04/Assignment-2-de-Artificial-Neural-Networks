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

Os resultados de validação dos três modelos de referência e da MLP com BFGS estão na tabela seguinte.

| Modelo | RMSE de validação (°C) | MAE de validação (°C) | R² de validação |
|---|---:|---:|---:|
| Persistência | 2,5925 | — | — |
| AR(7) com intercepto | 2,359835 | 1,880539 | 0,655292 |
| Regressão linear sazonal com intercepto | 2,300801 | 1,863624 | 0,672323 |
| MLP com BFGS — 3 neurónios (média de três sementes) | 2,2600 | — | — |

Os traços assinalam métricas sem valor registado nesta comparação. O AR(7) utiliza as sete temperaturas anteriores. A regressão linear sazonal e a MLP usam também o seno e o cosseno do ciclo anual, num total de nove características. Os dois modelos lineares foram ajustados em 2528 amostras de treino e avaliados em 541 amostras de validação. Os resultados foram confirmados por decomposição QR.

A MLP apresentou o menor RMSE médio de validação nesta comparação. A diferença face à regressão linear sazonal foi de cerca de 0,041 °C. A significância estatística dessa diferença exigiria uma análise própria.

## 7. Diagnóstico de convergência

O diagnóstico comparou os limites de 150, 300 e 600 iterações para a MLP com três neurónios escondidos e as sementes 7, 42 e 123. Para cada semente, os três treinos partiram dos mesmos parâmetros iniciais. Cada execução usou uma janela de sete dias, a normalização calculada com os alvos de treino e uma tolerância do gradiente de `1e-5`. A otimização usou o conjunto de treino e as métricas foram calculadas no conjunto de validação.

| Limite de iterações | RMSE médio de validação (°C) | Desvio-padrão amostral entre sementes (°C) |
|---:|---:|---:|
| 150 | 2,260015 | 0,004953 |
| 300 | 2,259227 | 0,012810 |
| 600 | 2,260112 | 0,016884 |

O limite de 300 iterações produziu o menor RMSE médio, com uma redução de cerca de 0,00079 °C face ao limite de 150. Com 600 iterações, o RMSE médio ficou ligeiramente acima dos valores obtidos com 150 e 300.

A diferença observada entre 150 e 300 iterações é muito pequena e foi obtida com apenas três sementes aleatórias. Por esse motivo, os resultados não permitem estabelecer uma vantagem robusta da utilização de 300 iterações. Além disso, o aumento do número de iterações não conduziu a uma melhoria consistente do desempenho de validação em todas as sementes.

A única execução que atingiu a tolerância do gradiente foi a da semente 123 com limite de 600 iterações. Parou na iteração 594 com o estado `grad_tol`. Para essa semente, o RMSE de validação subiu de 2,262815 °C com 150 iterações para 2,274683 °C no ponto de paragem. Neste caso, atingir o critério de convergência do treino coincidiu com um RMSE de validação mais elevado.

Mantém-se provisoriamente o limite de 150 iterações nas experiências principais. Foi a configuração com menor dispersão do RMSE entre sementes e requer o menor orçamento de iterações por treino.

## 8. Limitações

As nove experiências da comparação de arquiteturas atingiram o limite de 150 iterações (`max_iter`), sem alcançar o critério de convergência baseado na norma do gradiente.

Assim, os resultados representam o desempenho obtido com o orçamento computacional definido, não necessariamente o mínimo da função objetivo.

O número de arquiteturas e sementes analisadas também é limitado.

A seleção da arquitetura com três neurónios é provisória e deverá ser seguida por uma avaliação final no conjunto de teste, após fixar a configuração do modelo.

## 9. Reprodutibilidade

Antes da execução, é necessário garantir que o dataset `daily-minimum-temperatures-in-me.csv` está disponível na pasta `data/`.

As experiências podem ser reproduzidas a partir da raiz do projeto, utilizando GNU Octave:

```bash
octave --quiet scripts/run_melbourne_experiments.m
```

O script executa os nove treinos, calcula as métricas de validação e gera automaticamente os ficheiros:

- `results/melbourne/arquiteturas_validacao_reproduzido.csv`
- `results/melbourne/comparacao_arquiteturas_reproduzido.png`

A execução reproduzida confirmou os valores apresentados nas experiências originais.

### Modelos de referência lineares

A partir da raiz do projeto, executa os dois scripts no GNU Octave para obter as métricas de validação.

```bash
octave --quiet scripts/run_melbourne_ar7_baseline.m
octave --quiet scripts/run_melbourne_linear_seasonal_baseline.m
```

### Diagnóstico de convergência

A partir da raiz do projeto, executa os seguintes comandos no GNU Octave:

```bash
octave --quiet scripts/diagnose_bfgs_convergence.m
octave --quiet scripts/compare_bfgs_iteration_limits.m
```

O primeiro comando compara os três limites para a semente 7 e mostra a perda de treino, a norma do gradiente e as métricas de validação. O segundo repete a comparação para as sementes 7, 42 e 123 e apresenta o RMSE médio e o desvio-padrão por limite. Ambos apresentam os resultados no terminal.

## 10. Trabalho futuro

As próximas etapas incluem a integração com os restantes componentes desenvolvidos pela equipa, a comparação final dos métodos de treino e a avaliação no conjunto de teste.

Esta documentação descreve apenas a implementação experimental do BFGS e os componentes desenvolvidos e validados nesta fase.