# Artificial Neural Networks — Assignment 2

**Universidade da Madeira | Ano Letivo 2026/2027**

## Elementos do Grupo
- Leonardo Ferreira
- Nataniel Álvares

## Objetivo

Implementar e comparar três arquiteturas de redes neuronais artificiais para problemas de regressão, avaliando o impacto da arquitetura e do algoritmo de otimização no desempenho preditivo, convergência e custo computacional.

O projeto foi desenvolvido em MATLAB, com compatibilidade com GNU Octave.

## Datasets

**1. Concrete Compressive Strength**
- 1030 observações.
- 8 variáveis de entrada.
- Objetivo: prever a resistência à compressão do betão, em MPa.

**2. Daily Minimum Temperatures in Melbourne**
- 3650 observações diárias.
- Janela temporal de 7 temperaturas anteriores e 2 atributos sazonais.
- Objetivo: prever a temperatura mínima do dia seguinte, em °C.

## Arquiteturas

| Arquitetura | Configuração |
|---|---|
| Reference | 1 camada oculta com 10 neurónios |
| Wide | 1 camada oculta com 100 neurónios |
| Deep | 10 camadas ocultas com 10 neurónios cada |

Todas as arquiteturas utilizam a mesma função de ativação nas camadas ocultas e uma saída linear.

## Algoritmos de Treino

- Mini-batch Gradient Descent
- Adam
- BFGS

Os algoritmos são comparados utilizando um orçamento de 200 épocas equivalentes, com controlo das avaliações efetuadas pelo BFGS.

## Metodologia Experimental

- Divisão dos dados em treino, validação e teste.
- Normalização baseada exclusivamente no conjunto de treino.
- Afinação de 6 taxas de aprendizagem, com 5 repetições por configuração, para Adam e Gradient Descent.
- Avaliação de 18 combinações de datasets, arquiteturas e algoritmos.
- 30 execuções finais por combinação, com sementes de 1 a 30.
- Métricas: MSE, RMSE, MAE, R² e tempo de execução.
- Comparação com modelos de referência e análise gráfica dos resultados.

## Principais Resultados

| Dataset | Melhor configuração | RMSE média |
|---|---|---|
| Concrete | Wide + BFGS | 4,4434 MPa |
| Melbourne | Wide + Gradient Descent | 2,1732 °C |

Os resultados apresentados correspondem à média de 30 execuções independentes de cada configuração.

## Estrutura do Repositório

```text
Assignment-2-de-Artificial-Neural-Networks/
├── ann_lib.m
├── main_assignment2.m
├── data/
│   └── daily-minimum-temperatures-in-me.csv
├── src/
│   └── concrete/
│       └── Concrete_Data.csv
└── README.md
```

## Como Executar

**Requisitos:** GNU Octave ou MATLAB, com acesso aos datasets incluídos no repositório.

1. Abrir o GNU Octave ou MATLAB.
2. Definir a pasta principal do repositório como diretório de trabalho.
3. Executar:

```matlab
main_assignment2
```

Por predefinição, o programa executa o estudo completo, incluindo afinação de hiperparâmetros, treino dos modelos finais e avaliação.

**Atenção:** a execução completa pode demorar, devido às 900 experiências de treino. O programa utiliza checkpoints para permitir retomar experiências já realizadas.

Os resultados são guardados na pasta `results_compliant`, criada automaticamente.

## Ficheiros Principais

- **`ann_lib.m`** — biblioteca com carregamento e pré-processamento dos dados, redes neuronais, algoritmos de otimização, avaliação e visualização.
- **`main_assignment2.m`** — script principal que executa as experiências.

## Reprodutibilidade

As divisões dos datasets e as sementes aleatórias são controladas para permitir comparações consistentes entre as arquiteturas e os algoritmos.

O conjunto de teste é utilizado para a avaliação final, depois da seleção dos hiperparâmetros.

## Contexto Académico

Trabalho desenvolvido no âmbito da unidade curricular de **Redes Neuronais Artificiais**, da Universidade da Madeira, no ano letivo de 2026/2027.
