# # Proteção de Dados de Clientes com Álgebra Linear

Projeto da Sprint 11 em que aplico álgebra linear e Machine Learning para resolver quatro tarefas da seguradora Proteja Seu Amanhã. A principal delas é proteger os dados pessoais dos clientes sem prejudicar a qualidade dos modelos.

## Objetivos

1. **Clientes semelhantes:** encontrar clientes parecidos com um determinado cliente, para ajudar nas ações de marketing
2. **Classificação:** prever se um novo cliente vai receber algum pagamento de seguro
3. **Regressão:** prever quantos pagamentos de seguro um cliente vai receber
4. **Ofuscação de dados:** mascarar os dados pessoais sem piorar o desempenho do modelo de regressão

## Dados

| Arquivo | Conteúdo |
|---|---|
| `insurance_us.csv` | 5.000 clientes com gênero, idade, renda, número de membros da família e número de pagamentos de seguro recebidos |

## Etapas e resultados

### Exploração dos dados
- Não há valores nulos
- As 153 linhas duplicadas foram mantidas, já que clientes diferentes podem ter os mesmos valores nessas quatro variáveis
- O alvo é desbalanceado: só **11,3%** dos clientes receberam algum pagamento

### Tarefa 1: Clientes semelhantes (kNN)
Testei as distâncias Euclidiana e Manhattan, com e sem escalonamento (`MaxAbsScaler`).
- **Sem escalonamento**, a renda domina o cálculo da distância, por estar em uma escala muito maior. Os vizinhos encontrados têm a mesma renda, mas idades bem diferentes.
- **Com escalonamento**, as quatro variáveis passam a pesar de forma equilibrada e os vizinhos fazem mais sentido.

### Tarefa 2: Classificação
Comparação entre um classificador kNN (k de 1 a 10) e um modelo aleatório:

| Modelo | F1-score |
|---|---|
| Modelo aleatório (melhor caso) | 0,20 |
| kNN com dados originais | 0,00 a 0,61 |
| **kNN com dados escalados** | **0,88 a 0,97** |

O kNN com dados escalados é muito melhor que o modelo aleatório.

### Tarefa 3: Regressão linear implementada do zero
Implementei a regressão linear com a equação normal, `w = (XᵀX)⁻¹Xᵀy`, usando NumPy.

| Dados | REQM | R² |
|---|---|---|
| Originais | 0,34 | 0,43 |
| Escalados | 0,34 | 0,43 |

O escalonamento não altera o resultado, porque a regressão linear ajusta os pesos à escala de cada variável.

### Tarefa 4: Ofuscação de dados
Os dados foram mascarados multiplicando a matriz de características por uma **matriz aleatória invertível P** (`X' = XP`).

- Os dados transformados não permitem identificar idade, renda ou qualquer outra informação pessoal
- Com a matriz P, os dados originais podem ser recuperados (`X = X'P⁻¹`). A diferença máxima foi de apenas 2,2 × 10⁻¹¹, resultado de arredondamento numérico

**Prova analítica:**

```
w_P = [(XP)ᵀ(XP)]⁻¹ (XP)ᵀ y = P⁻¹ w
a_P = (XP) w_P = X P P⁻¹ w = X w = a
```

As previsões são as mesmas, então a qualidade do modelo não muda.

**Prova computacional:** REQM de 0,34 e R² de 0,43, idênticos com dados originais e com dados ofuscados.

## Conclusão

É possível proteger os dados pessoais dos clientes, tornando-os irreconhecíveis, **sem prejudicar a qualidade do modelo de regressão linear**. O projeto também mostra a importância do escalonamento em algoritmos baseados em distância, como o kNN.

## Tecnologias

- Python
- Pandas
- NumPy
- Scikit-learn
- Seaborn

## Como executar

1. Clone o repositório
2. Instale as dependências:
   ```bash
   pip install pandas numpy scikit-learn seaborn jupyter
   ```
3. Abra o notebook:
   ```bash
   jupyter notebook projetosprint11.ipynb
   ```

> Observação: o notebook lê o arquivo em `/datasets/insurance_us.csv`. Ajuste o caminho se o CSV estiver em outra pasta.
