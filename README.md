# 📈 Tech Challenge - Fase 4: Previsão de Ações com LSTM

> **Deep Learning & AI - FIAP**

Este projeto consiste em uma solução completa de *End-to-End Machine Learning* para prever o preço de fechamento de ações da **CEMIG (CMIG4.SA)**. A solução abrange desde a coleta e pré-processamento de dados financeiros até o treinamento de uma rede neural **LSTM (Long Short-Term Memory)** com **PyTorch**, disponibilizando o modelo final através de uma API **FastAPI** containerizada com **Docker**.

---

## 🚀 Funcionalidades Principais

* **Coleta Automática:** Script para download e limpeza de dados históricos via `yfinance`.
* **Processamento de Séries Temporais:** Normalização e criação de janelas deslizantes para treinamento supervisionado.
* **Deep Learning:** Modelo LSTM implementado em PyTorch para capturar dependências temporais de longo prazo.
* **API RESTful:** Interface web rápida (FastAPI) para inferência em tempo real.
* **Monitoramento:** Endpoints de saúde (`/health`) com métricas de uso de recursos (CPU/Memória) e latência.
* **Reprodutibilidade:** Ambiente isolado via Docker.

---

## 🛠️ Stack Tecnológico

O projeto foi desenvolvido utilizando as seguintes tecnologias e bibliotecas:

* **Linguagem:** Python 3.11
* **Gerenciamento de Dependências:** Poetry
* **Machine Learning:** PyTorch, Scikit-Learn, Numpy, Pandas
* **API Framework:** FastAPI, Uvicorn
* **Containerização:** Docker
* **Fonte de Dados:** Yahoo Finance (yfinance)

---

## 📂 Estrutura do Projeto

A organização de pastas segue princípios de modularidade para separar dados, código de modelagem, scripts de execução e a aplicação web.

```text
/
├── api/                  # Aplicação FastAPI (main.py)
├── data/                 # Armazenamento de dados (brutos e processados)
├── models/               # Artefatos binários (scaler.joblib, lstm_model.pth)
├── results/              # Gráficos de performance e avaliação
├── scripts/              # Pipelines de execução (coleta, treino, avaliação)
├── src/                  # Código fonte reutilizável (classes do modelo e dataset)
├── Dockerfile            # Receita para construção da imagem Docker
├── pyproject.toml        # Gerenciador de dependências Poetry
└── requirements.txt      # Dependências exportadas para o Docker
```
-----

## 📊 Performance do Modelo

O modelo foi avaliado utilizando dados de teste (não vistos durante o treinamento), obtendo os seguintes resultados de precisão para a ação `CMIG4.SA`:

| Métrica | Valor | Descrição |
| :--- | :--- | :--- |
| **MAPE** | **1.56%** | Erro Percentual Absoluto Médio |
| **MAE** | 0.1614 | Erro Médio Absoluto (em R$) |
| **RMSE** | 0.1996 | Raiz do Erro Quadrático Médio |

-----

## ⚡ Como Executar o Projeto

Existem duas formas de executar a aplicação: via **Docker** (recomendado para produção/avaliação) ou **Localmente** (para desenvolvimento).

### Opção 1: Via Docker

Certifique-se de ter o Docker instalado em sua máquina.

1.  **Construir a imagem:**

    ```bash
    docker build -t tech-challenge-lstm .
    ```

2.  **Rodar o container:**

    ```bash
    docker run -d -p 8000:8000 --name lstm-api tech-challenge-lstm
    ```

3.  **Acessar a API:**
    Acesse a documentação automática em: `http://localhost:8000/docs`

-----

### Opção 2: Execução Local (Desenvolvimento)

Pré-requisitos: Python 3.11+ e Poetry.

1.  **Instalar dependências:**

    ```bash
    poetry install
    ```

2.  **Ativar o ambiente virtual:**

    ```bash
    poetry shell
    ```

3.  **Executar o Pipeline de Treinamento (Opcional):**
    Caso queira retreinar o modelo do zero:

    ```bash
    python scripts/01_coleta_dados.py
    python scripts/02_preprocess.py
    python scripts/03_train.py
    python scripts/04_evaluate.py
    ```

4.  **Subir a API:**

    ```bash
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    ```

-----

## 🔌 Utilização da API

### 1\. Verificar Saúde do Sistema (`GET /health`)

Retorna o status da API e consumo de recursos.

**Exemplo de Resposta:**

```json
{
  "status": "healthy",
  "cpu": 1.5,
  "memory": 12.4,
  "model_loaded": true
}
```

### 2\. Realizar Previsão (`POST /predict`)

Recebe uma lista de preços de fechamento anteriores e retorna a previsão para o próximo dia.

**Corpo da Requisição (JSON):**

```json
{
  "last_prices": [12.50, 12.60, 12.55, 12.70, 12.80, ...] 
}
```

> **Nota:** Certifique-se de enviar uma sequência de preços compatível com a janela de tempo utilizada no treinamento.

**Exemplo de Resposta:**

```json
{
  "predicted_price": 12.85
}
```

-----

## 👥 Autores

- Fernando LFS — [GitHub](https://github.com/fernando-lfs) | [LinkedIn](https://www.linkedin.com/in/fernando-lfs/)

---

> Projeto desenvolvido para o FIAP Tech Challenge — Fase 4.