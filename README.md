
# 📈 Tech Challenge - Fase 4: Previsão de Ações com MLOps

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c?style=for-the-badge&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Container-2496ed?style=for-the-badge&logo=docker)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?style=for-the-badge&logo=mlflow)
![Pytest](https://img.shields.io/badge/Pytest-Testing-yellow?style=for-the-badge&logo=pytest)

> **Pós-Graduação em Deep Learning & AI - FIAP**

Este projeto apresenta uma solução completa de **End-to-End Machine Learning** para a previsão de preços de fechamento de ações da **CEMIG (CMIG4.SA)**.

A arquitetura abrange desde a engenharia de dados até o deploy produtivo, utilizando **LSTM (Long Short-Term Memory)** para modelagem temporal, **MLflow** para rastreamento de experimentos e **FastAPI** para servir o modelo, tudo orquestrado via **Docker**.

---

## 🚀 Funcionalidades e Diferenciais

*   **Pipeline Automatizado:** Scripts modulares para ETL, Treinamento e Avaliação.
*   **Deep Learning Moderno:** Uso de **PyTorch Lightning** para estruturar o código de treino e garantir reprodutibilidade.
*   **API Inteligente (Drift Detection):** O endpoint de predição monitora estatisticamente a entrada. Se os dados desviarem do padrão de treino (ex: alta volatilidade), um alerta é retornado no JSON de resposta.
*   **Treino Assíncrono:** Capacidade de retreinar o modelo em background (`BackgroundTasks`) sem bloquear a API.
*   **MLOps & Tracking:** Integração nativa com **MLflow** para registrar métricas detalhadas (**MAE, RMSE, MAPE**), hiperparâmetros e artefatos do modelo.
*   **Qualidade de Software:** Suíte robusta de **testes de integração** (`pytest`) que valida a API, o carregamento de artefatos e a lógica de detecção de anomalias antes do deploy.
*   **Containerização Segura:** Dockerfile otimizado rodando com **usuário não-root** (appuser) para mitigar riscos de segurança em produção.
*   **Documentação Interativa:** O Swagger UI vem pré-configurado com exemplos de dados e endpoints auxiliares para facilitar o teste manual.

---

## 🏗️ Arquitetura e Decisões Técnicas (ADR)

| Componente | Escolha Técnica | Justificativa (Why?) |
| :--- | :--- | :--- |
| **Framework DL** | **PyTorch + Lightning** | Flexibilidade dinâmica e remoção de *boilerplate* (loops manuais), facilitando a manutenção e uso de callbacks. |
| **Tracking** | **MLflow** | Padrão de mercado para rastreabilidade de experimentos (métricas e parâmetros). |
| **API** | **FastAPI** | Alta performance (ASGI), validação automática com Pydantic e suporte nativo a processamento assíncrono. |
| **Testes** | **Pytest + TestClient** | Padrão da indústria. O TestClient permite simular requisições à API sem necessidade de subir o servidor, validando o ciclo de vida (`lifespan`) da aplicação. |
| **Drift Detection** | **Estatística (In-App)** | Implementação leve baseada em estatísticas descritivas (Baseline JSON). Evita a complexidade de ferramentas externas pesadas para este escopo. |
| **Configuração** | **Single Source of Truth** | Uso de `src/config.py` centralizado para evitar "números mágicos" e inconsistências de caminhos. |

---

## ⚡ Guia de Instalação e Execução

### Pré-requisitos
*   **Docker** (Recomendado para execução isolada e avaliação).
*   **Python 3.11+** e **Poetry** (Para desenvolvimento local).

### 1. Clonar o Repositório
O primeiro passo é obter o código-fonte em sua máquina local.

```bash
git clone https://github.com/fernando-lfs/fiap-tech-challenge-fase4.git
cd fiap-tech-challenge-fase4
```

### 2. Configuração do Ambiente
Você pode executar o projeto via **Docker** (Recomendado para avaliação rápida) ou **Localmente** (Para desenvolvimento).

#### Opção A: Via Docker (Produção/Avaliação)
Sobe a API pronta para uso, contendo o modelo pré-treinado.

```bash
# 1. Construir a Imagem
docker build -t lstm-mlops .

# 2. Rodar o Container
docker run -d -p 8000:8000 --name api-lstm lstm-mlops
```
*Acesse a documentação interativa em:* [http://localhost:8000/docs](http://localhost:8000/docs)

---

#### Opção B: Execução Local (Desenvolvimento)
Recomendado se você deseja rodar o pipeline de treinamento passo a passo.

**Passo 1: Instalar Dependências**
```bash
# Se estiver usando Poetry (Recomendado)
poetry install
poetry shell

# OU via pip tradicional
pip install -r requirements.txt
```

**Passo 2: Executar o Pipeline de Dados e Treino**
Siga a ordem lógica dos scripts para reproduzir o ciclo de vida do modelo:

```bash
# 1. Coleta (Yahoo Finance) -> Gera data/01_raw/*.csv
python -m scripts.01_coleta_dados

# 2. Pré-processamento -> Gera data/02_processed/*.npy e baseline_stats.json
# IMPORTANTE: Essencial para o funcionamento do Drift Detection
python -m scripts.02_preprocess

# 3. Treinamento -> Gera models/lstm_model.pth e registra no MLflow
python -m scripts.03_train

# 4. Avaliação -> Gera métricas e gráficos em results/
python -m scripts.04_evaluate
```

**Passo 3: Iniciar a API**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🔌 Documentação da API

Abaixo, uma visão geral de todos os endpoints disponíveis. Para detalhes de implementação (JSON Body/Response), consulte as seções detalhadas logo após a tabela.

| Método | Endpoint | Descrição |
| :--- | :--- | :--- |
| `POST` | **/predict** | **Principal:** Realiza a predição de preço (D+1) e detecta Data Drift. |
| `POST` | **/train** | **Principal:** Dispara retreino ou tuning de hiperparâmetros em background. |
| `GET` | **/model/info** | **Principal:** Exibe métricas de performance (MAE, RMSE) do modelo atual. |
| `GET` | **/sample-data** | Retorna dados reais de teste para facilitar o uso do `/predict`. |
| `GET` | **/health** | Health Check (Liveness Probe) para monitoramento (K8s/Docker). |
| `GET` | **/config** | Consulta os hiperparâmetros carregados na memória. |
| `POST` | **/config** | Atualiza hiperparâmetros na memória (sem disparar treino). |
| `POST` | **/model/reload** | Força o recarregamento dos arquivos `.pth` e `.joblib` do disco. |
| `GET` | **/** | Verifica se a API está online (Root). |

### Detalhamento dos Endpoints Principais

#### 1. Predição de Preço (`POST /predict`)
Recebe uma janela histórica e prevê o fechamento do dia seguinte (D+1).

*   **Regra de Negócio:** É obrigatório enviar **exatamente 60 preços** (dias), correspondentes à janela de treinamento da LSTM.
*   **Drift:** Se os dados fugirem do padrão estatístico do treino, `drift_warning` será `true`.

**Exemplo de Requisição (Body):**
```json
{
  "last_prices": [10.5, 10.6, ..., 11.2] // Lista com 60 floats
}
```

**Exemplo de Resposta (Sucesso):**
```json
{
  "predicted_price": 12.45,
  "drift_warning": false,
  "drift_details": []
}
```

#### 2. Treinamento e Tuning (`POST /train`)
Dispara um job de retreino em background. Permite ajuste fino de hiperparâmetros (Tuning).

**Exemplo de Requisição (Tuning):**
```json
{
  "hyperparameters": {
    "learning_rate": 0.0005,
    "num_epochs": 100,
    "hidden_size": 128
  }
}
```

#### 3. Monitoramento (`GET /model/info`)
Retorna o estado atual do modelo em produção e métricas da última avaliação.

**Exemplo de Resposta:**
```json
{
  "version": "0.1.0",
  "current_params": {
    "seq_length": 60,
    "hidden_size": 64
  },
  "metrics": {
    "mae": 0.20,
    "rmse": 0.25,
    "mape": 1.94
  }
}
```

---

## 📂 Estrutura do Projeto

```text
/
├── api/                  # Aplicação Web (FastAPI)
├── data/                 # Data Lake Local (Raw e Processed)
├── mlruns/               # Registro de Experimentos MLflow
├── models/               # Artefatos Persistidos (.pth, .joblib, .json)
├── results/              # Gráficos de Performance
├── scripts/              # Pipeline de Execução (ETL, Train, Eval)
├── src/                  # Código Fonte Compartilhado (Model, Dataset, Config)
├── tests/                # Testes de Integração
├── Dockerfile            # Definição da Imagem
├── pyproject.toml        # Gerenciamento de Dependências (Poetry)
└── README.md             # Documentação
```

---

## 📈 Resultados Obtidos

O modelo atual (LSTM 2-Layers, Hidden=64) apresentou nos dados de teste:

| Métrica | Valor | Descrição |
| :--- | :--- | :--- |
| **MAPE** | **2.24%** | Erro percentual médio absoluto. |
| **RMSE** | **0.28** | Raiz do erro quadrático médio (na escala real em R$). |
| **MAE**  | **0.23** | Erro absoluto médio (na escala real em R$). |

---

## ☁️ Proposta de Escalabilidade

Para um cenário de alta demanda, a arquitetura evoluiria para:

1.  **Kubernetes (K8s):** Orquestração dos containers com HPA (Horizontal Pod Autoscaler) baseado em uso de CPU.
2.  **Fila de Mensagens (RabbitMQ/Celery):** O endpoint `/train` deixaria de processar localmente e enviaria jobs para workers dedicados em GPUs isoladas.

---

## 👥 Autor

**Fernando Luiz Ferreira**
```