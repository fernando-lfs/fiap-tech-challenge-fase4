# 📈 Tech Challenge - Fase 4: Previsão de Ações com MLOps

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c?style=for-the-badge&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi)
![Pytest](https://img.shields.io/badge/Pytest-Testing-yellow?style=for-the-badge&logo=pytest)
![Docker](https://img.shields.io/badge/Docker-Container-2496ed?style=for-the-badge&logo=docker)

> **Pós-Graduação em Deep Learning & AI - FIAP**

Este projeto apresenta uma solução completa de **End-to-End Machine Learning** para a previsão de preços de fechamento de ações da **CEMIG (CMIG4.SA)**.

A arquitetura abrange desde a engenharia de dados até o deploy produtivo, utilizando **LSTM (Long Short-Term Memory)** para modelagem temporal, **MLflow** para rastreamento de experimentos e **FastAPI** para servir o modelo, tudo orquestrado via **Docker** e validado com testes de integração.

---

## 🚀 Funcionalidades e Diferenciais

*   **Pipeline Automatizado:** Scripts modulares para ETL (Extração e Transformação), Treinamento e Avaliação.
*   **Deep Learning Moderno:** Uso de **PyTorch Lightning** para estruturar o código de treino, garantindo legibilidade e reprodutibilidade (seeds fixas).
*   **Qualidade de Software:** Suíte de **testes de integração** (`pytest`) que valida a API, o carregamento de artefatos e a lógica de detecção de anomalias antes do deploy.
*   **MLOps & Tracking:** Integração nativa com **MLflow** para registrar métricas (MAE, RMSE, MAPE), hiperparâmetros e artefatos do modelo.
*   **API Inteligente & Usabilidade:**
    *   **Detecção de Data Drift:** O endpoint de predição monitora estatisticamente a entrada. Se os dados desviarem do padrão de treino (ex: alta volatilidade), um alerta é retornado.
    *   **Treino Assíncrono:** O endpoint `/train` utiliza `BackgroundTasks`, permitindo que o modelo seja retreinado sem bloquear as inferências.
    *   **Documentação Interativa:** O Swagger UI vem pré-configurado com exemplos de dados e endpoints auxiliares para facilitar o teste manual.
*   **Containerização Segura:** Dockerfile otimizado (multi-stage concepts), rodando com usuário não-root para segurança.

---

## 🏗️ Arquitetura e Decisões Técnicas (ADR)

| Componente | Escolha Técnica | Justificativa (Why?) |
| :--- | :--- | :--- |
| **Framework DL** | **PyTorch + Lightning** | O PyTorch oferece flexibilidade dinâmica. O Lightning foi adotado para remover *boilerplate* (loops manuais), padronizar o código e facilitar o uso de *callbacks* (Early Stopping). |
| **Tracking** | **MLflow** | Ferramenta open-source padrão de mercado, agnóstica de infraestrutura, permitindo rastreabilidade total dos experimentos. |
| **API** | **FastAPI** | Alta performance (ASGI), validação automática de dados com Pydantic e suporte nativo a processamento assíncrono. |
| **Testes** | **Pytest + TestClient** | Padrão da indústria para testes em Python. O TestClient permite simular requisições à API sem necessidade de subir o servidor, validando o ciclo de vida (`lifespan`) da aplicação. |
| **Drift Detection** | **Estatística (In-App)** | Implementação de um detector leve baseado em estatísticas descritivas (Baseline JSON). Evita a complexidade de ferramentas externas pesadas para este escopo, garantindo monitoramento em tempo real. |
| **Configuração** | **Single Source of Truth** | Uso de um arquivo `src/config.py` centralizado para evitar "números mágicos" e inconsistências de caminhos entre scripts e API. |

---

## 📂 Estrutura do Projeto

```text
/
├── api/                  # Aplicação Web (FastAPI)
│   ├── main.py           # Endpoints e Lógica de Negócio
│   └── __init__.py       # Configuração de Logs
├── data/                 # Data Lake Local
│   ├── 01_raw/           # Dados brutos (CSV)
│   └── 02_processed/     # Dados normalizados (.npy)
├── mlruns/               # Registro de Experimentos MLflow
├── models/               # Artefatos Persistidos
│   ├── lstm_model.pth    # Pesos do Modelo (State Dict)
│   ├── scaler.joblib     # Normalizador (MinMaxScaler)
│   ├── baseline_stats.json # Estatísticas para Drift Detection
│   └── metrics.json      # Métricas do último treino (para API)
├── results/              # Gráficos de Performance
├── scripts/              # Pipeline de Execução
│   ├── 01_coleta_dados.py
│   ├── 02_preprocess.py
│   ├── 03_train.py
│   └── 04_evaluate.py
├── src/                  # Código Fonte Compartilhado
│   ├── config.py         # Configurações Globais
│   ├── dataset.py        # Classe Dataset (PyTorch)
│   └── model.py          # Arquitetura LSTM
├── tests/                # Testes Automatizados
│   └── test_integration.py
├── Dockerfile            # Definição da Imagem
├── pyproject.toml        # Gerenciamento de Dependências (Poetry)
└── README.md             # Documentação
```

---

## ⚡ Como Executar

### Pré-requisitos
*   Docker (para execução isolada)
*   Python 3.11+ e Poetry (para desenvolvimento local)

### Opção 1: Via Docker (Recomendado para Produção)

Esta opção sobe a API pronta para uso, contendo o modelo pré-treinado.

1.  **Gerar requirements (caso tenha alterado dependências):**
    ```bash
    poetry export -f requirements.txt --output requirements.txt --without-hashes
    ```

2.  **Construir a Imagem:**
    ```bash
    docker build -t lstm-mlops .
    ```

3.  **Rodar o Container:**
    ```bash
    docker run -d -p 8000:8000 --name api-lstm lstm-mlops
    ```

4.  **Acessar:**
    *   Documentação Interativa (Swagger): [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Opção 2: Execução Local (Desenvolvimento)

Siga esta ordem para reproduzir todo o ciclo de vida do modelo.

1.  **Instalação:**
    ```bash
    poetry install
    poetry shell
    ```

2.  **Pipeline de Dados e Treino:**
    ```bash
    # 1. Coleta (Yahoo Finance)
    python -m scripts.01_coleta_dados

    # 2. Pré-processamento (Gera dados .npy e baseline_stats.json)
    # IMPORTANTE: Essencial para o funcionamento do Drift Detection
    python -m scripts.02_preprocess

    # 3. Treinamento (Gera lstm_model.pth e registra no MLflow)
    python -m scripts.03_train

    # 4. Avaliação (Gera métricas e gráficos em /results)
    python -m scripts.04_evaluate
    ```

3.  **Validação (Testes Automatizados):**
    Execute a suíte de testes para garantir que a API e o modelo estão integrados corretamente.
    ```bash
    pytest -v
    ```

4.  **Visualizar Experimentos:**
    ```bash
    mlflow ui
    # Acesse http://127.0.0.1:5000
    ```

5.  **Iniciar API:**
    ```bash
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    ```

---

## 🔌 Documentação da API

A API expõe endpoints estratégicos documentados via Swagger UI.

### 1. `POST /predict` (Inferência)
Recebe uma janela histórica e prevê o próximo dia.
*   **Facilidade:** O Swagger já vem preenchido com um exemplo válido.
*   **Input:** Lista com **60 preços** (float).
*   **Output:** Preço previsto + Alerta de Drift.

### 2. `GET /sample-data` (Auxiliar)
Retorna os últimos 60 dias **reais** do dataset de teste.
*   **Uso:** Copie o retorno deste endpoint e cole no `/predict` para validar o modelo com dados reais.

### 3. `POST /train` (Treino & Tuning)
Dispara um job de treinamento em **background**.
*   **Tuning:** Permite enviar novos hiperparâmetros (ex: `learning_rate`, `hidden_size`) no corpo da requisição para ajustar o modelo.

### 4. `GET /model/info` (Monitoramento)
Exibe o estado atual do modelo em produção.
*   **Retorno:** Versão, hiperparâmetros ativos e **métricas de performance** (MAE, RMSE) do último treino realizado.

### 5. `GET /health`
Monitoramento de saúde (Liveness Probe) e uso de recursos (CPU/RAM).

---

## 📈 Resultados Obtidos

O modelo atual (LSTM 2-Layers, Hidden=64) apresentou nos dados de teste:

| Métrica | Valor | Descrição |
| :--- | :--- | :--- |
| **MAPE** | **1.94%** | Erro percentual médio absoluto. |
| **RMSE** | **0.25** | Raiz do erro quadrático médio (na escala real em R$). |
| **MAE**  | **0.20** | Erro absoluto médio (na escala real em R$). |

---

## ☁️ Proposta de Escalabilidade

Para um cenário de alta demanda, a arquitetura evoluiria para:

1.  **Kubernetes (K8s):** Orquestração dos containers.
2.  **HPA (Horizontal Pod Autoscaler):** Escalonamento automático de Pods da API baseado em CPU (>70%) ou métricas customizadas de latência.
3.  **Separação de Workloads:**
    *   O endpoint `/train` deixaria de processar localmente e enviaria mensagens para uma fila (**RabbitMQ**).
    *   **Workers dedicados (Celery)** consumiriam a fila para treinar modelos em GPUs isoladas, sem impactar a latência da API de inferência.

---

## 👥 Autor

**Fernando Luiz Ferreira**
```