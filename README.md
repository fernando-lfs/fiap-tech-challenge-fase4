# 📈 Tech Challenge - Fase 4: Previsão de Ações com MLOps

> **Deep Learning & AI - FIAP**

Este projeto consiste em uma solução completa de **End-to-End Machine Learning** para prever o preço de fechamento de ações da **CEMIG (CMIG4.SA)**. 

A solução abrange desde a coleta e pré-processamento de dados financeiros até o treinamento de uma rede neural **LSTM** (Long Short-Term Memory) utilizando **PyTorch Lightning** e **MLflow** , disponibilizando o modelo final através de uma API **FastAPI** robusta e containerizada com **Docker**. O sistema conta ainda com suporte a treinamento assíncrono e monitoramento em tempo real de **Data Drift** para garantir a confiabilidade das previsões em produção.

---

## 🚀 Funcionalidades Principais

* **Coleta & Baseline:** Download automático via `yfinance` e geração de estatísticas descritivas para detecção de anomalias.
* **Treinamento Padronizado:** Pipeline utilizando `PyTorch Lightning` para organizar loops de treino/validação e `EarlyStopping`.
* **Rastreamento (Tracking):** Registro automático de hiperparâmetros, métricas (Loss, MAPE) e artefatos (modelos `.pth`, gráficos) via **MLflow**.
* **API Gerenciável:** Interface RESTful que permite não apenas prever, mas também disparar **retreinos em background** e atualizar configurações dinamicamente.
* **Observabilidade de Dados:** O endpoint de predição detecta automaticamente **Data Drift** (mudanças bruscas de padrão ou volatilidade) comparando a entrada com o baseline de treino.
* **Escalabilidade:** Arquitetura desenhada para execução em containers e orquestração.

---

## 🛠️ Stack Tecnológico

* **Linguagem:** Python 3.11
* **Gerenciamento:** Poetry
* **Deep Learning:** PyTorch, PyTorch Lightning
* **MLOps:** MLflow
* **API:** FastAPI, Uvicorn, Pydantic
* **Dados:** Pandas, Numpy, Scikit-Learn, Yahoo Finance
* **Infraestrutura:** Docker

---

## 🏗️ Arquitetura e Decisões Técnicas (ADR)

Para atender aos requisitos de qualidade de engenharia, as seguintes decisões foram tomadas:

1. **PyTorch Lightning:** Adotado para remover *boilerplate code* (loops manuais) e padronizar o código de treinamento, facilitando a manutenção e a reprodutibilidade.
2. **MLflow:** Escolhido como ferramenta de *Tracking* por ser agnóstico à infraestrutura (roda localmente ou na nuvem) e permitir versionamento claro de cada experimento.
3. **FastAPI com BackgroundTasks:** Para o endpoint de treinamento (`/train`), utilizamos processamento assíncrono. Isso impede que uma requisição de treino bloqueie a API, mantendo-a responsiva para inferências simultâneas.
4. **Detecção de Drift "In-App":** Optou-se por implementar um detector estatístico leve dentro da própria API (comparação com Baseline JSON). Isso garante monitoramento de qualidade imediato sem a complexidade/custo de ferramentas externas pesadas (como Evidently AI) para este escopo acadêmico.

---

## 📂 Estrutura do Projeto

```text
/
├── api/                  # Aplicação Web e Logs Centralizados
│   ├── main.py           # Endpoints (Train, Predict, Config)
│   └── __init__.py       # Configuração de Logging
├── data/                 # Data Lake (Raw e Processed)
├── mlruns/               # Registro local do MLflow (Metadados dos experimentos)
├── models/               # Artefatos: .pth, .joblib e baseline_stats.json
├── results/              # Gráficos gerados
├── scripts/              # Pipelines ETL e Treino
│   ├── 01_coleta_dados.py
│   ├── 02_preprocess.py  # Gera dados normalizados e Baseline de Drift
│   ├── 03_train.py       # Treino com Lightning + MLflow
│   └── 04_evaluate.py    # Avaliação em dados de teste
├── src/                  # Código Fonte Reutilizável
│   ├── dataset.py
│   └── model.py
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## 📈 Performance e Resultados

O modelo final (LSTM com 2 camadas, 64 neurônios) atingiu os seguintes resultados nos dados de teste:

| Métrica                    | Valor     |
| -------------------------- | --------- |
| **MAPE** (Erro Percentual) | **1.56%** |
| **MAE** (Erro Absoluto)    | R$ 0.16   |
| **RMSE** (Erro Quadrático) | R$ 0.19   |

> **Nota:** Todos os gráficos de perda e métricas detalhadas podem ser visualizados via `mlflow ui`.

---

## ⚡ Como Executar o Projeto

### Opção 1: Via Docker (Produção)

1. **Construir a imagem:**
   
   ```bash
   docker build -t lstm-mlops .
   ```

2. **Rodar o container:**
   
   ```bash
   docker run -d -p 8000:8000 --name api-lstm lstm-mlops
   ```

3. **Acessar:**
* Swagger UI: `http://localhost:8000/docs`

---

### Opção 2: Execução Local (Desenvolvimento & Experimentos)

1. **Instalar dependências:**
   
   ```bash
   poetry install
   poetry shell
   ```

2. **Executar Pipeline Completo (ETL + Treino):**
   
   ```bash
   # 1. Coleta e Preprocessamento (Gera baseline_stats.json)
   python -m scripts.01_coleta_dados
   python -m scripts.02_preprocess
   # 2. Treinamento (Registra no MLflow)
   python -m scripts.03_train
   # 3. Avaliação
   python -m scripts.04_evaluate
   ```

3. **Visualizar Experimentos (MLflow):**
   
   ```bash
   mlflow ui
   # Acesse [http://127.0.0.1:5000](http://127.0.0.1:5000) para ver gráficos e parâmetros
   ```

4. **Subir a API:**
   
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## 🔌 Documentação da API

A API possui 5 endpoints principais para ciclo de vida completo do modelo.

### 1. Inferência com Monitoramento (`POST /predict`)

Realiza a previsão e verifica se há **Data Drift**.

* **Input:** Lista de preços (float).
* **Output:** Preço previsto e alerta de drift.

```json
// Resposta Exemplo
{
  "predicted_price": 12.85,
  "drift_warning": true,
  "drift_details": ["Alta volatilidade detectada (3x superior ao treino)."]
}
```

### 2. Treinamento (`POST /train`)

Dispara um novo treinamento em **background** (sem travar a API).

* **Input (Opcional):** Hiperparâmetros para sobrescrever o padrão.

```json
{
  "hyperparameters": {
    "num_epochs": 10,
    "learning_rate": 0.005
  }
}
```

### 3. Configuração (`GET/POST /config`)

Lê ou atualiza os hiperparâmetros globais usados nos próximos treinos.

### 4. Recarregar Modelo (`POST /model/reload`)

Atualiza o modelo em memória (Hot Reload) após um retreino, sem reiniciar o servidor.

### 5. Saúde (`GET /health`)

Monitora CPU, Memória e disponibilidade dos artefatos.

---

## ☁️ Escalabilidade e Monitoramento (Proposta)

Para garantir a elasticidade da solução em ambiente produtivo de alta demanda, propõe-se a seguinte arquitetura:

1. **Horizontal Pod Autoscaler (HPA) no Kubernetes:**
* Configuração de um HPA monitorando a métrica de **CPU** e **Latência**.
* **Regra:** Se a utilização de CPU ultrapassar 70%, o Kubernetes inicia novas réplicas (Pods) da API automaticamente.
2. **Desacoplamento de Treino:**
* Em produção, o endpoint `/train` enviaria uma mensagem para uma fila (Redis/RabbitMQ).
* Workers dedicados (Celery) consumiriam essa fila para treinar o modelo, evitando impacto na performance da inferência.
3. **Monitoramento de Qualidade:**
* O mecanismo de *Drift* atual gera logs estruturados (`WARNING`).
* Ferramentas como **Fluentd** ou **Filebeat** coletariam esses logs para gerar alertas em dashboards (Grafana/Kibana) quando a taxa de drift excedesse um limiar seguro.

---

## 👥 Autores

* Fernando LFS — [GitHub](https://github.com/fernando-lfs) | [LinkedIn](https://www.linkedin.com/in/fernando-lfs/)

---
