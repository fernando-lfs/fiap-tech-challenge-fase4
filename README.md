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
│   ├── 04_evaluate.py    # Avaliação em dados de teste
|   └── __init__.py       # Configuração de Logging
├── src/                  # Código Fonte Reutilizável
│   ├── dataset.py
│   ├── model.py
|   └── __init__.py
├── .dockerignore
├── .gitignore
├── CONTRIBUTING.md
├── Dockerfile
├── mlflow.db
├── poetry.lock
├── pyproject.toml
├── README.md
└── requiriments.txt
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

   ⚠️ **Atenção:** É obrigatório executar o script `02_preprocess.py` antes de iniciar a API ou o treinamento. Este script gera o arquivo `baseline_stats.json`, essencial para que o detector de Drift funcione corretamente. Caso ele não exista, o monitoramento de qualidade da API será desativado.

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
  * **Importante:** A lista deve conter **exatamente 60 valores** (correspondente ao `seq_length` configurado), representando os últimos 60 dias de fechamento para a previsão do dia seguinte.
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

O endpoint /config permite o tuning dinâmico de hiperparâmetros. Ao atualizar a configuração e disparar o /train, o sistema realiza o ajuste fino (fine-tuning) do modelo sem necessidade de alterar o código-fonte.

### 4. Recarregar Modelo (`POST /model/reload`)

Atualiza o modelo em memória (Hot Reload) após um retreino, sem reiniciar o servidor.

### 5. Saúde (`GET /health`)

Monitora CPU, Memória e disponibilidade dos artefatos.

---

## 📚 Glossário Técnico

* **LSTM (Long Short-Term Memory):** Tipo de rede neural recorrente capaz de aprender dependências de longo prazo, ideal para séries temporais (como preços de ações).

* **Data Drift:** Ocorre quando as propriedades estatísticas dos dados de entrada mudam de forma significativa em relação aos dados usados no treino. No mercado financeiro, isso pode ser causado por crises econômicas ou mudanças bruscas na volatilidade, o que pode invalidar as predições do modelo.

* **MAPE (Mean Absolute Percentage Error):** Métrica que indica o erro médio em porcentagem. Um MAPE de 1.56% significa que, em média, a previsão erra apenas 1.56% do valor real da ação.

---

## ☁️ Escalabilidade e Monitoramento (Proposta)

Para garantir a elasticidade da solução em ambiente produtivo de alta escala, propõe-se a seguinte arquitetura baseada em microsserviços e orquestração:

1. **Orquestração e Auto-scaling:**
* **Horizontal Pod Autoscaler (HPA) no Kubernetes:** Configuração de um HPA para monitorar métricas de **CPU** e **Latência de Requisição**.
* **Regra de Escala:** Caso a utilização de CPU ultrapasse 70% ou a latência média exceda um limite definido, o Kubernetes iniciará novas réplicas (Pods) da API automaticamente para suportar a carga.
2. **Arquitetura de Deploy e Balanceamento:**
* **Ingress Controller (Nginx):** Atua como o ponto de entrada único e **Load Balancer**, distribuindo o tráfego de forma inteligente entre os diversos Pods ativos da API, garantindo alta disponibilidade.
* **Serviço de Treinamento Dedicado:** O endpoint `/train` deve ser desacoplado para um **Worker** assíncrono especializado.
3. **Desacoplamento de Processos Pesados:**
* **Fila de Mensagens (Redis/RabbitMQ):** Em produção, a requisição de treinamento não é executada pela API de inferência, mas enviada para uma fila.
* **Workers Assíncronos (Celery):** Instâncias dedicadas consomem essa fila para processar o treinamento de forma isolada, evitando que o consumo intensivo de recursos (CPU/GPU) do treino prejudique a performance e a latência das previsões para o usuário final.
4. **Monitoramento de Qualidade (Observabilidade):**
* **Detecção de Drift:** O mecanismo de monitoramento implementado gera logs estruturados (`WARNING`) sempre que uma anomalia estatística é detectada.
* **Dashboards de Qualidade:** Utilização de ferramentas como **Fluentd** ou **Filebeat** para coletar esses logs e enviá-los para um stack de visualização (**Grafana/Kibana**), permitindo alertas em tempo real sobre a degradação da precisão do modelo.

---

## 👥 Autores

* Fernando LFS — [GitHub](https://github.com/fernando-lfs) | [LinkedIn](https://www.linkedin.com/in/fernando-lfs/)

---
