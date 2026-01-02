# Guia de Contribuição

Obrigado pelo interesse em contribuir para o projeto **Tech Challenge - LSTM Forecast**. Este documento define as diretrizes para garantir a qualidade, consistência e reprodutibilidade do código.

## 1. Código de Conduta

Este projeto adota um ambiente de respeito e colaboração. Críticas construtivas são bem-vindas; desrespeito ou discriminação não serão tolerados.

## 2. Fluxo de Desenvolvimento (Git Flow)

Para manter a integridade da *branch* principal (`main`), siga este fluxo:

1.  **Fork** o repositório.
2.  Crie uma **Feature Branch** para sua alteração:
    ```bash
    git checkout -b feature/nova-funcionalidade
    ```
3.  Implemente suas mudanças.
4.  Realize o **Commit** seguindo as convenções (veja abaixo).
5.  Abra um **Pull Request (PR)** para a branch `main`.

## 3. Padrões de Código e Estilo

* **Linguagem:** Python 3.11+.
* **Formatação:** O código deve seguir a **PEP 8**. Recomendamos o uso de formatadores como `black` ou `isort`.
* **Tipagem:** Utilize *Type Hints* nas assinaturas de funções sempre que possível.
    * *Bom:* `def train(epochs: int) -> float:`
    * *Evitar:* `def train(epochs):`
* **Logging:** Não utilize `print()`. Utilize o objeto `logger` configurado em `scripts/__init__.py` ou `api/__init__.py`.

## 4. MLOps e Experimentos

Se sua contribuição envolve alterações no modelo ou nos dados:

* **MLflow:** Certifique-se de que novos parâmetros sejam logados via `mlflow.log_params()`.
* **Reprodutibilidade:** Se alterar o pré-processamento, atualize a função `save_baseline_stats` em `scripts/02_preprocess.py`.

## 5. Mensagens de Commit

Recomendamos o padrão **Conventional Commits**:

* `feat:` Nova funcionalidade (ex: `feat: adiciona endpoint de reload`).
* `fix:` Correção de bug (ex: `fix: erro no cálculo do MAPE`).
* `docs:` Alterações na documentação.
* `refactor:` Refatoração de código sem mudança de funcionalidade.

## 6. Testes Antes do PR

Antes de submeter, execute o pipeline localmente para garantir que nada quebrou:

```bash
# 1. Garanta que o ETL roda
python -m scripts.02_preprocess

# 2. Garanta que o treino finaliza e salva o .pth
python -m scripts.03_train

# 3. Verifique se a API sobe
uvicorn api.main:app --reload