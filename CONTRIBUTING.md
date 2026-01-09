# Guia de Contribuição

Obrigado pelo interesse em contribuir para o projeto **Tech Challenge - LSTM Forecast**. Este documento define as diretrizes técnicas para garantir a qualidade, consistência e reprodutibilidade do código em um ambiente de MLOps.

## 1. Código de Conduta

Este projeto adota um ambiente de respeito e colaboração técnica. Críticas construtivas são bem-vindas; desrespeito ou discriminação não serão tolerados.

## 2. Fluxo de Desenvolvimento (Git Flow)

Para manter a integridade da *branch* principal (`main`), siga este fluxo:

1.  **Fork** o repositório.
2.  Crie uma **Feature Branch** descritiva:
    ```bash
    git checkout -b feature/nova-arquitetura-lstm
    # ou
    git checkout -b fix/correcao-data-drift
    ```
3.  Implemente suas mudanças.
4.  Realize o **Commit** seguindo as convenções (veja seção 6).
5.  Abra um **Pull Request (PR)** para a branch `main`.

## 3. Padrões de Engenharia de Software

*   **Linguagem:** Python 3.11+.
*   **Formatação:** O código deve seguir a **PEP 8**.
*   **Tipagem (Type Hinting):** É **obrigatório** o uso de tipagem estática nas assinaturas de funções e métodos. Isso facilita a leitura e o uso de ferramentas de análise estática.
    *   *Correto:* `def train(epochs: int, lr: float) -> Dict[str, float]:`
    *   *Incorreto:* `def train(epochs, lr):`
*   **Logging:** **Proibido o uso de `print()`**. Utilize sempre o objeto `logger` importado de `scripts` ou `api`.
*   **Configuração:** Não utilize "números mágicos" ou caminhos *hardcoded*. Todas as constantes (paths, hiperparâmetros padrão, datas) devem estar centralizadas em `src/config.py`.

## 4. Gerenciamento de Dependências

Este projeto utiliza **Poetry** como fonte da verdade.

1.  Para adicionar uma lib: `poetry add <nome-da-lib>`.
2.  **Nunca edite o `requirements.txt` manualmente.** Ele é um artefato gerado para o Docker.
3.  Se você alterou as dependências, **você deve atualizar o arquivo de requisitos**:
    ```bash
    poetry export -f requirements.txt --output requirements.txt --without-hashes
    ```

## 5. Diretrizes de MLOps e Data Science

Se sua contribuição envolve alterações no modelo ou nos dados:

*   **MLflow:** Novos parâmetros ou métricas devem ser registrados explicitamente (`mlflow.log_param`, `mlflow.log_metric`).
*   **PyTorch Lightning:** Mantenha a lógica de treino (`training_step`) desacoplada da arquitetura da rede (`nn.Module`).
*   **Reprodutibilidade:**
    *   Respeite as sementes aleatórias (`RANDOM_SEED` no config).
    *   Se alterar a lógica de pré-processamento, verifique se a função `save_baseline_stats` (em `scripts/02_preprocess.py`) continua gerando estatísticas válidas para o detector de Drift.

## 6. Mensagens de Commit (Conventional Commits)

Siga o padrão: `<tipo>: <descrição breve no imperativo>`

*   `feat:` Nova funcionalidade (ex: `feat: adiciona endpoint de healthcheck`).
*   `fix:` Correção de bug (ex: `fix: corrige cálculo do RMSE`).
*   `docs:` Alterações na documentação.
*   `refactor:` Melhoria de código sem mudança de comportamento.
*   `chore:` Configurações, dependências ou CI/CD.

## 7. Checklist de Validação (Antes do PR)

Garanta que o pipeline completo funciona localmente:

1.  **ETL e Drift:** O script gera os arquivos `.npy` e o `baseline_stats.json`?
    ```bash
    python -m scripts.02_preprocess
    ```
2.  **Treino:** O modelo treina e salva o `.pth`?
    ```bash
    python -m scripts.03_train
    ```
3.  **Testes:** A suíte de testes passa sem erros?
    ```bash
    pytest -v
    ```
4.  **Docker:** A imagem constrói sem erros?
    ```bash
    docker build -t teste-pr .
    ```