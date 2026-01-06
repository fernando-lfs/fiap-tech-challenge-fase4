# 1. Imagem Base
# Python 3.11 Slim: Equilíbrio ideal entre tamanho e compatibilidade
FROM python:3.11-slim

# 2. Metadados
LABEL maintainer="Fernando Luiz Ferreira"
LABEL description="API de Previsão de Ações (LSTM) - FIAP Tech Challenge"
LABEL version="0.1.0"

# 3. Configurações de Ambiente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Garante que o pip não reclame de rodar como root (durante o build)
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# 4. Instalação de Dependências de Sistema
# curl: necessário para o HEALTHCHECK
# build-essential: necessário para compilar certas libs python (ex: psutil)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Configuração de Diretório e Usuário (Segurança)
# Cria um usuário não-root para rodar a aplicação
RUN useradd -m -u 1000 appuser
WORKDIR /app

# 6. Instalação de Dependências Python
# Copiamos apenas o requirements.txt primeiro para aproveitar o cache de camadas do Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. Cópia do Código Fonte
# Copiamos o restante da aplicação
COPY api/ ./api/
COPY src/ ./src/
COPY scripts/ ./scripts/

# 8. Cópia de Artefatos (Modelos e Dados Processados)
# Necessário para a API funcionar sem precisar treinar do zero
COPY models/ ./models/
# Necessário para o endpoint /train funcionar
COPY data/02_processed/ ./data/02_processed/

# 9. Ajuste de Permissões
# Garante que o usuário appuser tenha acesso aos arquivos
RUN chown -R appuser:appuser /app

# 10. Troca para Usuário Não-Root
USER appuser

# 11. Healthcheck (Engenharia de Qualidade)
# O Docker verificará se a API está respondendo a cada 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# 12. Inicialização
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]