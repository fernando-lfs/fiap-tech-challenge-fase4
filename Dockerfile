# 1. Imagem Base
# Utilizado python:3.11-slim para manter o container leve e rápido.
FROM python:3.11-slim

# 2. Metadados
LABEL maintainer="Fernando Luiz Ferreira"
LABEL version="0.1.0"

# 3. Configurações de Ambiente
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Definição do Diretório de Trabalho
WORKDIR /app

# 5. Instalação de Dependências
COPY requirements.txt .
# Dependências instaladas sem cache para reduzir o tamanho da imagem.
RUN pip install --no-cache-dir -r requirements.txt

# 6. Cópia da Aplicação e Scripts
# Copiadas apenas pastas necessárias para a execução da API.
COPY api/ ./api/
COPY src/ ./src/
COPY scripts/ ./scripts/

# 7. Cópia de Artefatos e Dados
# Copiados modelos já treinados
COPY models/ ./models/
# Copiados dados processados para permitir o funcionamento do endpoint /train
COPY data/02_processed/ ./data/02_processed/

# 8. Diretório para Logs do MLflow
RUN mkdir -p mlruns

# 9. Exposição da porta e Inicialização do servidor Uvicorn
# Container escutará na porta 8000 (padrão FastAPI/Uvicorn).
EXPOSE 8000
# --host 0.0.0.0: Essencial para acessar o container externamente.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]