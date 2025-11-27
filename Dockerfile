# 1. Imagem Base
# Utilizamos python:3.11-slim para manter o container leve e rápido.
# Compatível com a versão 3.11.9 usada no projeto.
FROM python:3.11-slim

# 2. Definição do Diretório de Trabalho
# Todas as ações subsequentes ocorrerão dentro de /app no container.
WORKDIR /app

# 3. Instalação de Dependências
# Copiamos apenas o requirements.txt primeiro para aproveitar o cache do Docker.
COPY requirements.txt .

# Instalamos as dependências sem cache para reduzir o tamanho da imagem.
# --no-cache-dir: Evita salvar arquivos de cache do pip.
RUN pip install --no-cache-dir -r requirements.txt

# 4. Cópia do Código Fonte e Artefatos
# Copiamos apenas as pastas necessárias para a execução da API.
# A estrutura de pastas é mantida para garantir que os imports funcionem.
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

# 5. Exposição da Porta
# Informamos que o container escutará na porta 8000 (padrão FastAPI/Uvicorn).
EXPOSE 8000

# 6. Comando de Inicialização
# Iniciamos o servidor Uvicorn.
# --host 0.0.0.0: Essencial para acessar o container externamente.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]