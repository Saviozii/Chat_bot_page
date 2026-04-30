# Chat Bot Page - Sistema RAG com Django

Sistema de chatbot inteligente baseado em RAG (Retrieval-Augmented Generation) para processamento e consulta de documentos PDF com suporte a imagens.

## Tecnologias

- **Backend**: Django, LangChain, Qdrant
- **Embeddings**: Ollama (local) / OpenAI GPT (produção)
- **Vector Store**: Qdrant
- **Processamento**: PyPDF, Text Splitters

## Estrutura

```
├── pipeline_01.py    # Estrutura base com classes (testes)
├── pipeline_02.py    # Pipeline local com Ollama (bge-m3)
├── pipeline_03.py    # Pipeline produção com OpenAI GPT
├── Rag_db.py         # Integração RAG com banco de dados
├── chat/             # App Django do chatbot
├── app/              # Configurações Django
└── data/             # Arquivos de dados (imagens, textos)
```

## Configuração

### 1. Clone o repositório
```bash
git clone https://github.com/Saviozii/Chat_bot_page.git
cd Chat_bot_page
```

### 2. Crie e ative o ambiente virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
pip install langchain-openai python-dotenv
```

### 4. Configure as variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
# OpenAI (para pipeline_03.py)
OPENAI_API_KEY=sua_chave_openai_aqui
OPENAI_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=pdf_docs_prod

# Configurações de chunking
CHUNK_SIZE=600
CHUNK_OVERLAP=100
BATCH_SIZE=100
```

### 5. Configure o Qdrant

**Docker:**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

Ou use o Qdrant Cloud: https://cloud.qdrant.io

## Uso

### Processar documentos (Pipeline)

**Local (Ollama):**
```bash
python pipeline_02.py
```

**Produção (OpenAI):**
```bash
python pipeline_03.py
```

### Executar o Django

```bash
python manage.py migrate
python manage.py runserver
```

Acesse: http://localhost:8000

## Pipelines

| Arquivo | Descrição | Modelo | Ambiente |
|---------|-----------|--------|----------|
| pipeline_01.py | Estrutura base com classes | bge-m3 (simulado) | Desenvolvimento |
| pipeline_02.py | Processamento local | bge-m3 (Ollama) | Desenvolvimento |
| pipeline_03.py | Processamento produção | text-embedding-3-small | Produção |

## Funcionalidades

- ✅ Processamento de PDFs com extração de páginas
- ✅ Processamento de descrições de imagens
- ✅ Chunking inteligente com sobreposição
- ✅ Armazenamento vetorial no Qdrant
- ✅ Suporte a múltiplos modelos de embedding
- ✅ Interface web com Django
- ✅ Modo produção com batches e logs

## Contribuição

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

## Contato

- GitHub: [@Saviozii](https://github.com/Saviozii)
- Email: ramonsantosfernandes2016@gmail.com
