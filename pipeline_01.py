import os
from typing import List

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# ==============================================================================
# 1. CONFIGURAÇÃO (Centraliza todas as variáveis globais)
# ==============================================================================

class Config:
    """Armazena todas as configurações necessárias para o pipeline."""
    
    # Caminhos de Arquivos
    PDF_PATH = "/home/savi021/Downloads/Ebook.pdf"
    TXT_IMAGENS = "/home/savi021/Documentos/Novo_mundo/data/Infor.txt"
    
    # Configurações do Vetor Store
    QDRANT_URL = "http://localhost:6333"
    COLLECTION = "pdf_docs"
    
    # Configurações de Embedding
    EMBEDDING_MODEL = "bge-m3"
    EMBEDDING_BASE_URL = "http://localhost:11434"

    # Configurações de Splitting
    CHUNK_SIZE = 600
    CHUNK_OVERLAP = 100
    VECTOR_SIZE = 1024


# ==============================================================================
# 2. MÓDULO DE CARREGAMENTO DE DADOS (Data Loader)
# ==============================================================================

class DataLoader:
    #Pega e carrega documentos de diferentes fontes (PDF e TXT).


    @staticmethod
    def load_pdf(pdf_path: str) -> List[Document]:
        """Carrega documentos de um arquivo PDF."""
        print(f"\n[LOADER]  Lendo PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Marca a fonte para rastreamento
        for doc in documents:
            doc.metadata["source"] = "pdf"

        print(f"[LOADER]  ✅ {len(documents)} páginas carregadas do PDF.")
        return documents

    @staticmethod
    def load_txt_images(txt_path: str) -> List[Document]:
        """
        Carrega descrições de imagens de um arquivo TXT.
        Extrai o número da página do cabeçalho.
        """
        if not os.path.exists(txt_path):
            print(f"[LOADER] ⚠️  Arquivo de imagens não encontrado — indexando só o PDF.")
            print(f"[LOADER]     Esperado em: {txt_path}")
            return []

        print(f"[LOADER]  Lendo Imagens TXT: {txt_path}")
        documents: List[Document] = []

        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                conteudo = f.read()
        except Exception as e:
            print(f"[LOADER] ❌ Erro ao abrir o arquivo TXT: {e}")
            return []
            
        # Divide o arquivo nos separadores "---"
        blocos = [b.strip() for b in conteudo.split("---") if b.strip()]

        for bloco in blocos:
            page_num = "?"
            linhas = bloco.splitlines()
            
            # Lógica de extração de página
            for linha in linhas:
                if linha.startswith("[IMAGEM") and "Página" in linha:
                    try:
                        # Assume que o formato é consistente
                        page_num = int(linha.split("Página")[-1].replace("]", "").strip())
                    except ValueError:
                        pass # Mantém o ? se falhar
                    break

            # Cria um Document com a descrição e metadados de origem
            doc = Document(
                page_content=bloco,
                metadata={
                    "page":   page_num,
                    "source": "imagem",
                }
            )
            documents.append(doc)

        print(f"[LOADER]  ✅ {len(documents)} descrições de imagens carregadas.")
        return documents

# ==============================================================================
# 3. MÓDULO DE PRÉ-PROCESSAMENTO (Text Splitter)
# ==============================================================================

class TextSplitter:
    """
    Responsável por dividir documentos grandes em chunks menores.
    """
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        print(f"\n[SPLITTER] Configurado para Chunk Size: {chunk_size}, Overlap: {chunk_overlap}")

    def split(self, documents: List[Document]) -> List[Document]:
        """Divide a lista de documentos em chunks."""
        chunks = self.splitter.split_documents(documents)

        # Relatório de contagem
        chunks_pdf    = [c for c in chunks if c.metadata.get("source") == "pdf"]
        chunks_imagem = [c for c in chunks if c.metadata.get("source") == "imagem"]
        
        print("-" * 50)
        print(f"[SPLITTER] 📄 {len(chunks_pdf)} chunks do PDF")
        print(f"[SPLITTER] 🖼️ {len(chunks_imagem)} chunks de imagens")
        print(f"[SPLITTER] TOTAL DE CHUNKS GERADOS: {len(chunks)}")
        print("-" * 50)
        
        return chunks


class KnowledgeBasePipeline:
    """
    Orquestra todo o fluxo: Carrega dados -> Divide em chunks -> Armazena vetores.
    """
    def __init__(self, config: dict):
        self.config = config
        self.embedding_model = None
        self.vector_store = None
        
    def _initialize_embedding(self):
        """Inicializa o modelo de embedding (simulado aqui)."""
        print("[INFO] Inicializando modelo de embedding...")
        # Em um ambiente real, você inicializaria seu cliente OpenAI/HuggingFace aqui.
        self.embedding_model = "SimulatedEmbedder" 
        print(f"[SUCCESS] Modelo de embedding pronto: {self.embedding_model}")

    def _setup_vector_store(self):
        """Configura o armazenamento vetorial (simulado aqui)."""
        print("[INFO] Configurando armazenamento vetorial...")
        # Em um ambiente real, você conectaria a Chroma, Pinecone, Weaviate, etc.
        self.vector_store = "SimulatedVectorStore"
        print(f"[SUCCESS] Vetor store configurado: {self.vector_store}")

    def run_pipeline(self):
        """Executa o pipeline completo de construção da base de conhecimento."""
        print("\n===================================================")
        print("🚀 INICIANDO PIPELINE DE CONSTRUÇÃO DA BASE DE CONHECIMENTO 🚀")
        print("===================================================\n")
        
        # 1. Inicialização
        self._initialize_embedding()
        self._setup_vector_store()

        # 2. Carregamento de Dados
        print("\n--- 🧩 ETAPA 1: CARREGAMENTO DE DADOS ---")
        # Junta o carregamento de ambos os tipos de fontes
        all_documents = self._load_documents()
        
        if not all_documents:
            print("\n[WARNING] NENHUM DOCUMENTO FOI CARREGADO. ENCERRAMENTO DO PIPELINE.")
            return False

        # 3. Divisão em Chunks
        print("\n--- ✂️ ETAPA 2: DIVISÃO EM CHUNKS ---")
        text_chunks = TextChunker(all_documents)
        chunks = text_chunks.run_pipeline()

        # 4. Criação e Armazenamento de Vetores
        print("\n--- 💾 ETAPA 3: CRIAÇÃO E ARMAZENAMENTO DE VETORES ---")
        self._embed_and_store_chunks(chunks)
        
        print("\n===================================================")
        print("✅ PIPELINE CONCLUÍDO COM SUCESSO! BASE DE CONHECIMENTO PRONTA.")
        print("===================================================")
        return True

    def _load_documents(self):
        """Simula o carregamento de documentos de múltiplas fontes."""
        print("   > Carregando documentação de produto...")
        docs_product = [
            ("Título 1", "Descrição longa do Produto X, incluindo especificações técnicas importantes sobre o material e o uso principal.", "Product_X")
        ]
        
        print("   > Carregando artigos de artigos científicos...")
        docs_articles = [
            ("Artigo A", "Estudo sobre as tendências de IA em 2024, focando em modelos de linguagem e seu impacto na produtividade.", "Scientific_Article_A")
        ]
        
        all_docs = docs_product + docs_articles
        print(f"   > Total de documentos carregados: {len(all_docs)}")
        return all_docs

    def _embed_and_store_chunks(self, chunks):
        """Simula o processo de embedding e armazenamento vetorial."""
        print(f"[PROCESS] Início do Embedding para {len(chunks)} chunks usando {self.embedding_model}...")
        
        # Simula o processo: transformar texto -> vetor
        # Aqui, idealmente, cada chunk seria passado para a API de embedding.
        vectorized_chunks = []
        for i, chunk in enumerate(chunks):
             # Simulação do vetor: [float, float, ...]
             vector = [float(i * 0.1), float(i * 0.2), float(i * 0.3)]
             vectorized_chunks.append({"text": chunk, "vector": vector})
        
        print(f"[PROCESS] Embeddings criados. Iniciando armazenamento no {self.vector_store}...")
        
        # Simula a gravação no vetor store
        for i, item in enumerate(vectorized_chunks):
            # Comando de ingestão simulado
            print(f"      -> Indexando chunk {i+1}/{len(vectorized_chunks)}...")
            
        print(f"[SUCCESS] {len(vectorized_chunks)} vetores foram indexados com sucesso.")


#==============================================================================
#          IMPLEMENTAÇÃO DE UTILIDADES (Helper Classes)
#==============================================================================

class DocumentLoader:
    """Classe responsável por carregar dados de diferentes fontes."""
    def __init__(self):
        pass
    
    def load_data(self):
        """Retorna uma lista simulada de documentos (Título, Conteúdo, Fonte)."""
        # Dados simulados combinando fontes diferentes
        return [
            ("Título 1", "Descrição longa do Produto X, incluindo especificações técnicas importantes sobre o material e o uso principal. O custo médio é de R$ 150.", 
"Product_X"),
            ("Artigo A", "Estudo sobre as tendências de IA em 2024, focando em modelos de linguagem e seu impacto na produtividade. A curva de aprendizado é íngreme.", 
"Scientific_Article_A"),
            ("Artigo B", "Dados históricos sobre mudanças climáticas. O aquecimento global exige ações imediatas de mitigação.", "Scientific_Article_B"),
        ]


class TextChunker:
    """Classe responsável por dividir grandes blocos de texto em chunks menores."""
    def __init__(self, documents: list):
        self.documents = documents

    def run_pipeline(self) -> list:
        """Executa o processo de chunking."""
        chunks = []
        print("[CHUNK] Iniciando processo de chunking...")
        for i, (title, content, source) in enumerate(self.documents):
            # Lógica de chunking simulada: divide o conteúdo em pedaços baseados em pontuação.
            # Em um sistema real, seria usado um "RecursiveCharacterTextSplitter".
            
            # Divide o conteúdo simulado em 3 pedaços
            parts = content.split('. ')
            
            for j, part in enumerate(parts):
                # Junta o metadado com o conteúdo do chunk
                chunk_text = f"Fonte: {source} | Título: {title} | Chunk {j+1}: {part.strip()}."
                chunks.append(chunk_text)
                
        return chunks

#==============================================================================
#                 EXECUÇÃO PRINCIPAL (MAIN)
#==============================================================================

if __name__ == "__main__":
    # 1. Configuração (Onde as credenciais e parâmetros seriam lidos)
    pipeline_config = {
        "embedding_model": "bge-m3",
        "vector_store": "http://localhost:6333",
        "chunk_size": 600,
        "overlap": 100
    }
    
    # 2. Instanciação do Pipeline
    pipeline = KnowledgeBasePipeline(pipeline_config)
    
    # 3. Execução
    pipeline.run_pipeline()