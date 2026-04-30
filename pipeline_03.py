import os
import logging
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from dotenv import load_dotenv

# =============================================================================
# CONFIGURAÇÕES DE PRODUÇÃO
# =============================================================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURAÇÕES CENTRAIS (PRODUÇÃO)
# =============================================================================

PDF_PATH = "/home/savi021/Downloads/Ebook.pdf"
TXT_IMAGENS = "/home/savi021/Documentos/Novo_mundo/data/Infor.txt"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "pdf_docs_prod")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
VECTOR_SIZE = 1536

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))


# =============================================================================
# PASSO 1 — CARREGA O PDF
# =============================================================================

def load_pdf() -> List[Document]:
    logger.info(f"Lendo PDF: {PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    for doc in documents:
        doc.metadata["source"] = "pdf"

    logger.info(f"{len(documents)} páginas carregadas do PDF.")
    return documents


# =============================================================================
# PASSO 2 — CARREGA O TXT DE IMAGENS
# =============================================================================

def load_txt_imagens() -> List[Document]:
    if not os.path.exists(TXT_IMAGENS):
        logger.warning(f"Arquivo de imagens não encontrado: {TXT_IMAGENS}")
        return []

    logger.info(f"Lendo imagens TXT: {TXT_IMAGENS}")

    with open(TXT_IMAGENS, "r", encoding="utf-8") as f:
        conteudo = f.read()

    blocos = [b.strip() for b in conteudo.split("---") if b.strip()]

    documents = []
    for bloco in blocos:
        page_num = "?"
        linhas = bloco.splitlines()
        for linha in linhas:
            if linha.startswith("[IMAGEM") and "Página" in linha:
                try:
                    page_num = int(linha.split("Página")[-1].replace("]", "").strip())
                except ValueError:
                    page_num = "?"
                break

        doc = Document(
            page_content=bloco,
            metadata={
                "page": page_num,
                "source": "imagem",
            }
        )
        documents.append(doc)

    logger.info(f"{len(documents)} descrições de imagens carregadas.")
    return documents


# =============================================================================
# PASSO 3 — DIVIDE EM CHUNKS
# =============================================================================

def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    chunks_pdf = [c for c in chunks if c.metadata.get("source") == "pdf"]
    chunks_imagem = [c for c in chunks if c.metadata.get("source") == "imagem"]

    logger.info(f"Chunks PDF: {len(chunks_pdf)} | Imagens: {len(chunks_imagem)} | Total: {len(chunks)}")

    return chunks


# =============================================================================
# PASSO 4 — MODELO DE EMBEDDING (GPT/OpenAI)
# =============================================================================

def get_embeddings_function() -> OpenAIEmbeddings:
    logger.info(f"Carregando modelo de embedding: {EMBEDDING_MODEL}")
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_BASE_URL,
    )


# =============================================================================
# PASSO 5 — SALVA NO QDRANT (PRODUÇÃO COM BATCH)
# =============================================================================

def add_to_qdrant(chunks: List[Document]) -> None:
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        logger.info("Collection antiga apagada.")

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    logger.info(f"Collection '{COLLECTION}' criada.")

    logger.info(f"Gerando embeddings e salvando em batches de {BATCH_SIZE}...")

    embeddings = get_embeddings_function()

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        QdrantVectorStore.from_documents(
            documents=batch,
            embedding=embeddings,
            url=QDRANT_URL,
            collection_name=COLLECTION,
        )
        logger.info(f"Batch {i // BATCH_SIZE + 1} indexado ({len(batch)} chunks).")

    logger.info(f"✅ {len(chunks)} chunks indexados com sucesso!")


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY não configurada. Verifique o arquivo .env")
        return

    logger.info("🚀 INICIANDO PIPELINE DE PRODUÇÃO (GPT)")

    pdf_docs = load_pdf()
    img_docs = load_txt_imagens()
    all_docs = pdf_docs + img_docs
    chunks = split_documents(all_docs)
    add_to_qdrant(chunks)

    logger.info("✅ PIPELINE CONCLUÍDO!")


if __name__ == "__main__":
    main()
