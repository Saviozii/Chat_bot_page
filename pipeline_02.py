import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


# CONFIGURAÇÕES CENTRAIS


PDF_PATH = "data/Ebook.pdf"
TXT_IMAGENS = "data/Infor.txt"

QDRANT_URL = "http://localhost:6333"
COLLECTION = "pdf_docs"
EMBEDDING_MODEL = "bge-m3"
VECTOR_SIZE = 1024

CHUNK_SIZE      = 600

CHUNK_OVERLAP   = 100

def load_pdf() -> list[Document]:
    print(f"  Lendo: {PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    for doc in documents:
        doc.metadata["source"] = "pdf"

    print(f"  {len(documents)} páginas carregadas do PDF.")
    return documents



def load_txt_imagens() -> list[Document]:

    if not os.path.exists(TXT_IMAGENS):
        print("   Arquivo de imagens não encontrado — indexando só o PDF.")
        print(f"     Esperado em: {TXT_IMAGENS}")
        return []

    print(f"  Lendo: {TXT_IMAGENS}")

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
                "page":   page_num,
                "source": "imagem",  # distingue dos chunks do PDF
            }
        )
        documents.append(doc)

    print(f"  {len(documents)} descrições de imagens carregadas.")
    return documents


# =============================================================================
# PASSO 3 — DIVIDE EM CHUNKS
# =============================================================================
def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    chunks_pdf    = [c for c in chunks if c.metadata.get("source") == "pdf"]
    chunks_imagem = [c for c in chunks if c.metadata.get("source") == "imagem"]
    print(f"  {len(chunks_pdf)} chunks do PDF")
    print(f"  {len(chunks_imagem)} chunks de imagens")
    print(f"  {len(chunks)} chunks no total")

    return chunks


def get_embeddings_function() -> OllamaEmbeddings:
    print("  Carregando modelo de embedding.")
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url="http://localhost:11434",
    )


def add_to_qdrant(chunks: list[Document]) -> None:
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        print("  🗑️  Collection antiga apagada.")

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print("Collection nova criada.")

    print("Gerando embeddings e salvando no Qdrant...")
    print("(isso pode levar alguns minutos para 150 páginas...)")

    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=get_embeddings_function(),
        url=QDRANT_URL,
        collection_name=COLLECTION,
    )

    print(f"  ✅ {len(chunks)} chunks indexados com sucesso!")


#Funcao que faz rodar tudo
def main():
    pdf_docs = load_pdf()

    img_docs = load_txt_imagens()

    all_docs = pdf_docs + img_docs
    chunks = split_documents(all_docs)

    add_to_qdrant(chunks)

if __name__ == "__main__":
    main()
