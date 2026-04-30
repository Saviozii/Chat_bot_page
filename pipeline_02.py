import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# =============================================================================
# CONFIGURAÇÕES CENTRAIS
# =============================================================================

PDF_PATH = "/home/savi021/Downloads/Ebook.pdf"

# Caminho do arquivo .txt com as descrições das imagens.
# Estrutura esperada: blocos separados por "---", cada um começando com
# [IMAGEM — Página X] seguido da descrição.

TXT_IMAGENS = "/home/savi021/Documentos/Novo_mundo/data/Infor.txt"

QDRANT_URL = "http://localhost:6333"
COLLECTION = "pdf_docs"
EMBEDDING_MODEL = "bge-m3"
VECTOR_SIZE = 1024

CHUNK_SIZE      = 600
# Garante que frases que caem na "borda" de dois chunks não se percam.
CHUNK_OVERLAP   = 100


# =============================================================================
# PASSO 1 — CARREGA O PDF
# =============================================================================
# PyPDFLoader extrai o texto de cada página como um Document separado.
# Cada Document já vem com metadata["page"] indicando o número da página.
def load_pdf() -> list[Document]:
    print(f"  Lendo: {PDF_PATH}")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # Marca cada chunk do PDF com source="pdf" nos metadados
    # para sabermos depois de onde veio cada trecho
    for doc in documents:
        doc.metadata["source"] = "pdf"

    print(f"  {len(documents)} páginas carregadas do PDF.")
    return documents


# =============================================================================
# PASSO 2 — CARREGA O TXT DE IMAGENS
# =============================================================================
# Lê o arquivo .txt e transforma cada bloco de imagem em um Document,
# extraindo o número de página do cabeçalho [IMAGEM — Página X].
def load_txt_imagens() -> list[Document]:

    # Se o arquivo não existir, avisa e retorna lista vazia
    # (o script continua e indexa só o PDF)
    if not os.path.exists(TXT_IMAGENS):
        print("  ⚠️  Arquivo de imagens não encontrado — indexando só o PDF.")
        print(f"     Esperado em: {TXT_IMAGENS}")
        return []

    print(f"  Lendo: {TXT_IMAGENS}")

    with open(TXT_IMAGENS, "r", encoding="utf-8") as f:
        conteudo = f.read()

    # Divide o arquivo nos separadores "---"
    blocos = [b.strip() for b in conteudo.split("---") if b.strip()]

    documents = []
    for bloco in blocos:
        # Tenta extrair o número da página do cabeçalho [IMAGEM — Página X]
        page_num = "?"
        linhas = bloco.splitlines()
        for linha in linhas:
            if linha.startswith("[IMAGEM") and "Página" in linha:
                try:
                    # Pega o número após "Página " e antes do "]"
                    page_num = int(linha.split("Página")[-1].replace("]", "").strip())
                except ValueError:
                    page_num = "?"
                break

        # Cria um Document com o texto do bloco e metadados de origem
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

    # Mostra quantos chunks vieram de cada fonte (PDF vs imagem)
    chunks_pdf    = [c for c in chunks if c.metadata.get("source") == "pdf"]
    chunks_imagem = [c for c in chunks if c.metadata.get("source") == "imagem"]
    print(f"  {len(chunks_pdf)} chunks do PDF")
    print(f"  {len(chunks_imagem)} chunks de imagens")
    print(f"  {len(chunks)} chunks no total")

    return chunks


# =============================================================================
# PASSO 4 — MODELO DE EMBEDDING
# =============================================================================
def get_embeddings_function() -> OllamaEmbeddings:
    print("  Carregando modelo de embedding.")
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url="http://localhost:11434",
    )


# =============================================================================
# PASSO 5 — SALVA NO QDRANT
# =============================================================================

def add_to_qdrant(chunks: list[Document]) -> None:
    client = QdrantClient(url=QDRANT_URL)

    # Apaga e recria a collection para garantir índice limpo
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        print("  🗑️  Collection antiga apagada.")

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print("  📦 Collection nova criada.")

    # Indexa todos os chunks (PDF + imagens juntos)
    # Isso pode demorar alguns minutos dependendo do tamanho do PDF
    print("  🧠 Gerando embeddings e salvando no Qdrant...")
    print("     (isso pode levar alguns minutos para 150 páginas...)")

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