from langchain_community.embeddings import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
# =============================================================================
# CONFIGURAÇÕES CENTRAIS
# =============================================================================
QDRANT_URL      = "http://localhost:6333"
COLLECTION      = "pdf_docs"
EMBEDDING_MODEL = "bge-m3"
LLM_MODEL       = "llama3.2"
OLLAMA_URL      = "http://localhost:11434"

# Quantos chunks buscar antes de filtrar por score
TOP_K_BUSCA     = 8
# Quantos chunks manter após o filtro (os mais relevantes)
TOP_K_FINAL     = 5
# Score mínimo de similaridade para aceitar um chunk.
SCORE_MINIMO    = 0.4
# Quantas trocas de conversa anteriores incluir no prompt
MAX_HISTORICO   = 3

PROMPT_TEMPLATE = """
Você é Guida, assistente especializada no Guia de Acesso e Permanência nas Universidades.
Responda sempre em português, de forma clara e objetiva.
Regras obrigatórias:
- Responda como uma pessoa real conversando, não como um relatório formal.
- Use linguagem simples e acolhedora, como se estivesse ajudando um amigo.
- Responda somente com base no CONTEXTO DO DOCUMENTO ou no HISTÓRICO DA CONVERSA.
- Se o usuário mencionou algo sobre si mesmo no histórico (como nome, curso,
  situação), use essa informação para personalizar a resposta.
- Se a informação não estiver nem no contexto nem no histórico, diga exatamente:
  "Não encontrei essa informação no documento."
- NUNCA complemente uma resposta com suposições, sugestões ou informações
  que não estejam literalmente no CONTEXTO ou HISTÓRICO abaixo.
- Após responder a pergunta com os dados do contexto, PARE. Não adicione
  frases como "posso ajudá-lo com mais informações" ou "se precisar de
  mais detalhes". Responda apenas o que foi perguntado.
- Quando possível, cite o número da página entre colchetes, ex: [Página 12].
- Se a resposta envolver uma lista de itens, use marcadores (•).
- PROIBIDO inventar qualquer dado como nomes, e-mails, telefones,
  redes sociais, endereços ou valores numéricos. Se não estiver
  literalmente no CONTEXTO ou HISTÓRICO abaixo, diga que não encontrou.
- Se a pergunta for uma saudação ou conversa informal, responda
  normalmente e de forma simpática.
- NUNCA comece a resposta com "Infelizmente" ou frases negativas. Se tiver
  a informação, vá direto ao ponto.
- NUNCA agrupe informações em "Parte 1", "Parte 2" ou estruturas que não
  estejam literalmente no documento. Cite apenas o que está escrito.
- NUNCA invente intervalos de páginas como "Páginas 10-23". Cite apenas
  páginas que aparecem literalmente no contexto recebido.
  
HISTÓRICO DA CONVERSA (informações que o usuário já mencionou):
{historico}

CONTEXTO DO DOCUMENTO:
{context}

PERGUNTA ATUAL:
{question}

RESPOSTA (cite as páginas quando relevante):
"""

# =============================================================================
# PASSO 1 — CONEXÃO COM O BANCO VETORIAL
# =============================================================================
def get_vector_store() -> QdrantVectorStore:
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_URL,
    )
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION,
    )

# =============================================================================
# PASSO 2 — BUSCA DE CONTEXTO NO QDRANT
# =============================================================================
def buscar_contexto(pergunta: str) -> list[tuple]:
    store = get_vector_store()

    # Busca os TOP_K_BUSCA chunks mais similares no espaço vetorial
    resultados = store.similarity_search_with_score(pergunta, k=TOP_K_BUSCA)

    # Remove chunks com score baixo (provavelmente irrelevantes para a pergunta)
    relevantes = [
        (doc, score)
        for doc, score in resultados
        if score >= SCORE_MINIMO
    ]

    # Prioriza chunks vindos do TXT de imagens somando um bônus ao score
    BOOST_IMAGEM = 0.25

    def score_com_boost(item):
        doc, score = item
        if doc.metadata.get("source") == "imagem":
            return score + BOOST_IMAGEM
        return score

    # Ordena usando o score com boost aplicado
    relevantes.sort(key=score_com_boost, reverse=True)

    # Devolve apenas os TOP_K_FINAL melhores
    return relevantes[:TOP_K_FINAL]


# =============================================================================
# PASSO 3 — FORMATAÇÃO DO HISTÓRICO
# =============================================================================
def formatar_historico(historico: list[dict]) -> str:
    if not historico:
        return "(sem histórico — esta é a primeira pergunta)"

    linhas = []
    # Usa apenas as últimas MAX_HISTORICO trocas para não sobrecarregar o prompt
    for troca in historico[-MAX_HISTORICO:]:
        linhas.append(f"Usuário: {troca['pergunta']}")
        linhas.append(f"Guida: {troca['resposta']}")
        linhas.append("")
    return "\n".join(linhas)


# =============================================================================
# PASSO 4 — FUNÇÃO PRINCIPAL: PERGUNTAR
# =============================================================================
# Orquestra tudo: busca → monta prompt → chama LLM → retorna resposta.
#
# Parâmetros:
#   pergunta  → texto da pergunta do usuário
#   historico → lista de dicts {"pergunta": ..., "resposta": ...}
#               começa vazia e cresce a cada turno no loop de chat
#
# Retorna:
#   "resposta" → texto gerado pelo LLM
#   "fontes"   → lista com página, score e trecho de cada chunk usado

def perguntar(pergunta: str, historico: list[dict] = []) -> dict:
    # 1. Busca os chunks relevantes no Qdrant
    resultados = buscar_contexto(pergunta)

    # Se não encontrou nada acima do score mínimo, retorna mensagem padrão
    if not resultados:
        return {
            "resposta": "Não encontrei informações relevantes no documento para essa pergunta.",
            "fontes": [],
        }

    # 2. Monta o bloco de contexto para o prompt
    # Cada chunk aparece com sua página e origem (pdf ou imagem)
    contexto = "\n\n---\n\n".join(
        "[Página {pagina} | Fonte: {fonte}]\n{texto}".format(
            pagina=doc.metadata.get("page", "?"),
            # Mostra de onde veio o chunk: PDF ou descrição de imagem
            fonte=doc.metadata.get("source", "pdf"),
            texto=doc.page_content
        )
        for doc, score in resultados
    )

    # 3. Formata o histórico da conversa
    historico_str = formatar_historico(historico)

    # 4. Monta o prompt completo
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    prompt = prompt_template.format(
        historico=historico_str,
        context=contexto,
        question=pergunta,
    )

    # 5. Chama o LLM local via Ollama
    llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_URL)
    resposta = llm.invoke(prompt)

    # 6. Monta a lista de fontes usadas (para exibir no chat)
    fontes = [
        {
            "page":   doc.metadata.get("page", "?"),
            "source": doc.metadata.get("source", "pdf"),
            "score":  round(score, 3),
            # Preview dos primeiros 150 caracteres do trecho
            "trecho": doc.page_content[:150] + "...",
        }
        for doc, score in resultados
    ]

    return {
        "resposta": resposta,
        "fontes":   fontes,
    }

if __name__ == "__main__":