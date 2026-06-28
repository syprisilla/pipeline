import hashlib
import math
from pathlib import Path


CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "documents"
EMBEDDING_DIMENSION = 384


class HashEmbeddingFunction:
    @staticmethod
    def name():
        return "hash_embedding"

    def __call__(self, input):
        return [self._embed(text) for text in input]

    def embed_query(self, input):
        return self.__call__(input)

    def embed_documents(self, input):
        return self.__call__(input)

    def get_config(self):
        return {"embedding_dimension": EMBEDDING_DIMENSION}

    @staticmethod
    def build_from_config(config):
        return HashEmbeddingFunction()

    def _embed(self, text):
        vector = [0.0] * EMBEDDING_DIMENSION
        normalized_text = text.lower().strip()

        if not normalized_text:
            return vector

        tokens = normalized_text.split()
        tokens.extend(normalized_text[index : index + 2] for index in range(len(normalized_text) - 1))

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % EMBEDDING_DIMENSION
            vector[bucket] += 1.0

        length = math.sqrt(sum(value * value for value in vector))
        if length == 0:
            return vector

        return [value / length for value in vector]


def get_collection():
    try:
        import chromadb
    except ImportError as error:
        raise RuntimeError(
            "Chroma가 설치되어 있지 않습니다. python -m pip install -r requirements.txt 를 실행하세요."
        ) from error

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=HashEmbeddingFunction(),
    )


def document_to_text(document):
    return f"제목: {document.title}\n내용: {document.content}"


def upsert_document(document):
    collection = get_collection()
    collection.upsert(
        ids=[str(document.id)],
        documents=[document_to_text(document)],
        metadatas=[
            {
                "document_id": document.id,
                "title": document.title,
            }
        ],
    )


def sync_documents(documents):
    for document in documents:
        upsert_document(document)


def ask_rag(question, limit=3):
    collection = get_collection()
    result = collection.query(query_texts=[question], n_results=limit)

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    sources = []
    for index, document_text in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None
        sources.append(
            {
                "title": metadata.get("title", "제목 없음"),
                "content": document_text,
                "distance": distance,
            }
        )

    if not sources:
        return {
            "answer": "관련 문서를 찾지 못했습니다.",
            "sources": [],
        }

    answer = (
        "질문과 가장 관련 있는 문서를 찾았습니다. "
        "아래 출처 문서의 내용을 바탕으로 답변을 작성하면 됩니다."
    )

    return {
        "answer": answer,
        "sources": sources,
    }
