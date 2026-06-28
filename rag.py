import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "documents"
EMBEDDING_DIMENSION = 384
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


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


def build_context(sources):
    context_blocks = []
    for index, source in enumerate(sources, start=1):
        context_blocks.append(
            f"[문서 {index}]\n제목: {source['title']}\n내용:\n{source['content']}"
        )
    return "\n\n".join(context_blocks)


def generate_ai_answer(question, sources):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(".env에 GEMINI_API_KEY가 설정되어 있지 않습니다.")

    context = build_context(sources)
    prompt = (
        "너는 문서 기반 질의응답 도우미다. "
        "반드시 제공된 문서 내용만 근거로 한국어로 답변한다. "
        "문서에 근거가 없으면 모른다고 답한다.\n\n"
        f"질문:\n{question}\n\n"
        f"참고 문서:\n{context}\n\n"
        "위 참고 문서를 근거로 질문에 답변해줘."
    )

    request_body = {
        "contents": [
            {
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 500,
        },
    }

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={api_key}"
    )
    request = urllib.request.Request(
        url,
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response_body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8")
        raise RuntimeError(f"Gemini API 요청 실패: {error_body}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Gemini API 연결 실패: {error.reason}") from error

    candidates = response_body.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini가 답변을 반환하지 않았습니다.")

    parts = candidates[0].get("content", {}).get("parts", [])
    answer_parts = [part.get("text", "") for part in parts]
    answer = "".join(answer_parts).strip()

    if not answer:
        raise RuntimeError("Gemini 답변이 비어 있습니다.")

    return answer


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

    try:
        answer = generate_ai_answer(question, sources)
    except RuntimeError as error:
        answer = (
            "관련 문서는 찾았지만 AI 답변 생성은 아직 완료되지 않았습니다. "
            f"이유: {error}"
        )

    return {
        "answer": answer,
        "sources": sources,
    }
