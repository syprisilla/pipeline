import os
from io import BytesIO

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from models import Category, Document, User
from rag import ask_rag, sync_documents, upsert_document

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

Base.metadata.create_all(bind=engine)


def dashboard_context(
    username,
    documents,
    categories=None,
    active_view="home",
    selected_category_id=None,
    keyword="",
    rag_question="",
    rag_answer="",
    rag_sources=None,
    rag_error=None,
    message=None,
):
    return {
        "username": username,
        "documents": documents,
        "categories": categories or [],
        "active_view": active_view,
        "selected_category_id": selected_category_id,
        "keyword": keyword,
        "rag_question": rag_question,
        "rag_answer": rag_answer,
        "rag_sources": rag_sources or [],
        "rag_error": rag_error,
        "message": message,
    }


def decode_text_file(file_bytes: bytes):
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise RuntimeError("텍스트 파일 인코딩을 읽을 수 없습니다.")


def extract_pdf_text(file_bytes: bytes):
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError(
            "PDF 파일을 읽으려면 pypdf가 필요합니다. requirements.txt 설치를 확인하세요."
        ) from error

    reader = PdfReader(BytesIO(file_bytes))
    page_texts = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            page_texts.append(page_text.strip())

    extracted_text = "\n\n".join(page_texts).strip()
    if not extracted_text:
        raise RuntimeError("PDF에서 텍스트를 추출하지 못했습니다.")

    return extracted_text


async def extract_upload_text(upload_file: UploadFile):
    file_bytes = await upload_file.read()
    filename = upload_file.filename or "uploaded-file"
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if not file_bytes:
        raise RuntimeError(f"{filename} 파일이 비어 있습니다.")

    if suffix == "pdf":
        return extract_pdf_text(file_bytes)

    if suffix in {"txt", "md"}:
        extracted_text = decode_text_file(file_bytes).strip()
        if not extracted_text:
            raise RuntimeError(f"{filename} 파일에서 저장할 텍스트를 찾지 못했습니다.")
        return extracted_text

    raise RuntimeError(f"{filename} 파일 형식은 지원하지 않습니다. PDF, TXT, MD만 가능합니다.")


@app.get("/")
def home():
    return RedirectResponse(url="/login", status_code=302)


@app.get("/signup")
def signup_page(request: Request):
    return templates.TemplateResponse(request, "signup.html", {"error": None})


@app.post("/signup")
def signup(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    existing_user = (
        db.query(User)
        .filter((User.username == username) | (User.email == email))
        .first()
    )

    if existing_user:
        return templates.TemplateResponse(
            request,
            "signup.html",
            {"error": "이미 사용 중인 아이디 또는 이메일입니다."},
        )

    new_user = User(username=username, email=email, password=password)
    db.add(new_user)
    db.commit()

    return RedirectResponse(url="/login", status_code=302)


@app.get("/login")
def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login")
def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = (
        db.query(User)
        .filter(User.username == username, User.password == password)
        .first()
    )

    if user is None:
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "아이디 또는 비밀번호가 틀렸습니다."},
        )

    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        dashboard_context(user.username, documents, categories),
    )


@app.get("/dashboard")
def dashboard_page(request: Request, username: str, db: Session = Depends(get_db)):
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        dashboard_context(username, documents, categories),
    )


@app.get("/documents/new")
def new_document_page(request: Request, username: str, db: Session = Depends(get_db)):
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        dashboard_context(username, documents, categories, active_view="create"),
    )


@app.get("/documents/search-page")
def search_document_page(request: Request, username: str, db: Session = Depends(get_db)):
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        dashboard_context(username, documents, categories, active_view="search"),
    )


@app.get("/rag")
def rag_page(request: Request, username: str, db: Session = Depends(get_db)):
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        dashboard_context(username, documents, categories, active_view="rag"),
    )


@app.get("/documents/list")
def document_list_page(
    request: Request,
    username: str,
    category_id: int = None,
    db: Session = Depends(get_db),
):
    categories = db.query(Category).order_by(Category.name).all()

    if category_id:
        documents = (
            db.query(Document)
            .filter(Document.category_id == category_id)
            .order_by(Document.created_at.desc())
            .all()
        )
    else:
        documents = db.query(Document).order_by(Document.created_at.desc()).all()

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="list",
            selected_category_id=category_id,
        ),
    )


@app.post("/documents")
async def create_document(
    request: Request,
    username: str = Form(...),
    title: str = Form(""),
    content: str = Form(""),
    category_id: int = Form(None),
    attachments: list[UploadFile] | None = File(None),
    db: Session = Depends(get_db),
):
    rag_error = None
    saved_count = 0
    documents_to_index = []
    categories = db.query(Category).order_by(Category.name).all()

    cleaned_title = title.strip()
    cleaned_content = content.strip()

    if cleaned_content:
        direct_document = Document(
            title=cleaned_title or "직접 작성 문서",
            content=cleaned_content,
            category_id=category_id,
        )
        db.add(direct_document)
        documents_to_index.append(direct_document)

    uploaded_files = [
        upload_file
        for upload_file in (attachments or [])
        if upload_file.filename
    ]

    for upload_file in uploaded_files:
        try:
            extracted_content = await extract_upload_text(upload_file)
        except RuntimeError as error:
            rag_error = str(error)
            continue

        file_document = Document(
            title=upload_file.filename or "업로드 문서",
            content=extracted_content,
            category_id=category_id,
        )
        db.add(file_document)
        documents_to_index.append(file_document)

    if not documents_to_index:
        documents = db.query(Document).order_by(Document.created_at.desc()).all()
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            dashboard_context(
                username,
                documents,
                categories,
                active_view="create",
                rag_error=rag_error or "직접 작성 내용 또는 업로드 파일이 필요합니다.",
            ),
        )

    db.commit()

    for document in documents_to_index:
        db.refresh(document)
        try:
            upsert_document(document)
            saved_count += 1
        except RuntimeError as error:
            rag_error = str(error)

    documents = db.query(Document).order_by(Document.created_at.desc()).all()

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="create",
            rag_error=rag_error,
            message=f"문서 {saved_count}개가 저장되었습니다.",
        ),
    )


@app.get("/documents/search")
def search_documents(
    request: Request,
    username: str,
    keyword: str = "",
    db: Session = Depends(get_db),
):
    categories = db.query(Category).order_by(Category.name).all()

    if keyword:
        search_word = f"%{keyword}%"
        documents = (
            db.query(Document)
            .filter(
                (Document.title.like(search_word))
                | (Document.content.like(search_word))
            )
            .order_by(Document.created_at.desc())
            .all()
        )
    else:
        documents = db.query(Document).order_by(Document.created_at.desc()).all()

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        dashboard_context(username, documents, categories, active_view="search", keyword=keyword),
    )


@app.post("/rag/ask")
def ask_document_question(
    request: Request,
    username: str = Form(...),
    question: str = Form(...),
    category_id: int = Form(None),
    db: Session = Depends(get_db),
):
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()
    rag_answer = ""
    rag_sources = []
    rag_error = None

    try:
        rag_result = ask_rag(question, category_id=category_id)
        rag_answer = rag_result["answer"]
        rag_sources = rag_result["sources"]
    except RuntimeError as error:
        rag_error = str(error)

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="rag",
            rag_question=question,
            rag_answer=rag_answer,
            rag_sources=rag_sources,
            rag_error=rag_error,
        ),
    )


@app.post("/categories")
def create_category(
    request: Request,
    username: str = Form(...),
    name: str = Form(...),
    db: Session = Depends(get_db),
):
    existing = db.query(Category).filter(Category.name == name.strip()).first()
    if existing:
        documents = db.query(Document).order_by(Document.created_at.desc()).all()
        categories = db.query(Category).order_by(Category.name).all()
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            dashboard_context(
                username,
                documents,
                categories,
                active_view="categories",
                rag_error="이미 존재하는 카테고리 이름입니다.",
            ),
        )

    db.add(Category(name=name.strip()))
    db.commit()

    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="categories",
            message=f'카테고리 "{name.strip()}"이 생성되었습니다.',
        ),
    )


@app.post("/categories/{category_id}/delete")
def delete_category(
    request: Request,
    category_id: int,
    username: str = Form(...),
    db: Session = Depends(get_db),
):
    category = db.query(Category).filter(Category.id == category_id).first()
    if category:
        db.query(Document).filter(Document.category_id == category_id).update(
            {"category_id": None}
        )
        db.delete(category)
        db.commit()

    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="categories",
            message="카테고리가 삭제되었습니다.",
        ),
    )


@app.get("/categories")
def categories_page(request: Request, username: str, db: Session = Depends(get_db)):
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        dashboard_context(username, documents, categories, active_view="categories"),
    )
