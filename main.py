from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from models import User

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

Base.metadata.create_all(bind=engine)


@app.get("/")
def home():
    return RedirectResponse(url="/login", status_code=302)


@app.get("/signup")
def signup_page(request: Request):
    return templates.TemplateResponse(
        request,
        "signup.html",
        {"error": None},
    )


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
    return templates.TemplateResponse(
        request,
        "login.html",
        {"error": None},
    )


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

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {"username": user.username},
    )
