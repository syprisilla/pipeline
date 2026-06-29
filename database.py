
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "rag_project")

# 예: "my@password" -> "my%40password" 로 안전하게 변환됨
safe_password = quote_plus(DB_PASSWORD)

# 3. 인코딩된 비밀번호를 넣어 URI를 조합합니다.
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{safe_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
