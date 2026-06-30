import os
import base64
import re
from io import BytesIO, StringIO

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from kaggle_pipeline import download_and_preprocess_dataset
from models import Category, Document, User
from rag import ask_rag, upsert_document

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

Base.metadata.create_all(bind=engine)


def normalize_search_tokens(text):
    normalized = text.lower()
    normalized = re.sub(r"\.(csv|txt|md|pdf)\b", " ", normalized)
    tokens = re.split(r"[^0-9a-z가-힣]+", normalized)
    stop_words = {
        "csv",
        "txt",
        "md",
        "pdf",
        "file",
        "data",
        "dataset",
        "train",
        "test",
        "about",
        "explain",
        "설명",
        "데이터",
        "파일",
        "문서",
        "대해",
        "대한",
        "관련",
    }
    token_set = {
        token
        for token in tokens
        if len(token) >= 2 and token not in stop_words and not token.isdigit()
    }
    synonyms = {
        "지하철": "subway",
        "전철": "subway",
        "아파트": "apartment",
        "부동산": "apartment",
        "가격": "price",
        "기상": "weather",
        "날씨": "weather",
    }
    for token, synonym in synonyms.items():
        if token in token_set:
            token_set.add(synonym)
        if synonym in token_set:
            token_set.add(token)

    return token_set


def find_title_matched_document_ids(question, documents):
    question_tokens = normalize_search_tokens(question)
    if not question_tokens:
        return []

    scored_documents = []
    for document in documents:
        title_tokens = normalize_search_tokens(document.title)
        if not title_tokens:
            continue

        exact_overlap = question_tokens & title_tokens
        partial_overlap = {
            question_token
            for question_token in question_tokens
            for title_token in title_tokens
            if question_token in title_token or title_token in question_token
        }
        score = (len(exact_overlap) * 2) + len(partial_overlap)
        if score:
            scored_documents.append((score, document.id))

    scored_documents.sort(reverse=True)
    return [document_id for _, document_id in scored_documents[:5]]


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
    error=None,
    message=None,
    pipeline_stats=None,
    eda_charts=None,
    csv_profiles=None,
    csv_documents=None,
    selected_document=None,
    target_column="",
    ml_result=None,
    preprocess_summary=None,
    kaggle_dataset_id="",
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
        "rag_error": rag_error or error,
        "error": error or rag_error,
        "message": message,
        "pipeline_stats": pipeline_stats or {},
        "eda_charts": eda_charts or {},
        "csv_profiles": csv_profiles or [],
        "csv_documents": csv_documents or [],
        "selected_document": selected_document,
        "target_column": target_column,
        "ml_result": ml_result,
        "preprocess_summary": preprocess_summary,
        "kaggle_dataset_id": kaggle_dataset_id,
    }


def decode_text_file(file_bytes: bytes):
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue

    raise RuntimeError("텍스트 파일 인코딩을 읽을 수 없습니다.")


def save_public_data_file(filename: str, file_bytes: bytes):
    safe_name = os.path.basename(filename).replace("\\", "_").replace("/", "_")
    storage_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_files")
    os.makedirs(storage_dir, exist_ok=True)

    path = os.path.join(storage_dir, safe_name)
    name, extension = os.path.splitext(safe_name)
    counter = 1
    while os.path.exists(path):
        path = os.path.join(storage_dir, f"{name}_{counter}{extension}")
        counter += 1

    with open(path, "wb") as saved_file:
        saved_file.write(file_bytes)

    return path


def extract_pdf_text(file_bytes: bytes):
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError(
            "PDF 파일을 읽으려면 pypdf가 필요합니다. requirements.txt 설치를 확인하세요."
        ) from error

    reader = PdfReader(BytesIO(file_bytes))
    page_texts = []

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            page_texts.append(f"[페이지 {page_num:02d}]\n{page_text.strip()}")

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

    if suffix == "csv":
        save_public_data_file(filename, file_bytes)
        extracted_text = decode_text_file(file_bytes).strip()
        if not extracted_text:
            raise RuntimeError(f"{filename} CSV 파일에서 저장할 텍스트를 찾지 못했습니다.")
        return extracted_text

    raise RuntimeError(f"{filename} 파일 형식은 지원하지 않습니다. PDF, TXT, MD, CSV만 가능합니다.")


def build_pipeline_stats(documents, categories=None):
    total_chars = sum(len(document.content or "") for document in documents)
    return {
        "source_count": len(documents),
        "category_count": len(categories or []),
        "vector_count": len(documents),
        "table_count": 1,
        "csv_count": sum(1 for document in documents if document.title.lower().endswith(".csv")),
        "total_chars": total_chars,
    }


def make_chart_uri(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buffer.seek(0)
    return "data:image/png;base64," + base64.b64encode(buffer.read()).decode("ascii")


def read_csv_documents(documents, pd):
    csv_frames = []

    for document in documents:
        if not document.title.lower().endswith(".csv"):
            continue

        try:
            dataframe = pd.read_csv(StringIO(document.content))
        except Exception:
            continue

        if dataframe.empty:
            continue

        csv_frames.append(
            {
                "title": document.title,
                "document": document,
                "dataframe": dataframe,
            }
        )

    return csv_frames


def build_csv_profile(document, dataframe, pd):
    title = document.title
    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    date_columns = []

    for column in dataframe.columns:
        if column in numeric_columns:
            continue

        parsed = pd.to_datetime(dataframe[column], errors="coerce")
        if parsed.notna().mean() >= 0.6:
            date_columns.append(column)

    missing_counts = dataframe.isna().sum()
    top_missing = [
        {"column": column, "count": int(count)}
        for column, count in missing_counts.sort_values(ascending=False).head(5).items()
        if int(count) > 0
    ]
    categorical_columns = [
        column
        for column in dataframe.columns
        if column not in numeric_columns and column not in date_columns
    ]
    total_cells = len(dataframe) * len(dataframe.columns)
    missing_total = int(missing_counts.sum())
    missing_ratio = (missing_total / total_cells * 100) if total_cells else 0

    lower_columns = {column.lower(): column for column in dataframe.columns}
    lower_title = title.lower()
    if any(keyword in lower_title for keyword in ("weather", "기상")) or {
        "temperature",
        "precipitation",
        "visibility",
    }.intersection(lower_columns):
        data_domain = "기상 관측 데이터"
        data_character = "날짜, 관측 지점, 날씨 관련 수치가 함께 들어 있어 시간·지역별 기상 변화를 살펴보는 데 적합합니다."
    elif any(keyword in lower_title for keyword in ("subway", "지하철")):
        data_domain = "지하철 이용 데이터"
        data_character = "역이나 시간 단위의 이용 패턴을 비교하고 수요 변화를 분석하는 데 적합합니다."
    elif any(keyword in lower_title for keyword in ("apartment", "apt", "아파트", "price")):
        data_domain = "부동산 가격 데이터"
        data_character = "가격과 위치·면적 같은 설명 변수를 함께 보며 가격 흐름이나 영향 요인을 탐색하는 데 적합합니다."
    else:
        data_domain = "CSV 기반 공공데이터"
        data_character = "여러 행의 관측값과 변수로 구성되어 전체 분포, 결측치, 변수 관계를 탐색하는 데 적합합니다."

    period_text = "명확한 날짜 범위는 감지되지 않았습니다."
    if date_columns:
        date_column = date_columns[0]
        parsed_dates = pd.to_datetime(dataframe[date_column], errors="coerce").dropna()
        if not parsed_dates.empty:
            start_date = parsed_dates.min().date()
            end_date = parsed_dates.max().date()
            period_text = f"{date_column} 기준 {start_date}부터 {end_date}까지의 기간이 포함됩니다."
    elif "year" in lower_columns:
        year_values = pd.to_numeric(dataframe[lower_columns["year"]], errors="coerce").dropna()
        if not year_values.empty:
            period_text = f"year 기준 {int(year_values.min())}년부터 {int(year_values.max())}년까지의 값이 포함됩니다."

    category_name = document.category.name if document.category else "미분류"
    created_at = document.created_at.strftime("%Y-%m-%d %H:%M") if document.created_at else "알 수 없음"
    representative_columns = dataframe.columns.tolist()[:8]
    representative_text = ", ".join(representative_columns)
    if len(dataframe.columns) > len(representative_columns):
        representative_text = f"{representative_text} 외 {len(dataframe.columns) - len(representative_columns)}개"

    if missing_total:
        quality_text = f"전체 셀 중 결측치는 {missing_total}개로 약 {missing_ratio:.1f}%입니다."
    else:
        quality_text = "감지된 결측치는 없어 기본적인 데이터 완성도는 좋은 편입니다."

    overview = {
        "source": f"{title} 파일에서 읽어온 데이터입니다. 앱에는 '{category_name}' 카테고리로 저장되어 있으며 저장 시각은 {created_at}입니다.",
        "summary": f"{data_domain}로 보이며, 총 {len(dataframe)}개의 관측 행과 {len(dataframe.columns)}개의 변수로 구성되어 있습니다. {data_character}",
        "period": period_text,
        "structure": f"수치형 변수 {len(numeric_columns)}개, 범주/문자형 변수 {len(categorical_columns)}개, 날짜형 변수 {len(date_columns)}개가 감지되었습니다.",
        "columns": f"대표 컬럼은 {representative_text}입니다.",
        "quality": quality_text,
    }

    return {
        "title": title,
        "row_count": len(dataframe),
        "column_count": len(dataframe.columns),
        "columns": dataframe.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "date_columns": date_columns,
        "top_missing": top_missing,
        "overview": overview,
    }


def build_eda_charts(documents, include_storage_charts=True):
    matplotlib_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib_cache")
    os.makedirs(matplotlib_cache, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", matplotlib_cache)

    try:
        import pandas as pd
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("EDA 시각화를 사용하려면 pandas와 matplotlib 설치가 필요합니다.") from error

    csv_frames = read_csv_documents(documents, pd)
    csv_profiles = [
        build_csv_profile(csv_file["document"], csv_file["dataframe"], pd)
        for csv_file in csv_frames
    ]

    plt.rcParams["font.family"] = ["Malgun Gothic", "Arial", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    charts = {}

    if include_storage_charts:
        rows = [
            {
                "title": document.title,
                "category": document.category.name if document.category else "미분류",
                "content_length": len(document.content or ""),
                "created_at": document.created_at,
            }
            for document in documents
        ]

        if rows:
            dataframe = pd.DataFrame(rows)
            dataframe["created_date"] = pd.to_datetime(dataframe["created_at"]).dt.date

            category_counts = dataframe.groupby("category").size().sort_values(ascending=False)
            fig_category, ax_category = plt.subplots(figsize=(7.2, 4.2))
            category_counts.plot(kind="bar", ax=ax_category, color="#4a6fe3")
            ax_category.set_title("카테고리별 수집 데이터 수")
            ax_category.set_xlabel("카테고리")
            ax_category.set_ylabel("건수")
            ax_category.grid(axis="y", alpha=0.22)
            ax_category.tick_params(axis="x", rotation=20)
            charts["category"] = make_chart_uri(fig_category)
            plt.close(fig_category)

            date_counts = dataframe.groupby("created_date").size()
            fig_time, ax_time = plt.subplots(figsize=(7.2, 4.2))
            date_counts.plot(kind="line", marker="o", ax=ax_time, color="#50c878", linewidth=2.5)
            ax_time.set_title("날짜별 저장 추이")
            ax_time.set_xlabel("저장일")
            ax_time.set_ylabel("건수")
            ax_time.grid(alpha=0.24)
            charts["time"] = make_chart_uri(fig_time)
            plt.close(fig_time)

            fig_length, ax_length = plt.subplots(figsize=(7.2, 4.2))
            dataframe["content_length"].plot(kind="hist", bins=min(8, max(3, len(dataframe))), ax=ax_length, color="#2c5282")
            ax_length.set_title("텍스트 길이 분포")
            ax_length.set_xlabel("문자 수")
            ax_length.set_ylabel("문서 수")
            ax_length.grid(axis="y", alpha=0.22)
            charts["length"] = make_chart_uri(fig_length)
            plt.close(fig_length)

    if csv_frames:
        first_csv = csv_frames[0]
        csv_title = first_csv["title"]
        csv_dataframe = first_csv["dataframe"]
        numeric_dataframe = csv_dataframe.select_dtypes(include="number")
        helper_numeric_names = {
            "id",
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "weekday",
            "day_of_week",
            "week",
            "quarter",
        }
        analysis_numeric_columns = [
            column
            for column in numeric_dataframe.columns
            if column.lower() not in helper_numeric_names
        ]
        analysis_numeric_dataframe = numeric_dataframe[analysis_numeric_columns] if analysis_numeric_columns else numeric_dataframe

        if not analysis_numeric_dataframe.empty:
            numeric_summary = (
                analysis_numeric_dataframe.agg(["mean", "max"])
                .transpose()
                .sort_values("mean", ascending=False)
                .head(8)
            )
            fig_numeric, ax_numeric = plt.subplots(figsize=(7.8, 4.5))
            numeric_summary.plot(kind="bar", ax=ax_numeric, color=["#4a6fe3", "#50c878"])
            ax_numeric.set_title(f"{csv_title} 주요 수치 변수 평균/최댓값")
            ax_numeric.set_xlabel("수치 변수")
            ax_numeric.set_ylabel("값")
            ax_numeric.grid(axis="y", alpha=0.22)
            ax_numeric.tick_params(axis="x", rotation=25)
            charts["csv_numeric"] = make_chart_uri(fig_numeric)
            widest_column = numeric_summary.assign(
                gap=numeric_summary["max"] - numeric_summary["mean"]
            ).sort_values("gap", ascending=False).index[0]
            charts["csv_numeric_note"] = f"이 데이터에서는 {', '.join(numeric_summary.index.tolist())} 변수를 비교했습니다. 평균과 최댓값 차이가 가장 큰 변수는 {widest_column}로, 일부 관측값이 평균보다 크게 튀는 구간이 있습니다."
            plt.close(fig_numeric)

        date_column = None
        parsed_dates = None
        for column in csv_dataframe.columns:
            if column in numeric_dataframe.columns:
                continue
            candidate_dates = pd.to_datetime(csv_dataframe[column], errors="coerce")
            if candidate_dates.notna().mean() >= 0.6:
                date_column = column
                parsed_dates = candidate_dates
                break

        if date_column and not analysis_numeric_dataframe.empty:
            trend_columns = analysis_numeric_dataframe.columns.tolist()[:3]
            trend_dataframe = pd.DataFrame({"date": parsed_dates})
            for column in trend_columns:
                trend_dataframe[column] = pd.to_numeric(csv_dataframe[column], errors="coerce")
            trend_dataframe = trend_dataframe.dropna(subset=["date"])

            if not trend_dataframe.empty:
                trend_dataframe["date"] = trend_dataframe["date"].dt.date
                trend_frame = trend_dataframe.groupby("date")[trend_columns].mean()
                fig_csv_time, ax_csv_time = plt.subplots(figsize=(7.8, 4.5))
                trend_frame.plot(kind="line", marker="o", ax=ax_csv_time, linewidth=2.3)
                ax_csv_time.set_title(f"{date_column} 기준 주요 수치 변수 평균 추이")
                ax_csv_time.set_xlabel("날짜")
                ax_csv_time.set_ylabel("평균값")
                ax_csv_time.grid(alpha=0.24)
                charts["csv_time"] = make_chart_uri(fig_csv_time)
                most_variable = trend_frame.std(numeric_only=True).sort_values(ascending=False).index[0]
                start_date = trend_frame.index.min()
                end_date = trend_frame.index.max()
                charts["csv_time_note"] = f"{date_column} 기준 {start_date}부터 {end_date}까지의 흐름을 보면 {', '.join(trend_columns)} 중 {most_variable}의 날짜별 변동이 가장 큽니다."
                plt.close(fig_csv_time)

        day_column = next(
            (column for column in csv_dataframe.columns if column.lower() in {"day_of_week", "weekday", "day_name"}),
            None,
        )
        if day_column and not analysis_numeric_dataframe.empty:
            weekday_columns = analysis_numeric_dataframe.columns.tolist()[:3]
            weekday_frame = csv_dataframe[[day_column] + weekday_columns].copy()
            for column in weekday_columns:
                weekday_frame[column] = pd.to_numeric(weekday_frame[column], errors="coerce")
            weekday_summary = weekday_frame.groupby(day_column)[weekday_columns].mean()
            fig_weekday, ax_weekday = plt.subplots(figsize=(7.8, 4.5))
            weekday_summary.plot(kind="bar", ax=ax_weekday)
            ax_weekday.set_title(f"{day_column} 기준 주요 수치 변수 평균")
            ax_weekday.set_xlabel("요일")
            ax_weekday.set_ylabel("평균값")
            ax_weekday.grid(axis="y", alpha=0.22)
            ax_weekday.tick_params(axis="x", rotation=20)
            charts["csv_weekday"] = make_chart_uri(fig_weekday)
            first_weekday_column = weekday_columns[0]
            top_weekday = weekday_summary[first_weekday_column].idxmax()
            charts["csv_weekday_note"] = f"{day_column}별 평균을 비교하면 {first_weekday_column} 값은 {top_weekday}에서 가장 높게 나타납니다. 나머지 변수도 요일별 높낮이를 함께 비교할 수 있습니다."
            plt.close(fig_weekday)

        if len(analysis_numeric_dataframe.columns) >= 2:
            corr_columns = analysis_numeric_dataframe.columns.tolist()[:8]
            corr_frame = analysis_numeric_dataframe[corr_columns].corr()
            fig_corr, ax_corr = plt.subplots(figsize=(7.8, 5.2))
            image = ax_corr.imshow(corr_frame, cmap="coolwarm", vmin=-1, vmax=1)
            ax_corr.set_title("주요 수치 변수 상관관계")
            ax_corr.set_xticks(range(len(corr_columns)))
            ax_corr.set_xticklabels(corr_columns, rotation=35, ha="right")
            ax_corr.set_yticks(range(len(corr_columns)))
            ax_corr.set_yticklabels(corr_columns)
            fig_corr.colorbar(image, ax=ax_corr, fraction=0.046, pad=0.04)
            for row_index in range(len(corr_columns)):
                for column_index in range(len(corr_columns)):
                    ax_corr.text(
                        column_index,
                        row_index,
                        f"{corr_frame.iloc[row_index, column_index]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="#172033",
                    )
            charts["csv_corr"] = make_chart_uri(fig_corr)
            corr_pairs = corr_frame.where(~pd.DataFrame(
                [[row == column for column in corr_columns] for row in corr_columns],
                index=corr_columns,
                columns=corr_columns,
            )).abs().stack()
            strongest_pair = corr_pairs.idxmax()
            strongest_value = corr_frame.loc[strongest_pair[0], strongest_pair[1]]
            charts["csv_corr_note"] = f"상관관계가 가장 강한 조합은 {strongest_pair[0]}와 {strongest_pair[1]}이며, 상관계수는 {strongest_value:.2f}입니다."
            plt.close(fig_corr)

    return {
        "charts": charts,
        "csv_profiles": csv_profiles,
    }


def build_preprocess_charts(documents):
    matplotlib_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib_cache")
    os.makedirs(matplotlib_cache, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", matplotlib_cache)

    try:
        import pandas as pd
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("전처리 시각화를 사용하려면 pandas와 matplotlib 설치가 필요합니다.") from error

    csv_frames = read_csv_documents(documents, pd)
    csv_profiles = [
        build_csv_profile(csv_file["document"], csv_file["dataframe"], pd)
        for csv_file in csv_frames
    ]
    charts = {}
    preprocess_summary = None

    plt.rcParams["font.family"] = ["Malgun Gothic", "Arial", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    if csv_frames:
        first_csv = csv_frames[0]
        csv_dataframe = first_csv["dataframe"]
        numeric_columns = csv_dataframe.select_dtypes(include="number").columns.tolist()
        date_columns = []
        for column in csv_dataframe.columns:
            if column in numeric_columns:
                continue
            parsed = pd.to_datetime(csv_dataframe[column], errors="coerce")
            if parsed.notna().mean() >= 0.6:
                date_columns.append(column)

        categorical_columns = [
            column
            for column in csv_dataframe.columns
            if column not in numeric_columns and column not in date_columns
        ]
        before_missing = csv_dataframe.isna().sum()
        processed_dataframe = csv_dataframe.copy()

        for column in date_columns:
            processed_dataframe[column] = pd.to_datetime(processed_dataframe[column], errors="coerce")
        for column in numeric_columns:
            numeric_series = pd.to_numeric(processed_dataframe[column], errors="coerce")
            processed_dataframe[column] = numeric_series.fillna(numeric_series.median())
        for column in categorical_columns:
            codes, _ = pd.factorize(processed_dataframe[column].fillna("missing").astype(str))
            processed_dataframe[column] = codes

        after_missing = processed_dataframe.isna().sum()
        missing_summary = before_missing.sort_values(ascending=False).head(10)
        missing_total = int(before_missing.sum())
        if missing_total:
            fig_missing, ax_missing = plt.subplots(figsize=(7.8, 4.5))
            missing_summary.plot(kind="bar", ax=ax_missing, color="#f59e0b")
            ax_missing.set_title("컬럼별 결측치 개수")
            ax_missing.set_xlabel("컬럼")
            ax_missing.set_ylabel("결측치")
            ax_missing.grid(axis="y", alpha=0.22)
            ax_missing.tick_params(axis="x", rotation=25)
            charts["missing_by_column"] = make_chart_uri(fig_missing)
            plt.close(fig_missing)

            top_missing_column = before_missing.sort_values(ascending=False).index[0]
            charts["missing_by_column_note"] = f"전처리 전 결측치는 총 {missing_total}개이며, 가장 많이 비어 있는 컬럼은 {top_missing_column}입니다."

            compare_frame = pd.DataFrame(
                {
                    "전처리 전": before_missing,
                    "전처리 후": after_missing,
                }
            ).sort_values("전처리 전", ascending=False).head(10)
            fig_compare, ax_compare = plt.subplots(figsize=(7.8, 4.5))
            compare_frame.plot(kind="bar", ax=ax_compare, color=["#f59e0b", "#50c878"])
            ax_compare.set_title("전처리 전/후 결측치 비교")
            ax_compare.set_xlabel("컬럼")
            ax_compare.set_ylabel("결측치")
            ax_compare.grid(axis="y", alpha=0.22)
            ax_compare.tick_params(axis="x", rotation=25)
            charts["missing_before_after"] = make_chart_uri(fig_compare)
            after_total = int(after_missing.sum())
            charts["missing_before_after_note"] = f"결측치는 전처리 전 {missing_total}개에서 전처리 후 {after_total}개로 정리됩니다."
            plt.close(fig_compare)
        else:
            charts["missing_status"] = "이 데이터는 전처리 전부터 감지된 결측치가 없습니다. 따라서 결측치 개수 그래프와 전/후 비교 그래프는 생략했습니다."

        if numeric_columns:
            box_columns = numeric_columns[:4]
            before_box = csv_dataframe[box_columns].apply(pd.to_numeric, errors="coerce")
            after_box = before_box.copy()
            for column in box_columns:
                q1 = after_box[column].quantile(0.25)
                q3 = after_box[column].quantile(0.75)
                iqr = q3 - q1
                if pd.notna(iqr) and iqr > 0:
                    after_box[column] = after_box[column].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

            fig_box, axes = plt.subplots(1, 2, figsize=(9.4, 4.5), sharey=False)
            before_box.plot(kind="box", ax=axes[0], rot=25)
            after_box.plot(kind="box", ax=axes[1], rot=25)
            axes[0].set_title("이상치 처리 전")
            axes[1].set_title("이상치 처리 후")
            for axis in axes:
                axis.grid(axis="y", alpha=0.22)
            charts["outlier_boxplot"] = make_chart_uri(fig_box)
            changed_count = int((before_box.fillna(0) != after_box.fillna(0)).sum().sum())
            charts["outlier_boxplot_note"] = f"{', '.join(box_columns)} 기준으로 IQR 범위를 벗어난 값 {changed_count}개가 경계값 안으로 조정됩니다."
            plt.close(fig_box)

        preprocess_summary = {
            "missing": f"결측치는 수치형 {len(numeric_columns)}개 컬럼은 중앙값으로, 범주형 {len(categorical_columns)}개 컬럼은 missing 값으로 보완합니다.",
            "date": f"날짜로 해석 가능한 컬럼 {len(date_columns)}개를 datetime 형식으로 변환합니다.",
            "outlier": "수치형 변수는 IQR 기준으로 과도하게 튀는 값을 상·하한 경계로 조정합니다.",
            "numeric": f"수치형 변수 {len(numeric_columns)}개는 모델링과 시각화를 위해 숫자 타입으로 정리합니다.",
            "categorical": f"범주형 변수 {len(categorical_columns)}개는 문자열 결측치를 정리한 뒤 숫자 코드로 인코딩합니다.",
        }

    return {
        "charts": charts,
        "csv_profiles": csv_profiles,
        "preprocess_summary": preprocess_summary,
    }


def build_ml_analysis(document, target_column):
    matplotlib_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib_cache")
    os.makedirs(matplotlib_cache, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", matplotlib_cache)

    try:
        import pandas as pd
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler
    except ImportError as error:
        raise RuntimeError("머신러닝 분석을 사용하려면 scikit-learn 설치가 필요합니다.") from error

    plt.rcParams["font.family"] = ["Malgun Gothic", "Arial", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    dataframe = pd.read_csv(StringIO(document.content))
    if target_column not in dataframe.columns:
        raise RuntimeError("선택한 target 컬럼을 CSV에서 찾지 못했습니다.")

    dataframe = dataframe.dropna(subset=[target_column]).copy()
    if len(dataframe) < 10:
        raise RuntimeError("머신러닝 분석에는 최소 10행 이상의 데이터가 필요합니다.")

    y = dataframe[target_column]
    if y.nunique() < 2:
        raise RuntimeError("target 컬럼에는 최소 2개 이상의 클래스가 필요합니다.")

    is_numeric_target = pd.api.types.is_numeric_dtype(y)
    unique_ratio = y.nunique() / len(y)
    is_regression = is_numeric_target and (y.nunique() > 10 or unique_ratio > 0.2)

    feature_columns = [
        column
        for column in dataframe.columns
        if column != target_column and column.lower() != "id"
    ]
    if not feature_columns:
        raise RuntimeError("학습에 사용할 feature 컬럼이 없습니다.")

    X = dataframe[feature_columns].copy()
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    for column in numeric_columns:
        X[column] = X[column].fillna(X[column].median()).astype(float)

    for column in categorical_columns:
        X[column] = X[column].fillna("missing").astype(str)
        encoder = LabelEncoder()
        X[column] = encoder.fit_transform(X[column])

    stratify = None if is_regression else y if y.value_counts().min() >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=123,
        stratify=stratify,
    )

    if numeric_columns:
        scaler = StandardScaler()
        X_train.loc[:, numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
        X_val.loc[:, numeric_columns] = scaler.transform(X_val[numeric_columns])

    if is_regression:
        models = [
            ("LinearRegression", LinearRegression()),
            ("RandomForestRegressor", RandomForestRegressor(random_state=123)),
        ]
    else:
        models = [
            ("LogisticRegression", LogisticRegression(max_iter=1000)),
            ("RandomForestClassifier", RandomForestClassifier(random_state=123)),
        ]

    try:
        from xgboost import XGBClassifier, XGBRegressor

        if is_regression:
            xgb_model = XGBRegressor(
                n_estimators=80,
                max_depth=3,
                min_child_weight=1,
                random_state=123,
            )
            xgb_model.fit(X_train, y_train)
            xgb_train_pred = xgb_model.predict(X_train)
            xgb_val_pred = xgb_model.predict(X_val)
            xgb_row = {
                "name": "XGBRegressor",
                "train_score": round(r2_score(y_train, xgb_train_pred), 4),
                "valid_score": round(r2_score(y_val, xgb_val_pred), 4),
                "secondary_score": round(mean_squared_error(y_val, xgb_val_pred) ** 0.5, 4),
            }
        else:
            y_encoder = LabelEncoder()
            y_train_xgb = y_encoder.fit_transform(y_train)
            y_val_xgb = y_encoder.transform(y_val)
            xgb_model = XGBClassifier(
                n_estimators=80,
                max_depth=3,
                min_child_weight=1,
                random_state=123,
                eval_metric="mlogloss",
            )
            xgb_model.fit(X_train, y_train_xgb)
            xgb_train_pred = y_encoder.inverse_transform(xgb_model.predict(X_train))
            xgb_val_pred = y_encoder.inverse_transform(xgb_model.predict(X_val))
            xgb_row = {
                "name": "XGBClassifier",
                "train_score": round(f1_score(y_train, xgb_train_pred, average="macro"), 4),
                "valid_score": round(f1_score(y_val, xgb_val_pred, average="macro"), 4),
                "secondary_score": round(accuracy_score(y_val, xgb_val_pred), 4),
            }
    except Exception:
        xgb_row = None

    scores = []
    for name, model in models:
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        if is_regression:
            scores.append(
                {
                    "name": name,
                    "train_score": round(r2_score(y_train, train_pred), 4),
                    "valid_score": round(r2_score(y_val, val_pred), 4),
                    "secondary_score": round(mean_squared_error(y_val, val_pred) ** 0.5, 4),
                }
            )
        else:
            scores.append(
                {
                    "name": name,
                    "train_score": round(f1_score(y_train, train_pred, average="macro"), 4),
                    "valid_score": round(f1_score(y_val, val_pred, average="macro"), 4),
                    "secondary_score": round(accuracy_score(y_val, val_pred), 4),
                }
            )

    if xgb_row:
        scores.append(xgb_row)

    score_frame = pd.DataFrame(scores).sort_values("valid_score", ascending=False)
    best_model = score_frame.iloc[0].to_dict()
    row_count = len(dataframe)
    column_count = len(dataframe.columns)
    feature_preview = feature_columns[:8]
    remaining_feature_count = max(len(feature_columns) - len(feature_preview), 0)

    if is_regression:
        valid_score = best_model["valid_score"]
        if valid_score >= 0.7:
            result_summary = "검증 데이터에서도 target 값을 비교적 잘 설명하는 모델입니다."
        elif valid_score >= 0.3:
            result_summary = "일부 패턴은 잡았지만 실제 예측에는 추가 검증이 필요합니다."
        elif valid_score >= 0:
            result_summary = "검증 설명력이 낮아 모델 결과는 참고용으로 해석하는 것이 좋습니다."
        else:
            result_summary = "검증 R2가 음수라서 단순 평균 예측보다도 성능이 낮을 수 있습니다."
    else:
        valid_score = best_model["valid_score"]
        if valid_score >= 0.8:
            result_summary = "검증 데이터에서도 분류 성능이 높은 편입니다."
        elif valid_score >= 0.5:
            result_summary = "기본적인 분류 패턴은 잡았지만 오분류 가능성을 함께 봐야 합니다."
        else:
            result_summary = "검증 분류 성능이 낮아 feature 보강이나 target 재검토가 필요합니다."

    category_name = document.category.name if document.category else "미분류"
    created_at = document.created_at.strftime("%Y-%m-%d %H:%M") if document.created_at else "알 수 없음"
    feature_text = ", ".join(feature_preview)
    if remaining_feature_count:
        feature_text = f"{feature_text} 외 {remaining_feature_count}개"

    fig_score, ax_score = plt.subplots(figsize=(7.8, 4.5))
    score_frame.plot(x="name", y=["train_score", "valid_score"], kind="bar", ax=ax_score, color=["#4a6fe3", "#50c878"])
    primary_metric = "R2 score" if is_regression else "F1 macro"
    secondary_metric = "RMSE" if is_regression else "Accuracy"
    ax_score.set_title(f"{target_column} 모델별 {primary_metric}")
    ax_score.set_xlabel("모델")
    ax_score.set_ylabel(primary_metric)
    if not is_regression:
        ax_score.set_ylim(0, 1.05)
    ax_score.grid(axis="y", alpha=0.22)
    ax_score.tick_params(axis="x", rotation=15)
    score_chart = make_chart_uri(fig_score)
    plt.close(fig_score)

    return {
        "target_column": target_column,
        "task_type": "회귀" if is_regression else "분류",
        "primary_metric": primary_metric,
        "secondary_metric": secondary_metric,
        "row_count": row_count,
        "column_count": column_count,
        "train_count": len(X_train),
        "valid_count": len(X_val),
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "class_count": int(y.nunique()),
        "scores": score_frame.to_dict("records"),
        "best_model": best_model,
        "score_chart": score_chart,
        "explanation": {
            "source": f"{document.title} / 카테고리: {category_name} / 저장일: {created_at}",
            "dataset": f"총 {row_count}행, {column_count}열 CSV에서 target 결측치가 있는 행을 제외하고 분석했습니다.",
            "target": f"'{target_column}' 컬럼을 예측 대상으로 선택해 {'연속형 수치를 예측하는 회귀' if is_regression else '범주를 맞히는 분류'} 문제로 처리했습니다.",
            "features": f"예측에는 target과 id를 제외한 {len(feature_columns)}개 특성을 사용했습니다. 주요 특성: {feature_text}",
            "preprocess": f"수치형 {len(numeric_columns)}개는 결측치를 중앙값으로 채운 뒤 스케일링했고, 범주형 {len(categorical_columns)}개는 missing 처리 후 LabelEncoding했습니다.",
            "result": f"검증 {primary_metric} 기준 최고 모델은 {best_model['name']}이며 점수는 {best_model['valid_score']}입니다. {result_summary}",
        },
    }


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
        dashboard_context(
            user.username,
            documents,
            categories,
            pipeline_stats=build_pipeline_stats(documents, categories),
        ),
    )


@app.get("/dashboard")
def dashboard_page(request: Request, username: str, db: Session = Depends(get_db)):
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        dashboard_context(
            username,
            documents,
            categories,
            pipeline_stats=build_pipeline_stats(documents, categories),
        ),
    )


@app.get("/eda")
def eda_page(request: Request, username: str, db: Session = Depends(get_db)):
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()
    csv_documents = [
        document for document in documents
        if document.title.lower().endswith(".csv")
    ]

    return templates.TemplateResponse(
        request,
        "eda.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="eda",
            pipeline_stats=build_pipeline_stats(documents, categories),
            csv_documents=csv_documents,
        ),
    )


@app.get("/preprocess")
def preprocess_page(request: Request, username: str, db: Session = Depends(get_db)):
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()
    csv_documents = [
        document for document in documents
        if document.title.lower().endswith(".csv")
    ]

    return templates.TemplateResponse(
        request,
        "preprocess.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="preprocess",
            pipeline_stats=build_pipeline_stats(documents, categories),
            csv_documents=csv_documents,
        ),
    )


@app.get("/preprocess/{document_id}")
def preprocess_detail_page(
    request: Request,
    document_id: int,
    username: str,
    db: Session = Depends(get_db),
):
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()
    csv_documents = [
        document for document in documents
        if document.title.lower().endswith(".csv")
    ]
    selected_document = (
        db.query(Document)
        .filter(Document.id == document_id)
        .first()
    )
    error = None
    preprocess_result = {"charts": {}, "csv_profiles": [], "preprocess_summary": None}

    if not selected_document or not selected_document.title.lower().endswith(".csv"):
        error = "전처리할 CSV 파일을 찾지 못했습니다."
    else:
        try:
            preprocess_result = build_preprocess_charts([selected_document])
        except Exception as exc:
            error = str(exc)

    return templates.TemplateResponse(
        request,
        "preprocess.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="preprocess",
            error=error,
            pipeline_stats=build_pipeline_stats(documents, categories),
            eda_charts=preprocess_result["charts"],
            csv_profiles=preprocess_result["csv_profiles"],
            csv_documents=csv_documents,
            selected_document=selected_document,
            preprocess_summary=preprocess_result["preprocess_summary"],
        ),
    )


@app.get("/eda/{document_id}")
def analysis_select_page(
    request: Request,
    document_id: int,
    username: str,
    db: Session = Depends(get_db),
):
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()
    csv_documents = [
        document for document in documents
        if document.title.lower().endswith(".csv")
    ]
    selected_document = (
        db.query(Document)
        .filter(Document.id == document_id)
        .first()
    )
    error = None

    if not selected_document or not selected_document.title.lower().endswith(".csv"):
        error = "분석할 CSV 파일을 찾지 못했습니다."

    return templates.TemplateResponse(
        request,
        "analysis_select.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="eda",
            error=error,
            pipeline_stats=build_pipeline_stats(documents, categories),
            csv_documents=csv_documents,
            selected_document=selected_document,
        ),
    )


@app.get("/eda/{document_id}/charts")
def eda_detail_page(
    request: Request,
    document_id: int,
    username: str,
    db: Session = Depends(get_db),
):
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()
    csv_documents = [
        document for document in documents
        if document.title.lower().endswith(".csv")
    ]
    selected_document = (
        db.query(Document)
        .filter(Document.id == document_id)
        .first()
    )
    error = None
    eda_result = {"charts": {}, "csv_profiles": []}

    if not selected_document or not selected_document.title.lower().endswith(".csv"):
        error = "분석할 CSV 파일을 찾지 못했습니다."
    else:
        try:
            eda_result = build_eda_charts([selected_document], include_storage_charts=False)
        except Exception as exc:
            error = str(exc)

    return templates.TemplateResponse(
        request,
        "eda.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="eda",
            error=error,
            pipeline_stats=build_pipeline_stats(documents, categories),
            eda_charts=eda_result["charts"],
            csv_profiles=eda_result["csv_profiles"],
            csv_documents=csv_documents,
            selected_document=selected_document,
        ),
    )


@app.get("/documents/new")
def new_document_page(request: Request, username: str, db: Session = Depends(get_db)):
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()

    return templates.TemplateResponse(
        request,
        "document_create.html",
        dashboard_context(username, documents, categories, active_view="create"),
    )


@app.get("/documents/search-page")
def search_document_page(request: Request, username: str, db: Session = Depends(get_db)):
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()

    return templates.TemplateResponse(
        request,
        "document_search.html",
        dashboard_context(username, documents, categories, active_view="search"),
    )


@app.get("/rag")
def rag_page(request: Request, username: str, db: Session = Depends(get_db)):
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    categories = db.query(Category).order_by(Category.name).all()

    return templates.TemplateResponse(
        request,
        "rag.html",
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
        "document_list.html",
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
    error = None
    saved_count = 0
    documents_to_save = []
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
        documents_to_save.append(direct_document)

    uploaded_files = [
        upload_file
        for upload_file in (attachments or [])
        if upload_file.filename
    ]

    for upload_file in uploaded_files:
        try:
            extracted_content = await extract_upload_text(upload_file)
        except RuntimeError as exc:
            error = str(exc)
            continue

        file_document = Document(
            title=upload_file.filename or "업로드 문서",
            content=extracted_content,
            category_id=category_id,
        )
        db.add(file_document)
        documents_to_save.append(file_document)

    if not documents_to_save:
        documents = db.query(Document).order_by(Document.created_at.desc()).all()
        return templates.TemplateResponse(
            request,
            "document_create.html",
            dashboard_context(
                username,
                documents,
            categories,
            active_view="create",
            rag_error=error or "직접 작성 내용 또는 업로드 파일이 필요합니다.",
        ),
    )

    db.commit()

    for document in documents_to_save:
        db.refresh(document)
        try:
            upsert_document(document)
            saved_count += 1
        except RuntimeError as exc:
            error = str(exc)

    documents = db.query(Document).order_by(Document.created_at.desc()).all()

    return templates.TemplateResponse(
        request,
        "document_create.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="create",
            rag_error=error,
            message=f"문서 {saved_count}개가 저장되었습니다.",
        ),
    )


@app.post("/documents/kaggle")
def create_kaggle_document(
    request: Request,
    username: str = Form(...),
    dataset_id: str = Form(...),
    category_id: int = Form(None),
    db: Session = Depends(get_db),
):
    categories = db.query(Category).order_by(Category.name).all()
    cleaned_dataset_id = dataset_id.strip()

    try:
        kaggle_result = download_and_preprocess_dataset(cleaned_dataset_id)
        document = Document(
            title=kaggle_result["document_title"],
            content=kaggle_result["processed_csv"],
            category_id=category_id,
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        upsert_document(document)
        message = (
            f"{cleaned_dataset_id} Kaggle 데이터셋을 수집하고 전처리한 CSV를 저장했습니다. "
            f"metadata: {kaggle_result['metadata_path']}"
        )
        error = None
    except Exception as exc:
        db.rollback()
        message = None
        error = str(exc)

    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    return templates.TemplateResponse(
        request,
        "document_create.html",
        dashboard_context(
            username,
            documents,
            categories,
            active_view="create",
            rag_error=error,
            message=message,
            kaggle_dataset_id=cleaned_dataset_id,
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
        "document_search.html",
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
        document_ids = None
        search_documents = documents
        if category_id:
            search_documents = (
                db.query(Document)
                .filter(Document.category_id == category_id)
                .order_by(Document.created_at.desc())
                .all()
            )
            document_ids = [document.id for document in search_documents]

        title_matched_ids = find_title_matched_document_ids(question, search_documents)
        if title_matched_ids:
            document_ids = title_matched_ids
            for document in search_documents:
                if document.id in title_matched_ids:
                    upsert_document(document)

        rag_result = ask_rag(question, document_ids=document_ids)
        rag_answer = rag_result["answer"]
        rag_sources = rag_result["sources"]
    except RuntimeError as exc:
        rag_error = str(exc)

    return templates.TemplateResponse(
        request,
        "rag.html",
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


@app.get("/documents/{document_id}")
def document_viewer(
    request: Request,
    document_id: int,
    username: str,
    highlight: str = "",
    db: Session = Depends(get_db),
):
    import json as _json
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        return RedirectResponse(url=f"/documents/list?username={username}", status_code=302)

    try:
        highlight_list = _json.loads(highlight) if highlight else []
    except Exception:
        highlight_list = []

    return templates.TemplateResponse(
        request,
        "viewer.html",
        {
            "username": username,
            "document": document,
            "highlight": highlight_list,
        },
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
            "categories.html",
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
        "categories.html",
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
        "categories.html",
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
        "categories.html",
        dashboard_context(username, documents, categories, active_view="categories"),
    )
