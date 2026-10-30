"""DTHU text/audio search API. Run: uvicorn main:app --reload"""
import os
import re
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import bm25s
import pandas as pd
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import pipeline
from underthesea import word_tokenize

CSV_PATH = Path(os.getenv('CSV_PATH', 'dthu_all.csv'))
MODEL_NAME = os.getenv('ASR_MODEL', 'vinai/PhoWhisper-small')
MIN_ID = int(os.getenv('MIN_ID', '800'))
MAX_AUDIO_BYTES = 20 * 1024 * 1024
SUPPORTED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.webm', '.ogg', '.flac', '.mp4'}

DIGITS = {'không': 0, 'một': 1, 'mốt': 1, 'hai': 2, 'ba': 3,
          'bốn': 4, 'tư': 4, 'năm': 5, 'lăm': 5, 'sáu': 6,
          'bảy': 7, 'tám': 8, 'chín': 9}
UNITS = '|'.join(sorted(DIGITS, key=len, reverse=True))
YEAR_WORDS = re.compile(
    rf'\bnăm\s+((?:hai\s+nghìn(?:\s+không\s+trăm)?(?:\s+hai\s+mươi)?(?:\s+(?:{UNITS}))?'
    rf'|hai\s+không\s+hai\s+(?:{UNITS})|20\d{{2}}|\d{{2}}))\b', re.I)


def parse_year(value: str):
    value = re.sub(r'\s+', ' ', value.lower()).strip()
    if re.fullmatch(r'20\d{2}', value):
        return value
    if re.fullmatch(r'\d{2}', value):
        return '20' + value
    parts = value.split()
    if len(parts) == 4 and all(x in DIGITS for x in parts):
        year = ''.join(str(DIGITS[x]) for x in parts)
        return year if year.startswith('20') else None
    if value.startswith('hai nghìn'):
        rest = value[len('hai nghìn'):].strip()
        if rest.startswith('không trăm'):
            rest = rest[len('không trăm'):].strip()
        tens = 0
        if rest.startswith('hai mươi'):
            tens = 20
            rest = rest[len('hai mươi'):].strip()
        if not rest or rest in DIGITS:
            return str(2000 + tens + (DIGITS[rest] if rest else 0))
    return None


def normalize_year(query: str) -> str:
    return YEAR_WORDS.sub(lambda m: 'năm ' + (parse_year(m.group(1)) or m.group(1)), query.lower())


def tokenizer(value: str) -> list[str]:
    return word_tokenize(str(value).lower(), format='text').split()


class TextRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)


class SearchResult(BaseModel):
    id: int
    title: str
    url: str
    score: float


class SearchResponse(BaseModel):
    query: str
    normalized_query: str
    tokens: list[str]
    total: int
    results: list[SearchResult]
    transcription: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not CSV_PATH.is_file():
        raise RuntimeError(f'CSV not found: {CSV_PATH.resolve()}')
    df = pd.read_csv(CSV_PATH)
    for column in ('id', 'title', 'url'):
        if column not in df.columns:
            raise RuntimeError(f'Missing CSV column: {column}')
    df['id'] = pd.to_numeric(df['id'], errors='coerce')
    df = df.dropna(subset=['id', 'title', 'url']).copy()
    df = df[df['id'] > MIN_ID].reset_index(drop=True)
    if df.empty:
        raise RuntimeError('No searchable rows after filtering')
    corpus = [tokenizer(title) for title in df['title']]
    retriever = bm25s.BM25()
    retriever.index(corpus)
    device = 0 if torch.cuda.is_available() else -1
    asr = pipeline('automatic-speech-recognition', model=MODEL_NAME, device=device)
    app.state.df = df
    app.state.retriever = retriever
    app.state.asr = asr
    app.state.asr_lock = Lock()
    yield


app = FastAPI(title='DTHU Search API', version='1.0.0', lifespan=lifespan)


def search(query: str, top_k: int, transcription: str | None = None) -> SearchResponse:
    normalized = normalize_year(query.strip())
    tokens = tokenizer(normalized)
    items = []
    if tokens:
        df = app.state.df
        indexes, scores = app.state.retriever.retrieve([tokens], k=min(top_k, len(df)))
        for idx, score in zip(indexes[0], scores[0]):
            row = df.iloc[int(idx)]
            items.append(SearchResult(id=int(row['id']), title=str(row['title']),
                                      url=str(row['url']), score=float(score)))
        # Preserve original notebook behavior: sort by latest document ID.
        items.sort(key=lambda x: x.id, reverse=True)
    return SearchResponse(query=query, normalized_query=normalized, tokens=tokens,
                          total=len(items), results=items, transcription=transcription)


@app.get('/api/health')
def health():
    return {'status': 'ok', 'documents': len(app.state.df), 'asr_model': MODEL_NAME}


@app.post('/api/search/text', response_model=SearchResponse)
def text_search(payload: TextRequest):
    return search(payload.query, payload.top_k)


@app.post('/api/search/audio', response_model=SearchResponse)
async def audio_search(file: UploadFile = File(...), top_k: int = Query(5, ge=1, le=100)):
    extension = Path(file.filename or '').suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail='Unsupported audio extension')
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp:
            temp_path = temp.name
            total_bytes = 0
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_AUDIO_BYTES:
                    raise HTTPException(status_code=413, detail='Audio exceeds 20 MB')
                temp.write(chunk)
        if total_bytes == 0:
            raise HTTPException(status_code=400, detail='Empty audio file')
        try:
            with app.state.asr_lock:
                transcription = app.state.asr(temp_path)['text'].strip()
        except Exception as exc:
            raise HTTPException(status_code=422, detail='Cannot decode/transcribe audio') from exc
        return search(transcription, top_k, transcription=transcription)
    finally:
        await file.close()
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


# Serve the search interface from the same origin as the API.
app.mount('/static', StaticFiles(directory=Path(__file__).parent / 'static'), name='static')

@app.get('/', include_in_schema=False)
def home():
    return FileResponse(Path(__file__).parent / 'static' / 'index.html')
