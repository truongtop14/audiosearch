import os, re, tempfile, threading
from pathlib import Path
from contextlib import asynccontextmanager
import bm25s, pandas as pd, torch
from transformers import pipeline
from underthesea import word_tokenize
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

BASE=Path('/content')
CSV_PATH=BASE/'dthu_all.csv'
STATIC=BASE/'static'
MODEL_NAME=os.getenv('ASR_MODEL','vinai/PhoWhisper-small')
MIN_ID=int(os.getenv('MIN_ID','800'))
MAX_AUDIO_BYTES=20*1024*1024
EXTENSIONS={'.wav','.mp3','.m4a','.webm','.ogg','.flac','.mp4'}
DIGITS={'không':0,'một':1,'mốt':1,'hai':2,'ba':3,'bốn':4,'tư':4,'năm':5,'lăm':5,'sáu':6,'bảy':7,'tám':8,'chín':9}
UNITS='|'.join(sorted(DIGITS,key=len,reverse=True))
YEAR_WORDS=re.compile(rf'\bnăm\s+((?:hai\s+nghìn(?:\s+không\s+trăm)?(?:\s+hai\s+mươi)?(?:\s+(?:{UNITS}))?|hai\s+không\s+hai\s+(?:{UNITS})|20\d{{2}}|\d{{2}}))\b',re.I)
def parse_year(value):
    value=re.sub(r'\s+',' ',value.lower()).strip()
    if re.fullmatch(r'20\d{2}',value): return value
    if re.fullmatch(r'\d{2}',value): return '20'+value
    parts=value.split()
    if len(parts)==4 and all(x in DIGITS for x in parts):
        year=''.join(str(DIGITS[x]) for x in parts)
        return year if year.startswith('20') else None
    if value.startswith('hai nghìn'):
        rest=value[len('hai nghìn'):].strip()
        if rest.startswith('không trăm'): rest=rest[len('không trăm'):].strip()
        tens=0
        if rest.startswith('hai mươi'):
            tens=20
            rest=rest[len('hai mươi'):].strip()
        if not rest or rest in DIGITS:
            return str(2000+tens+(DIGITS[rest] if rest else 0))
    return None
def normalize_year(query):
    return YEAR_WORDS.sub(lambda m:'năm '+(parse_year(m.group(1)) or m.group(1)),query.lower())
def tokenizer(text): return word_tokenize(str(text).lower(),format='text').split()
class TextRequest(BaseModel):
    query:str=Field(min_length=1)
    top_k:int=Field(default=5,ge=1,le=100)
class SearchResult(BaseModel):
    id:int
    title:str
    url:str
    score:float
class SearchResponse(BaseModel):
    query:str
    normalized_query:str
    tokens:list[str]
    total:int
    results:list[SearchResult]
    transcription:str|None=None
@asynccontextmanager
async def lifespan(app):
    df=pd.read_csv(CSV_PATH,encoding='utf-8-sig')
    for col in ('id','title','url'):
        if col not in df.columns: raise RuntimeError('Missing CSV column: '+col)
    df['id']=pd.to_numeric(df['id'],errors='coerce')
    df=df.dropna(subset=['id','title','url'])
    df=df[df.id>MIN_ID].reset_index(drop=True)
    if df.empty: raise RuntimeError('No searchable rows')
    retriever=bm25s.BM25()
    retriever.index([tokenizer(title) for title in df.title])
    device=0 if torch.cuda.is_available() else -1
    asr=pipeline('automatic-speech-recognition',model=MODEL_NAME,device=device)
    app.state.df=df;app.state.retriever=retriever;app.state.asr=asr
    app.state.asr_lock=threading.Lock()
    print('Ready:',len(df),'documents; ASR:',MODEL_NAME,'GPU:',device==0)
    yield
app=FastAPI(title='DTHU Search API',lifespan=lifespan)
app.mount('/static',StaticFiles(directory=str(STATIC)),name='static')
@app.get('/',include_in_schema=False)
def home(): return FileResponse(STATIC/'index.html')
@app.get('/api/health')
def health(): return {'status':'ok','documents':len(app.state.df),'asr_model':MODEL_NAME}
def search(query,top_k,transcription=None):
    normalized=normalize_year(query.strip())
    tokens=tokenizer(normalized)
    items=[]
    if tokens:
        indexes,scores=app.state.retriever.retrieve([tokens],k=min(top_k,len(app.state.df)))
        for idx,score in zip(indexes[0],scores[0]):
            row=app.state.df.iloc[int(idx)]
            items.append(SearchResult(id=int(row['id']),title=str(row['title']),url=str(row['url']),score=float(score)))
        items.sort(key=lambda x:x.id,reverse=True)
    return SearchResponse(query=query,normalized_query=normalized,tokens=tokens,total=len(items),results=items,transcription=transcription)
@app.post('/api/search/text',response_model=SearchResponse)
def text_search(payload:TextRequest): return search(payload.query,payload.top_k)
@app.post('/api/search/audio',response_model=SearchResponse)
async def audio_search(file:UploadFile=File(...),top_k:int=Query(5,ge=1,le=100)):
    ext=Path(file.filename or '').suffix.lower()
    if ext not in EXTENSIONS: raise HTTPException(415,'Unsupported audio extension')
    path=None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext,delete=False) as tmp:
            path=tmp.name;total=0
            while chunk:=await file.read(1024*1024):
                total+=len(chunk)
                if total>MAX_AUDIO_BYTES: raise HTTPException(413,'Audio exceeds 20 MB')
                tmp.write(chunk)
        if total==0: raise HTTPException(400,'Empty audio file')
        def transcribe():
            with app.state.asr_lock: return app.state.asr(path)['text'].strip()
        try: transcription=await run_in_threadpool(transcribe)
        except Exception as exc: raise HTTPException(422,'Cannot decode/transcribe audio') from exc
        return search(transcription,top_k,transcription)
    finally:
        await file.close()
        if path and os.path.exists(path): os.unlink(path)
