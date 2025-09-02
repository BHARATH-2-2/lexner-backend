"""
LEXner FastAPI Backend with BiLSTM+CRF NER + Neo4j Graph (Multi-user, DB-backed)

Run:
  pip install fastapi uvicorn pydantic "python-multipart" spacy PyMuPDF PyPDF2 torch torchcrf neo4j sqlalchemy passlib[bcrypt] python-jose[cryptography] boto3 python-dotenv
  uvicorn server:app --reload --port 8000
"""
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import os, re, json, time, uuid, logging
from typing import List, Dict, Any, Optional, Union

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Body, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import torch
import torch.nn as nn
from torchcrf import CRF

from neo4j import GraphDatabase

from sqlalchemy import create_engine, Column, String, Text, JSON, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from datetime import datetime, timedelta

from passlib.context import CryptContext
from jose import jwt, JWTError
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

# ---------------- Logging ---------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lexner")

# ---------------- Auth / JWT ---------------- #
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

JWT_SECRET = os.getenv("JWT_SECRET", "dev_secret")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
JWT_EXPIRE_MIN = int(os.getenv("JWT_EXPIRE_MIN", "60"))

def create_token(sub: str) -> str:
    payload = {"sub": sub, "exp": datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MIN)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def require_user(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        sub = payload.get("sub")
        if not sub:
            raise HTTPException(status_code=401, detail="Invalid token subject")
        return sub
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ---------------- Database ---------------- #
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserORM(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class DocumentORM(Base):
    __tablename__ = "documents"
    source_id = Column(String, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)
    user_id = Column(String, index=True, nullable=False)
    raw_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    meta = Column(JSON, default={})
    entities = Column(JSON, default={})
    summary = Column(Text, default="")

# ---------- Model Config ---------- #
EMBEDDING_DIM = 200
HIDDEN_DIM = 256
DROPOUT = 0.5

# ---------------- Neo4j Setup ---------------- #
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "change_me")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def add_case_to_graph(session_id: str, case_title: str, entities: dict, user_id: str):
    """Write nodes/edges to Neo4j, scoped by user_id."""
    try:
        with driver.session() as session:
            session.run(
                "MERGE (c:Case {title: $title, session_id: $session_id, user_id: $user_id})",
                title=case_title, session_id=session_id, user_id=user_id
            )
            for judge in entities.get("judges", []):
                session.run(
                    """
                    MERGE (j:Judge {name: $judge, user_id: $user_id})
                    MERGE (c:Case {session_id: $session_id, user_id: $user_id})
                    MERGE (j)-[:PART_OF]->(c)
                    """,
                    judge=judge, session_id=session_id, user_id=user_id
                )
            for act in entities.get("acts_sections", []):
                session.run(
                    """
                    MERGE (a:Act {name: $act, user_id: $user_id})
                    MERGE (c:Case {session_id: $session_id, user_id: $user_id})
                    MERGE (c)-[:REFERENCES]->(a)
                    """,
                    act=act, session_id=session_id, user_id=user_id
                )
            for citation in entities.get("citations", []):
                session.run(
                    """
                    MERGE (cit:Citation {name: $citation, user_id: $user_id})
                    MERGE (c:Case {session_id: $session_id, user_id: $user_id})
                    MERGE (c)-[:CITES]->(cit)
                    """,
                    citation=citation, session_id=session_id, user_id=user_id
                )
    except Exception as e:
        logger.warning(f"Neo4j write failed: {e}")

# ---------------- S3 Export Helper ---------------- #
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX", "lexner/exports")

def upload_json_to_s3(key: str, data: dict) -> Optional[str]:
    """Upload JSON to S3 (if configured). Returns https URL or None."""
    if not S3_BUCKET:
        return None
    try:
        import boto3
        s3 = boto3.client("s3")
        payload = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=payload, ContentType="application/json")
        region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        return f"https://{S3_BUCKET}.s3.{region}.amazonaws.com/{key}"
    except Exception as e:
        logger.warning(f"S3 upload failed: {e}")
        return None

# ---------------- Response Models ---------------- #
class CaseDataResponse(BaseModel):
    judges: List[str]
    citations: List[str]
    acts_sections: List[str]
    summary: str

class JudgesResponse(BaseModel):
    judges: List[str]

class CitationsResponse(BaseModel):
    citations: List[str]

class ActsSectionsResponse(BaseModel):
    acts_sections: List[str]

class SummaryResponse(BaseModel):
    summary: str

class ExportResponse(BaseModel):
    message: str
    file_path: str

class PasteTextIn(BaseModel):
    session_id: str
    text: str
    meta: Optional[Dict[str, Any]] = None

# ---------------- NLP + Text Extraction ---------------- #
def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\u00A0", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[\t\f\v]", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    text = ""
    try:
        import fitz  # PyMuPDF
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            parts = [page.get_text() for page in doc]
        text = "\n".join(parts)
    except Exception:
        try:
            import PyPDF2
            from io import BytesIO
            reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            parts = [page.extract_text() or "" for page in reader.pages]
            text = "\n".join(parts)
        except Exception as e:
            raise RuntimeError(f"PDF text extraction failed: {e}")
    return normalize_text(text)

# ---------------- BiLSTM + CRF Model ---------------- #
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=EMBEDDING_DIM,
                 hidden_dim=HIDDEN_DIM, dropout=DROPOUT, embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embeddings is not None:
            self.embedding.weight.data.copy_(torch.tensor(embeddings))
            self.embedding.weight.requires_grad = False
        self.dropout_emb = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True
        )
        self.dropout_lstm = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.crf = CRF(tagset_size, batch_first=True)

    def forward(self, x, tags=None, mask=None):
        embeds = self.dropout_emb(self.embedding(x))
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout_lstm(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        if tags is not None:
            return -self.crf(emissions, tags, mask=mask, reduction='mean')
        else:
            return self.crf.decode(emissions, mask=mask)

# ---------------- Load Trained Vocab/Tags/Model ---------------- #
VOCAB_PATH = "vocab.json"
TAG_PATH = "tags.json"
MODEL_PATH = "legal_bilstm_crf.pth"

if os.path.exists(VOCAB_PATH):
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        word2idx = json.load(f)
else:
    raise FileNotFoundError(f"Vocab file not found at {VOCAB_PATH}")

idx2word = {v: k for k, v in word2idx.items()}

if os.path.exists(TAG_PATH):
    with open(TAG_PATH, "r", encoding="utf-8") as f:
        tag2idx = json.load(f)
else:
    raise FileNotFoundError(f"Tag file not found at {TAG_PATH}")

idx2tag = {v: k for k, v in tag2idx.items()}
vocab_size = len(word2idx)
tagset_size = len(tag2idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BiLSTM_CRF(vocab_size, tagset_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# ---------------- Tokenization & Encoding ---------------- #
def tokenize(text: str):
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

def encode_tokens(tokens: List[str]) -> List[int]:
    unk_idx = word2idx.get("<UNK>", 1)
    return [word2idx.get(tok.lower(), unk_idx) for tok in tokens]

# ---------------- NER Prediction ---------------- #
def predict_entities(text: str) -> Dict[str, List[str]]:
    tokens = tokenize(text)
    if not tokens:
        return {"judges": [], "citations": [], "acts_sections": []}
    input_ids = torch.tensor([encode_tokens(tokens)], dtype=torch.long).to(device)
    mask = torch.ones_like(input_ids, dtype=torch.bool).to(device)
    with torch.no_grad():
        try:
            predicted_tags = model(input_ids, mask=mask)
            if isinstance(predicted_tags, list) and isinstance(predicted_tags[0], list):
                predicted_tags = predicted_tags[0]
        except Exception as e:
            logging.error(f"[NER ERROR] {e}")
            return {"judges": [], "citations": [], "acts_sections": []}
    entities = {"judges": [], "citations": [], "acts_sections": []}
    current_entity, current_label = [], None
    for tok, tag_idx in zip(tokens, predicted_tags):
        tag = idx2tag.get(tag_idx, "O")
        if tag == "O":
            if current_entity:
                text_ent = " ".join(current_entity)
                if current_label == "JUDGE": entities["judges"].append(text_ent)
                elif current_label == "CITATION": entities["citations"].append(text_ent)
                elif current_label == "ACT": entities["acts_sections"].append(text_ent)
                current_entity, current_label = [], None
        else:
            label_type = tag.split("-")[-1] if "-" in tag else tag
            if tag.startswith("B-") or current_label != label_type:
                if current_entity:
                    text_ent = " ".join(current_entity)
                    if current_label == "JUDGE": entities["judges"].append(text_ent)
                    elif current_label == "CITATION": entities["citations"].append(text_ent)
                    elif current_label == "ACT": entities["acts_sections"].append(text_ent)
                current_entity, current_label = [tok], label_type
            else:
                current_entity.append(tok)
    if current_entity:
        text_ent = " ".join(current_entity)
        if current_label == "JUDGE": entities["judges"].append(text_ent)
        elif current_label == "CITATION": entities["citations"].append(text_ent)
        elif current_label == "ACT": entities["acts_sections"].append(text_ent)
    return entities

# ---------------- Entity Cleaning ---------------- #
def clean_entities(entities: dict) -> dict:
    cleaned = {"judges": [], "citations": [], "acts_sections": []}
    noise_words = {"will", "because", "appears", "prescribes", "thereof",
                   "such", "may", "shall", "must", "upon", "hereby"}
    for key, values in entities.items():
        seen = set()
        for v in values:
            v_norm = (v or "").strip().lower().title()
            if len(v_norm) < 3:
                continue
            if any(w in v_norm.lower().split() for w in noise_words):
                continue
            if v_norm not in seen:
                seen.add(v_norm)
                cleaned[key].append(v_norm)
    return cleaned

# ---------------- Summarization ---------------- #
STOPWORDS = set(
    "the a an is are was were be been being have has had do does did of in on at to from by with "
    "for and or if as that this those these there here it its into over under than then so such "
    "their his her our your my we you they i not no nor can could should would may might will shall "
    "also while where when who whom which what".split()
)
def summarize(text: str, max_sentences: int = 5) -> str:
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    if not sents:
        return ""
    freqs: Dict[str, int] = {}
    for s in sents:
        for tok in re.findall(r"[A-Za-z][A-Za-z-]+", s):
            if tok.lower() not in STOPWORDS:
                freqs[tok.lower()] = freqs.get(tok.lower(), 0) + 1
    scored = [(sum(freqs.get(tok.lower(), 0) for tok in re.findall(r"[A-Za-z][A-Za-z-]+", s)), i, s)
              for i, s in enumerate(sents)]
    top = sorted(sorted(scored, key=lambda x: -x[0])[:max_sentences], key=lambda x: x[1])
    return " ".join(s for _, _, s in top).strip()

# ---------------- Document Store ---------------- #
class DocumentRecord(BaseModel):
    session_id: str
    source_id: str
    raw_text: str
    created_at: float
    meta: Dict[str, Any] = {}
    entities: Dict[str, Any] = {}
    summary: str = ""

class Store:
    def upsert(
        self,
        session_id: str,
        user_id: str,
        text: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[DocumentRecord]:
        text = (text or "").strip()
        if not text:
            return None
        entities = clean_entities(predict_entities(text))
        summary_text = summarize(text)
        rec = DocumentRecord(
            session_id=session_id,
            source_id=str(uuid.uuid4()),
            raw_text=text,
            created_at=time.time(),
            meta=meta or {},
            entities=entities,
            summary=summary_text,
        )
        db: Session = SessionLocal()
        try:
            db.add(
                DocumentORM(
                    source_id=rec.source_id,
                    session_id=session_id,
                    user_id=user_id,
                    raw_text=rec.raw_text,
                    created_at=datetime.utcfromtimestamp(rec.created_at),
                    meta=rec.meta,
                    entities=rec.entities,
                    summary=rec.summary,
                )
            )
            db.commit()
        finally:
            db.close()
        # Push entities to Neo4j (user-scoped)
        add_case_to_graph(session_id, f"Case_{session_id}", entities, user_id)
        return rec

    def get_all_records(self, session_id: str, user_id: str) -> List[DocumentRecord]:
        db: Session = SessionLocal()
        try:
            rows = (
                db.query(DocumentORM)
                .filter(DocumentORM.session_id == session_id, DocumentORM.user_id == user_id)
                .order_by(DocumentORM.created_at.asc())
                .all()
            )
            return [
                DocumentRecord(
                    session_id=r.session_id,
                    source_id=r.source_id,
                    raw_text=r.raw_text,
                    created_at=r.created_at.timestamp(),
                    meta=r.meta or {},
                    entities=r.entities or {},
                    summary=r.summary or "",
                )
                for r in rows
            ]
        finally:
            db.close()

    def get_merged_record(self, session_id: str, user_id: str) -> Optional[DocumentRecord]:
        records = self.get_all_records(session_id, user_id)
        if not records:
            return None
        merged_text = " ".join(r.raw_text for r in records)
        merged_entities = {"judges": [], "citations": [], "acts_sections": []}
        for r in records:
            r_clean = clean_entities(r.entities)
            for k in merged_entities.keys():
                merged_entities[k].extend(r_clean.get(k, []))
        merged_entities = clean_entities(merged_entities)
        merged_summary = summarize(merged_text)
        return DocumentRecord(
            session_id=session_id,
            source_id="merged",
            raw_text=merged_text,
            created_at=time.time(),
            meta={},
            entities=merged_entities,
            summary=merged_summary,
        )

STORE = Store()

# ---------------- Startup ---------------- #
app = FastAPI(title="LEXner Legal NER Backend (BiLSTM+CRF + Neo4j)", version="5.0")

@app.on_event("startup")
async def startup_event():
    Base.metadata.create_all(bind=engine)
    try:
        with driver.session() as session:
            session.run("RETURN 1")
    except Exception as e:
        logger.warning(f"Neo4j not reachable at startup: {e}")

# ---------------- Health ---------------- #
@app.get("/ping")
async def health_check():
    try:
        _ = next(model.parameters())
        model_ok = True
    except Exception:
        model_ok = False
    try:
        with driver.session() as session:
            session.run("RETURN 1")
        neo4j_ok = True
    except Exception:
        neo4j_ok = False
    try:
        db: Session = SessionLocal()
        db.execute("SELECT 1")
        db_ok = True
    except Exception:
        db_ok = False
    finally:
        try:
            db.close()
        except Exception:
            pass
    return {"status": "ok", "model": model_ok, "neo4j": neo4j_ok, "db": db_ok, "device": str(device)}

# ---------------- Endpoints (User-Scoped) ---------------- #
@app.post("/upload")
async def upload_document(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    user_id: str = Depends(require_user),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")
    text = extract_text_from_pdf(await file.read())
    STORE.upsert(session_id, user_id, text, meta={"filename": file.filename})
    return {"ok": True, "session_id": session_id, "chars": len(text)}

@app.post("/paste_text")
async def paste_text(
    payload: Union[PasteTextIn, List[PasteTextIn]] = Body(...),
    user_id: str = Depends(require_user),
):
    if isinstance(payload, dict):
        payload = [PasteTextIn(**payload)]
    elif isinstance(payload, list):
        payload = [PasteTextIn(**p) if isinstance(p, dict) else p for p in payload]
    results = []
    for case in payload:
        text = normalize_text(case.text)
        STORE.upsert(case.session_id, user_id, text, meta=case.meta)
        results.append({"session_id": case.session_id, "chars": len(text)})
    return {"ok": True, "results": results}

@app.get("/full_case_data")
async def full_case_data(session_id: str, user_id: str = Depends(require_user)):
    record = STORE.get_merged_record(session_id, user_id)
    if not record:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "case_title": f"Case_{session_id}",
        "judges": record.entities.get("judges", []),
        "citations": record.entities.get("citations", []),
        "acts_sections": record.entities.get("acts_sections", []),
        "summary": record.summary,
    }

@app.get("/ask_about_judges", response_model=JudgesResponse)
async def ask_about_judges(session_id: str, user_id: str = Depends(require_user)):
    record = STORE.get_merged_record(session_id, user_id)
    return JudgesResponse(judges=record.entities.get("judges", ["Not found"]) if record else ["No text found"])

@app.get("/ask_about_citations", response_model=CitationsResponse)
async def ask_about_citations(session_id: str, user_id: str = Depends(require_user)):
    record = STORE.get_merged_record(session_id, user_id)
    return CitationsResponse(citations=record.entities.get("citations", ["Not found"]) if record else ["No text found"])

@app.get("/ask_about_acts_sections", response_model=ActsSectionsResponse)
async def ask_about_acts_sections(session_id: str, user_id: str = Depends(require_user)):
    record = STORE.get_merged_record(session_id, user_id)
    return ActsSectionsResponse(acts_sections=record.entities.get("acts_sections", ["Not found"]) if record else ["No text found"])

@app.get("/summarization", response_model=SummaryResponse)
async def summarization_endpoint(session_id: str, user_id: str = Depends(require_user)):
    record = STORE.get_merged_record(session_id, user_id)
    return SummaryResponse(summary=record.summary or "No text found")

@app.get("/export_case", response_model=ExportResponse)
async def export_case(session_id: str, user_id: str = Depends(require_user)):
    record = STORE.get_merged_record(session_id, user_id)
    if not record:
        return ExportResponse(message="Failed", file_path="")
    data = {
        "case_title": f"Case_{session_id}",
        "judges": record.entities.get("judges", []),
        "citations": record.entities.get("citations", []),
        "acts_sections": record.entities.get("acts_sections", []),
        "summary": record.summary,
    }
    key = f"{S3_PREFIX}/case_{session_id}_{int(time.time())}.json"
    url = upload_json_to_s3(key, data)
    if url:
        return ExportResponse(message="Export successful", file_path=url)
    # Local fallback
    file_path = f"case_{session_id}_{int(time.time())}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return ExportResponse(message="Export saved locally", file_path=file_path)

# ---------------- Auth Endpoints ---------------- #
@app.post("/auth/signup")
async def signup(username: str = Form(...), password: str = Form(...)):
    db: Session = SessionLocal()
    try:
        if db.query(UserORM).filter_by(username=username).first():
            raise HTTPException(status_code=400, detail="Username already exists")
        user = UserORM(id=str(uuid.uuid4()), username=username, password_hash=pwd_context.hash(password))
        db.add(user)
        db.commit()
        return {"message": "User created successfully"}
    finally:
        db.close()

@app.post("/auth/login")
async def login(form: OAuth2PasswordRequestForm = Depends()):
    db: Session = SessionLocal()
    try:
        user = db.query(UserORM).filter_by(username=form.username).first()
        if not user or not pwd_context.verify(form.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return {"access_token": create_token(user.id), "token_type": "bearer"}
    finally:
        db.close()

# ---------------- Neo4j Graph Endpoint ---------------- #
@app.get("/graph")
async def get_graph(session_id: Optional[str] = None, user_id: str = Depends(require_user)):
    query = """
    MATCH (c:Case {user_id: $user_id})-[r]->(e)
    RETURN c.session_id AS session_id, c.title AS case_title,
           type(r) AS rel_type, e.name AS entity_name, labels(e) AS entity_type
    """
    params = {"user_id": user_id}
    if session_id:
        query = """
        MATCH (c:Case {session_id: $session_id, user_id: $user_id})-[r]->(e)
        RETURN c.session_id AS session_id, c.title AS case_title,
               type(r) AS rel_type, e.name AS entity_name, labels(e) AS entity_type
        """
        params["session_id"] = session_id
    results = []
    try:
        with driver.session() as session:
            res = session.run(query, **params)
            for record in res:
                results.append({
                    "session_id": record["session_id"],
                    "case_title": record["case_title"],
                    "relation": record["rel_type"],
                    "entity_name": record["entity_name"],
                    "entity_type": record["entity_type"],
                })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neo4j query failed: {e}")
    return {"graph": results}

# ---------------- Dialogflow Webhook (Protected) ---------------- #
@app.post("/webhook")
async def webhook(req: Request, user_id: str = Depends(require_user)):
    try:
        body = await req.json()
    except Exception:
        return JSONResponse(content={"fulfillmentText": "Invalid JSON payload"}, status_code=400)

    intent = body.get("queryResult", {}).get("intent", {}).get("displayName")
    session_id = body.get("session", str(uuid.uuid4())).split("/")[-1]
    parameters = body.get("queryResult", {}).get("parameters", {})
    user_text = (parameters.get("text", "") or "").strip()

    if user_text:
        STORE.upsert(session_id, user_id, normalize_text(user_text), meta={"source": "dialogflow"})

    record = STORE.get_merged_record(session_id, user_id)
    reply = "I did not understand or no document available."

    if intent == "upload_document":
        reply = "Document uploaded. You can now ask questions."
    elif intent == "paste_text":
        reply = "Text received." if user_text else "No text provided."
    elif record:
        if intent == "ask_judges":
            reply = ", ".join(record.entities.get("judges", ["Not found"]))
        elif intent == "ask_citations":
            reply = ", ".join(record.entities.get("citations", ["Not found"]))
        elif intent == "ask_acts_sections":
            reply = ", ".join(record.entities.get("acts_sections", ["Not found"]))
        elif intent == "summarize":
            reply = record.summary or "No summary available."

    return JSONResponse(content={"fulfillmentText": reply})
