from fastapi.middleware.cors import CORSMiddleware
import os
import random
import resend
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from dotenv import load_dotenv
import cohere
from groq import Groq
import asyncpg
from passlib.context import CryptContext
from jose import jwt, JWTError

load_dotenv()

cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
DATABASE_URL = os.getenv("DATABASE_URL")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"
resend.api_key = os.getenv("RESEND_API_KEY")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- MODELS ----------

class UserSignup(BaseModel):
    email: str
    password: str


class UserLogin(BaseModel):
    email: str
    password: str


class VerifyEmail(BaseModel):
    email: str
    code: str


# ---------- HELPER FUNCTIONS ----------

def hash_password(password: str):
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=7)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


async def get_connection():
    return await asyncpg.connect(DATABASE_URL)


def generate_otp():
    return str(random.randint(100000, 999999))


def send_verification_email(email: str, code: str):
    resend.Emails.send({
        "from": "onboarding@resend.dev",
        "to": email,
        "subject": "Verify your email - AI Knowledge API",
        "html": f"<p>Your verification code is: <strong>{code}</strong></p><p>This code expires in 10 minutes.</p>"
    })


# ---------- ROOT ----------

@app.get("/")
def read_root():
    return {"message": "AI Knowledge API is running!"}


# ---------- AUTH ROUTES ----------

@app.post("/signup")
async def signup(user: UserSignup):
    conn = await get_connection()

    existing = await conn.fetchrow("SELECT id FROM users WHERE email = $1", user.email)
    if existing:
        await conn.close()
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pw = hash_password(user.password)
    otp = generate_otp()
    expires_at = datetime.utcnow() + timedelta(minutes=10)

    new_user = await conn.fetchrow(
        "INSERT INTO users (email, password_hash, verification_code, code_expires_at) VALUES ($1, $2, $3, $4) RETURNING id",
        user.email, hashed_pw, otp, expires_at
    )
    await conn.close()

    send_verification_email(user.email, otp)

    return {"message": "Signup successful! Check your email for verification code.", "user_id": new_user["id"]}


@app.post("/verify-email")
async def verify_email(data: VerifyEmail):
    conn = await get_connection()
    user = await conn.fetchrow(
        "SELECT id, verification_code, code_expires_at FROM users WHERE email = $1",
        data.email
    )

    if not user:
        await conn.close()
        raise HTTPException(status_code=404, detail="User not found")

    if user["verification_code"] != data.code:
        await conn.close()
        raise HTTPException(status_code=400, detail="Invalid verification code")

    if datetime.utcnow() > user["code_expires_at"]:
        await conn.close()
        raise HTTPException(status_code=400, detail="Verification code expired")

    await conn.execute(
        "UPDATE users SET is_verified = TRUE, verification_code = NULL WHERE id = $1",
        user["id"]
    )
    await conn.close()

    token = create_access_token({"user_id": user["id"]})
    return {"message": "Email verified successfully!", "access_token": token, "token_type": "bearer"}


@app.post("/login")
async def login(user: UserLogin):
    conn = await get_connection()
    db_user = await conn.fetchrow("SELECT id, password_hash FROM users WHERE email = $1", user.email)
    await conn.close()

    if not db_user or not verify_password(user.password, db_user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"user_id": db_user["id"]})
    return {"message": "Login successful!", "access_token": token, "token_type": "bearer"}


@app.get("/me")
async def get_me(user_id: int = Depends(get_current_user)):
    return {"user_id": user_id, "message": "Token is valid!"}


# ---------- DOCUMENT / RAG ROUTES ----------

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), user_id: int = Depends(get_current_user)):
    content = await file.read()
    text = content.decode("utf-8")
    chunks = chunk_text(text)

    response = cohere_client.embed(
        texts=chunks,
        model="embed-english-v3.0",
        input_type="search_document"
    )
    embeddings = response.embeddings

    conn = await get_connection()
    for chunk, embedding in zip(chunks, embeddings):
        embedding_str = str(embedding)
        await conn.execute(
            "INSERT INTO documents (filename, text, embedding, user_id) VALUES ($1, $2, $3, $4)",
            file.filename, chunk, embedding_str, user_id
        )
    await conn.close()

    return {
        "filename": file.filename,
        "total_chunks": len(chunks),
        "message": "Document processed and saved permanently in database!"
    }


@app.get("/documents/count")
async def get_document_count(user_id: int = Depends(get_current_user)):
    conn = await get_connection()
    count = await conn.fetchval("SELECT COUNT(*) FROM documents WHERE user_id = $1", user_id)
    await conn.close()
    return {"total_chunks_stored": count}


@app.post("/query")
async def query_documents(question: str, user_id: int = Depends(get_current_user)):
    query_embedding_response = cohere_client.embed(
        texts=[question],
        model="embed-english-v3.0",
        input_type="search_query"
    )
    query_embedding = str(query_embedding_response.embeddings[0])

    conn = await get_connection()
    rows = await conn.fetch(
        "SELECT text FROM documents WHERE user_id = $1 ORDER BY embedding <=> $2 LIMIT 3",
        user_id, query_embedding
    )
    await conn.close()

    if not rows:
        return {"answer": "Koi document upload nahi hua abhi tak."}

    context = "\n\n".join([row["text"] for row in rows])

    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Answer the question based only on the given context. If the answer isn't in the context, say you don't know."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )

    answer = completion.choices[0].message.content
    return {"answer": answer, "sources_used": len(rows)}