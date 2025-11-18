import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from bson import ObjectId

from database import db, create_document, get_documents

app = FastAPI(title="AI Model Rankings API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateCategory(BaseModel):
    name: str
    slug: str
    description: Optional[str] = None
    icon: Optional[str] = None


class CreateModel(BaseModel):
    name: str
    vendor: str
    open_source: bool
    parameters_b: Optional[float] = None
    modalities: List[str] = []
    categories: List[str] = []
    context_length: Optional[int] = None
    url: Optional[str] = None
    tags: List[str] = []


class CreateBenchmark(BaseModel):
    name: str
    slug: str
    description: Optional[str] = None
    higher_is_better: bool = True
    unit: Optional[str] = None
    domain: Optional[str] = None


class CreateScore(BaseModel):
    model_id: str
    benchmark_id: str
    score: float
    source: Optional[str] = None
    date: Optional[str] = None


@app.get("/")
def read_root():
    return {"message": "AI Rankings Backend Running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response


# Utility

def to_object_id(id_str: str) -> ObjectId:
    try:
        return ObjectId(id_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ObjectId format")


# Categories
@app.post("/api/categories")
def create_category(payload: CreateCategory):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    # Ensure unique slug
    exists = db["category"].find_one({"slug": payload.slug})
    if exists:
        raise HTTPException(status_code=400, detail="Category with this slug already exists")
    new_id = create_document("category", payload.dict())
    return {"id": new_id}


@app.get("/api/categories")
def list_categories() -> List[Dict[str, Any]]:
    docs = get_documents("category")
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs


# Models
@app.post("/api/models")
def create_model(payload: CreateModel):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    new_id = create_document("aimodel", payload.dict())
    return {"id": new_id}


@app.get("/api/models")
def list_models(category: Optional[str] = None, vendor: Optional[str] = None, modality: Optional[str] = None):
    query: Dict[str, Any] = {}
    if category:
        query["categories"] = {"$in": [category]}
    if vendor:
        query["vendor"] = vendor
    if modality:
        query["modalities"] = {"$in": [modality]}
    docs = list(db["aimodel"].find(query)) if db else []
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs


# Benchmarks
@app.post("/api/benchmarks")
def create_benchmark(payload: CreateBenchmark):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    exists = db["benchmark"].find_one({"slug": payload.slug})
    if exists:
        raise HTTPException(status_code=400, detail="Benchmark with this slug already exists")
    new_id = create_document("benchmark", payload.dict())
    return {"id": new_id}


@app.get("/api/benchmarks")
def list_benchmarks():
    docs = get_documents("benchmark")
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs


# Scores
@app.post("/api/scores")
def create_score(payload: CreateScore):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")
    # Validate FK existence
    m = db["aimodel"].find_one({"_id": to_object_id(payload.model_id)})
    b = db["benchmark"].find_one({"_id": to_object_id(payload.benchmark_id)})
    if not m or not b:
        raise HTTPException(status_code=404, detail="Model or Benchmark not found")
    new_id = create_document("score", payload.dict())
    return {"id": new_id}


@app.get("/api/rankings")
def rankings(category: Optional[str] = None, benchmark: Optional[str] = None, top: int = 10):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")

    # Resolve benchmark by slug if provided
    benchmark_doc = None
    if benchmark:
        benchmark_doc = db["benchmark"].find_one({"slug": benchmark})
        if not benchmark_doc:
            raise HTTPException(status_code=404, detail="Benchmark not found")

    # Build model filter
    model_filter: Dict[str, Any] = {}
    if category:
        model_filter["categories"] = {"$in": [category]}

    models = list(db["aimodel"].find(model_filter))
    model_ids = [m["_id"] for m in models]
    scores_filter: Dict[str, Any] = {"model_id": {"$in": [str(_id) for _id in model_ids]}}
    if benchmark_doc:
        scores_filter["benchmark_id"] = str(benchmark_doc["_id"])  # stored as string in Score

    scores = list(db["score"].find(scores_filter))
    # Aggregate best score per model (assume higher is better by benchmark flag
    best_by_model: Dict[str, float] = {}
    high_better = True
    if benchmark_doc and "higher_is_better" in benchmark_doc:
        high_better = bool(benchmark_doc["higher_is_better"])

    for s in scores:
        mid = s["model_id"]
        val = float(s["score"])
        if mid not in best_by_model:
            best_by_model[mid] = val
        else:
            if (high_better and val > best_by_model[mid]) or ((not high_better) and val < best_by_model[mid]):
                best_by_model[mid] = val

    # Join with model details
    enriched: List[Dict[str, Any]] = []
    for m in models:
        mid = str(m["_id"])
        score_val = best_by_model.get(mid)
        if score_val is not None:
            enriched.append({
                "model_id": mid,
                "model": m["name"],
                "vendor": m.get("vendor"),
                "open_source": m.get("open_source"),
                "modalities": m.get("modalities", []),
                "categories": m.get("categories", []),
                "score": score_val
            })

    enriched.sort(key=lambda x: x["score"], reverse=high_better)

    return {"items": enriched[: max(1, min(top, 100))], "count": len(enriched)}


@app.post("/api/seed")
def seed():
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")

    # Create categories
    categories = [
        {"name": "Assistant / Chat", "slug": "chat", "description": "General assistants", "icon": "MessageSquare"},
        {"name": "Coding", "slug": "coding", "description": "Code generation & reasoning", "icon": "Code"},
        {"name": "Vision", "slug": "vision", "description": "Image understanding", "icon": "Image"},
        {"name": "Multimodal", "slug": "multimodal", "description": "Text+Vision+Audio", "icon": "Sparkles"},
        {"name": "Reasoning", "slug": "reasoning", "description": "Math & logic", "icon": "Sigma"}
    ]
    for c in categories:
        if not db["category"].find_one({"slug": c["slug"]}):
            create_document("category", c)

    # Create benchmarks
    benches = [
        {"name": "MMLU", "slug": "mmlu", "description": "General knowledge", "higher_is_better": True, "unit": "%", "domain": "knowledge"},
        {"name": "GSM8K", "slug": "gsm8k", "description": "Grade school math", "higher_is_better": True, "unit": "%", "domain": "math"},
        {"name": "HumanEval", "slug": "humaneval", "description": "Coding pass@1", "higher_is_better": True, "unit": "%", "domain": "coding"},
        {"name": "MT-Bench", "slug": "mtbench", "description": "Chat quality", "higher_is_better": True, "unit": "score", "domain": "chat"}
    ]
    for b in benches:
        if not db["benchmark"].find_one({"slug": b["slug"]}):
            create_document("benchmark", b)

    # Helper to fetch benchmark ids
    bench_map = {b["slug"]: str(db["benchmark"].find_one({"slug": b["slug"]})["_id"]) for b in benches}

    # Create models (examples incl. GPT-5.1)
    models = [
        {"name": "GPT-5.1", "vendor": "OpenAI", "open_source": False, "modalities": ["text", "vision", "audio"], "categories": ["chat", "multimodal", "reasoning"], "context_length": 200000, "tags": ["SOTA", "2025"]},
        {"name": "Llama 3.2 90B", "vendor": "Meta", "open_source": True, "modalities": ["text", "vision"], "categories": ["chat", "vision"], "context_length": 131072, "tags": ["open-source"]},
        {"name": "Mistral Large 2", "vendor": "Mistral", "open_source": False, "modalities": ["text"], "categories": ["chat", "reasoning"], "context_length": 128000},
        {"name": "DeepSeek R1", "vendor": "DeepSeek", "open_source": True, "modalities": ["text"], "categories": ["reasoning", "coding"], "context_length": 131072},
        {"name": "Qwen2.5-Coder", "vendor": "Alibaba", "open_source": True, "modalities": ["text"], "categories": ["coding"], "context_length": 131072}
    ]

    model_ids: Dict[str, str] = {}
    for m in models:
        found = db["aimodel"].find_one({"name": m["name"], "vendor": m["vendor"]})
        if not found:
            mid = create_document("aimodel", m)
            model_ids[m["name"]] = mid
        else:
            model_ids[m["name"]] = str(found["_id"])

    # Create scores (synthetic for demo)
    scores = [
        {"model_id": model_ids["GPT-5.1"], "benchmark_id": bench_map["mmlu"], "score": 92.5},
        {"model_id": model_ids["GPT-5.1"], "benchmark_id": bench_map["mtbench"], "score": 9.6},
        {"model_id": model_ids["GPT-5.1"], "benchmark_id": bench_map["gsm8k"], "score": 96.1},
        {"model_id": model_ids["GPT-5.1"], "benchmark_id": bench_map["humaneval"], "score": 92.0},

        {"model_id": model_ids["Llama 3.2 90B"], "benchmark_id": bench_map["mmlu"], "score": 86.3},
        {"model_id": model_ids["Llama 3.2 90B"], "benchmark_id": bench_map["mtbench"], "score": 8.8},
        {"model_id": model_ids["Llama 3.2 90B"], "benchmark_id": bench_map["gsm8k"], "score": 84.7},

        {"model_id": model_ids["Mistral Large 2"], "benchmark_id": bench_map["mmlu"], "score": 84.1},
        {"model_id": model_ids["Mistral Large 2"], "benchmark_id": bench_map["mtbench"], "score": 8.6},

        {"model_id": model_ids["DeepSeek R1"], "benchmark_id": bench_map["gsm8k"], "score": 92.0},
        {"model_id": model_ids["DeepSeek R1"], "benchmark_id": bench_map["humaneval"], "score": 89.0},

        {"model_id": model_ids["Qwen2.5-Coder"], "benchmark_id": bench_map["humaneval"], "score": 90.5}
    ]

    for s in scores:
        exists = db["score"].find_one({"model_id": s["model_id"], "benchmark_id": s["benchmark_id"]})
        if not exists:
            create_document("score", s)

    return {"status": "ok", "categories": db["category"].count_documents({}), "models": db["aimodel"].count_documents({}), "benchmarks": db["benchmark"].count_documents({}), "scores": db["score"].count_documents({})}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
