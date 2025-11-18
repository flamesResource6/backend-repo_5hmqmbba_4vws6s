"""
Database Schemas for AI Model Rankings

Each Pydantic model represents a MongoDB collection. The collection name is the
lowercased class name. For example: AIModel -> "aimodel".
"""
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, HttpUrl


class Category(BaseModel):
    """
    AI model category, e.g., "Chat/Assistant", "Coding", "Vision".
    Collection name: "category"
    """
    name: str = Field(..., description="Category display name")
    slug: str = Field(..., description="URL-friendly identifier, unique")
    description: Optional[str] = Field(None, description="Short description")
    icon: Optional[str] = Field(None, description="Icon name (lucide) or emoji")


class AIModel(BaseModel):
    """
    AI model metadata. Collection name: "aimodel"
    """
    name: str = Field(..., description="Model name, e.g., GPT-5.1")
    vendor: str = Field(..., description="Provider, e.g., OpenAI, Meta, Mistral")
    open_source: bool = Field(..., description="Whether the model is open-source")
    parameters_b: Optional[float] = Field(None, description="Parameters in billions")
    modalities: List[str] = Field(default_factory=list, description="e.g., text, vision, audio")
    categories: List[str] = Field(default_factory=list, description="List of category slugs")
    context_length: Optional[int] = Field(None, description="Max context tokens")
    url: Optional[HttpUrl] = None
    tags: List[str] = Field(default_factory=list)


class Benchmark(BaseModel):
    """
    Benchmark entity, e.g., MMLU, GSM8K, HumanEval, MT-Bench
    Collection name: "benchmark"
    """
    name: str
    slug: str
    description: Optional[str] = None
    higher_is_better: bool = Field(True, description="Whether higher score is better")
    unit: Optional[str] = Field(None, description="e.g., %, pass@1, score")
    domain: Optional[str] = Field(None, description="task domain, e.g., reasoning, coding")


class Score(BaseModel):
    """
    Score of a model on a benchmark. Collection name: "score"
    """
    model_id: str = Field(..., description="ObjectId string of AIModel")
    benchmark_id: str = Field(..., description="ObjectId string of Benchmark")
    score: float = Field(..., description="Numeric score")
    source: Optional[str] = Field(None, description="Source or paper link")
    date: Optional[str] = Field(None, description="ISO date of evaluation")


# Example additional schema kept for reference in the DB viewer
class User(BaseModel):
    name: str
    email: str
    address: str
    age: Optional[int] = None
    is_active: bool = True
