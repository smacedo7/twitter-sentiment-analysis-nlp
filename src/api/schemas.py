"""Request and response models of the REST API (validated automatically by Pydantic)."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

MAX_TWEET_LENGTH = 1000
MAX_BATCH_SIZE = 100

Tweet = Annotated[str, Field(max_length=MAX_TWEET_LENGTH)]


class TweetRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=MAX_TWEET_LENGTH,
        description="Tweet text to analyze.",
        examples=["I love how easy this application is to use!"],
    )


class BatchRequest(BaseModel):
    texts: list[Tweet] = Field(
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=f"Between 1 and {MAX_BATCH_SIZE} tweets.",
        examples=[["I love this!", "Worst service ever."]],
    )


class PredictionResponse(BaseModel):
    label: Literal["Positive", "Negative", "Neutral"] = Field(
        description="`Neutral` means the model is not sure (mixed or ambiguous sentiment)."
    )
    confidence: float = Field(ge=0.5, le=1.0, description="How sure the model is about the more likely side (0.5 to 1.0).")
    positive_probability: float = Field(ge=0.0, le=1.0, description="Calibrated probability that the tweet is positive.")


class BatchItem(BaseModel):
    label: Literal["Positive", "Negative", "Neutral"] | None = Field(
        description="`null` when the text has no analyzable words (only links, mentions or symbols)."
    )
    confidence: float | None = Field(ge=0.5, le=1.0)
    positive_probability: float | None = Field(ge=0.0, le=1.0)


class BatchResponse(BaseModel):
    predictions: list[BatchItem] = Field(description="One result per input text, in the same order.")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool = Field(description="Whether the trained model file is available.")
