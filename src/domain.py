from pydantic import BaseModel, field_validator


class Weights(BaseModel):
    weights: dict[str, float]

    @field_validator("weights")
    def check_sum(cls, v):
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to ~1.0, got {total}")
        return v
