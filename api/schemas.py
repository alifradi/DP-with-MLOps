from pydantic import BaseModel

class TimeSeriesResponse(BaseModel):
    date: str
    predictions: list[float]