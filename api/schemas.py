from pydantic import BaseModel

class ITSupportRequest(BaseModel):
    question: str
