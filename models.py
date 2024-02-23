import pandas as pd

from pydantic import create_model, BaseModel, Field
from typing import Optional

def build_response_model(questions: pd.DataFrame) -> BaseModel:
    
    try:
        str_questions = questions['questions'].tolist()
        str_questions = [x.replace(' ', '_').replace('?', '') for x in str_questions]
        fields = {q: (Optional[dict], Field(default=None, description=f"Provide an answer to this questions based on the context. Must be a dictionary of the form {Answer.model_json_schema()}")) for q in str_questions}
        model = create_model("Response", **fields) # type: ignore
        return model
    except KeyError as e:
        raise KeyError("Questions must contain a column named 'questions'") from e


class Answer(BaseModel):
    answer: str
    timestamp: str

