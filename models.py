import pandas as pd

from pydantic import create_model, BaseModel, Field
from typing import Optional

def build_response_model(questions: pd.DataFrame) -> BaseModel:
    
    try:
        str_questions = questions['questions'].tolist()
        str_questions = [x.replace(' ', '_').replace('?', '') for x in str_questions]
        fields = {q: (Optional[str], Field(default=None, description="Provide an answer to this questions based on the context")) for q in str_questions}
        model = create_model("Response", **fields) # type: ignore
        return model
    except KeyError as e:
        raise KeyError("Questions must contain a column named 'questions'") from e


