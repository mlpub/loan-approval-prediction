# if fastapi is not installed, run: 
# pip install fastapi uvicorn
# if using uv, add it dependencies:
# uv add fastapi uvicorn

import joblib
import pandas as pd
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field, field_validator

# define constant value
best_threshold = 0.62
model_path = 'final_model.pkl'
api_port = 9696

# use Pydantic models to define the request and response schemas
# to make sure the input data is structured correctly
# and the response data is formatted properly
class PredictRequest(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

    # Auto-lowercase certain string fields
    @field_validator("person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file", mode="before")
    def lowercase_str_fields(cls, v):
        return v.lower() if isinstance(v, str) else v
    

class PredictResponse(BaseModel):
    loan_probability: float
    loan_status: int


# initialize a FastAPI application instance
app = FastAPI(title="Loan Approval Prediction")

# load model
with open(model_path, 'rb') as f_in:
    model = joblib.load(f_in)


@app.post("/predict")
def predict(input_data: PredictRequest) -> PredictResponse:
    #print("Received prediction request", flush=True)
    #print(input_data, flush=True)
    #data_2d_array = [list(input_data.values())]
    df = pd.DataFrame([input_data.model_dump()])
    #result = 0.0
    result = model.predict_proba(df)[0, 1]
    print(result, flush=True)

    return PredictResponse(
        loan_probability=result,
        loan_status=int(result >= best_threshold)
    )


if __name__ == "__main__":
    # Run the FastAPI app using Uvicorn server
    # 'host="0.0.0.0"' allows external access (not just localhost)
    # 'port=9696' specifies the port number
    uvicorn.run(app, host="0.0.0.0", port=api_port)

