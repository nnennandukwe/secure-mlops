from fastapi import FastAPI, Request
import joblib
import numpy as np
import uvicorn

app = FastAPI()
model = joblib.load("model.pkl")

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    input_array = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(input_array)
    return {"prediction": prediction.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)