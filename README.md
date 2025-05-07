# Secure MLOps with JFrog: A Step-by-Step Guide for AI/ML Practitioners

[Presentation of Overview of Security for MLOps](https://gamma.app/docs/DevSecOps-for-ML-Securing-Your-Model-Supply-Chain-57uwanqs39hk6ib)

## Overview

This tutorial provides a comprehensive, hands-on walkthrough for building a secure MLOps workflow using the JFrog Platform. You’ll train a real machine learning model, containerize the serving application, store artifacts securely using JFrog Artifactory, and simulate vulnerability scanning with JFrog Xray.

Whether you're an ML Engineer, AI Engineer, or Data Scientist, this guide will help you integrate DevSecOps principles into your ML lifecycle using tools built for production-grade software delivery.

### What You'll Learn

* How to train and serve a real ML model
* How to store model artifacts and Docker images in JFrog Artifactory
* How to use the JFrog Platform UI for managing and scanning artifacts
* How to automate secure deployment workflows using CI/CD

### Prerequisites

* Basic knowledge of Python, Git, and Docker
* Docker installed locally
* A free JFrog Platform account ([start here](https://jfrog.com/start-free/))
* GitHub account (for CI integration)

---

## Step 1: Set Up the Project

You’ll begin by creating the code and configuration files for your ML application. These include a model training script, a FastAPI-based inference server, dependency definitions, and a Dockerfile to package everything together.

### Create a Working Directory

```bash
git clone https://github.com/<your-org>/secure-mlops-tutorial.git
cd secure-mlops-tutorial
```

If you're not using a starter repo, manually create the following files:

### `train.py`

This script trains a simple scikit-learn model and saves it locally for use by the server.

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

data = load_iris()
X, y = data.data, data.target

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model.pkl")
```

### `app.py`

This FastAPI application loads the model and serves predictions via a `/predict` endpoint.

```python
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
```

### `requirements.txt`

This file specifies the Python dependencies for your project. An older version of scikit-learn is intentionally used to simulate a known vulnerability for later Xray scanning.

```
fastapi
uvicorn
scikit-learn==0.24.0  # Older version with known CVEs for demo
joblib
```

### `Dockerfile`

This Dockerfile installs the dependencies, trains the model, and sets up the container to run the inference server.

```Dockerfile
FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .
RUN python train.py

COPY app.py .
COPY model.pkl .

CMD ["python", "app.py"]
```

---

## Step 2: Store Artifacts in JFrog Artifactory

Now that you’ve built the model and packaged your application, you’ll publish both the trained model and Docker image to JFrog Artifactory.

### Log In to Your JFrog Platform

* Go to `https://your-instance.jfrog.io`
* Log in with your credentials
* You’ll land on the main dashboard with access to Artifactory and Xray

### Upload the Model Artifact via UI

1. Navigate to **Artifactory** → **Artifacts**
2. Select or create a repository called `ml-models`
3. Click **Deploy** (top right)
4. Upload `model.pkl`
5. Set the target path as `model/1.0.0/model.pkl`
6. Click **Deploy Artifact**

✅ Your trained model is now versioned, trackable, and secured in your central artifact store.

### Push the Docker Image via CLI

Build and push your container image:

```bash
docker login your-instance.jfrog.io
docker build -t your-instance.jfrog.io/your-docker-repo/secure-ml-app:v1.0 .
docker push your-instance.jfrog.io/your-docker-repo/secure-ml-app:v1.0
```

### Verify Artifacts in the UI

* Return to **Artifacts**
* Locate the Docker image and model file in their respective repositories
* Click on each to view metadata, SHA hashes, and associated build info

---

## Step 3: Scan Artifacts with JFrog Xray

JFrog Xray scans artifacts in Artifactory for security and license risks. After uploading your assets, you can view their scan results in the UI.

### Use the Xray Dashboard

1. Navigate to **Xray** → **Vulnerabilities**
2. Filter by your relevant repository (e.g., `docker-local`, `ml-models`)
3. Click your Docker image (`secure-ml-app`) or model file (`model.pkl`)
4. Review the details:

   * Vulnerabilities grouped by severity
   * Components affected (e.g., `scikit-learn`)
   * Suggested fixes or patches
   * Policy violations triggered

This ensures your ML artifacts meet your organization’s security and compliance policies.

---

## Step 4: Automate with GitHub Actions (Optional)

Once your manual workflow is working, you can automate it using GitHub Actions. Below is a workflow for CI/CD that pushes both the image and model to Artifactory.

### `.github/workflows/ci.yml`

```yaml
name: Secure ML CI Pipeline

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Log in to JFrog
      uses: docker/login-action@v2
      with:
        registry: ${{ secrets.JFROG_URL }}
        username: ${{ secrets.JFROG_USER }}
        password: ${{ secrets.JFROG_API_KEY }}

    - name: Build and Push Docker Image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ secrets.JFROG_URL }}/${{ secrets.JFROG_DOCKER_REPO }}/secure-ml-app:${{ github.ref_name }}

    - name: Upload Model Artifact
      run: |
        curl -u ${{ secrets.JFROG_USER }}:${{ secrets.JFROG_API_KEY }} \
          -T model.pkl \
          "https://${{ secrets.JFROG_URL }}/artifactory/ml-models/model/${{ github.ref_name }}/model.pkl"

    - name: Simulate Xray Scan
      run: echo "In production, this would trigger a scan and fail the build if issues are found."
```

---

## Step 5: Best Practices for Secure MLOps

* **Use the JFrog UI** for artifact visibility, traceability, and governance
* **Version your models and containers** with semantic tags
* **Scan both model artifacts and application code** for security issues
* **Automate builds and scans** in CI pipelines
* **Establish Xray policies** to enforce vulnerability and license compliance

---

## Conclusion

You’ve successfully implemented a secure, reproducible MLOps pipeline using the JFrog Platform. This tutorial covered the full lifecycle—from training and packaging to storing, scanning, and automating your machine learning application.

By combining the JFrog UI with DevOps automation, you empower your team to build safer, more reliable AI solutions at scale.

---

## Further Reading

* [What is MLOps?](https://jfrog.com/devops-tools/article/mlops-machine-learning-operations/)
* [Using JFrog with Docker](https://jfrog.com/help/r/jfrog-artifactory-documentation/docker-repositories)
* [Understanding JFrog Xray](https://jfrog.com/xray/)
* [JFrog CLI Documentation](https://jfrog.com/help/r/jfrog-cli)

## Tags

mlops, secure mlops, docker, jfrog artifactory, jfrog xray, container security, vulnerability scanning, ci/cd, ai engineering, devsecops, model registry, machine learning, data science security
