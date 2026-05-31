from fastapi import FastAPI


app = FastAPI(
    title="Profile Deduplication API",
    version="1.0.0",
)


@app.get("/")
def healthcheck() -> dict:
    return {
        "status": "ok",
        "service": "profile-deduplication-api",
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
    }