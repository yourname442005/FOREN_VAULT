from fastapi import FastAPI

from app.api.evidence import router as evidence_router
from app.core.database import initialize_database


app = FastAPI(
    title="FORENVAULT",
    description="Multi-Vendor DVR/NVR Forensic Analysis Platform",
    version="0.1.0",
)

initialize_database()

app.include_router(evidence_router)


@app.get("/")
def root():
    return {
        "application": "FORENVAULT",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }