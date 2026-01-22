from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os
from MLOps.monitoring import generate_drift_report

app = FastAPI(title="Emotion Classification API")

@app.get("/monitoring", response_class=HTMLResponse)
async def get_monitoring():
    """Endpoint to trigger drift detection and return the HTML report."""
    
    # Generate report using the utility
    report_path = generate_drift_report()
    
    if report_path and os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    
    return HTMLResponse(
        content="<h1>Error</h1><p>Reference or logged data not found.</p>", 
        status_code=404
    )

