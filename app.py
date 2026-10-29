#!/usr/bin/env python3
"""
Interactive Web Application for Plant Disease Detection.
Zero extra web dependencies required - powered by Python's built-in http.server and PyTorch.

Usage:
    python app.py [--port 8000] [--model models/plant_disease_resnet18.pth]
"""

import argparse
import base64
import io
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
from PIL import Image
from src.config import Config
from src.model import PlantDiseaseModel
from src.utils import DEFAULT_PLANT_CLASSES, predict_single_image

# Remedies and management tips for common plant diseases
DISEASE_TIPS = {
    "Bacterial spot": "Avoid overhead watering; remove infected leaves; apply copper-based bactericides early in the morning.",
    "Early blight": "Ensure adequate plant spacing for airflow; mulch around the base; prune lower infected foliage.",
    "Late blight": "Highly contagious water-mold pathogen. Destroy heavily infected plants; apply preventive fungicides.",
    "Leaf mold": "Reduce greenhouse humidity below 85%; prune lower foliage for ventilation; use resistant varieties.",
    "Septoria leaf spot": "Clean up debris at season end; avoid splashing soil onto foliage; apply fungicides at first lesion.",
    "Spider mites": "Spray underside of leaves with water or insecticidal soap; introduce predatory mites; maintain humidity.",
    "Target spot": "Apply protective fungicides; avoid handling plants when wet; practice 3-year crop rotation.",
    "Yellow leaf curl virus": "Transmitted by whiteflies. Use insect netting, reflective silver mulch, and yellow sticky traps.",
    "Mosaic virus": "Transmitted mechanically and via aphids. Disinfect gardening tools; remove infected plants immediately.",
    "Healthy": "No pathological symptoms detected! Continue regular watering, balanced fertilization, and sunlight."
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌿 Plant Disease Detection AI</title>
    <style>
        :root {
            --bg: #0f172a;
            --card: #1e293b;
            --card-border: #334155;
            --accent: #10b981;
            --accent-glow: rgba(16, 185, 129, 0.3);
            --danger: #ef4444;
            --text: #f8fafc;
            --text-muted: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 2rem 1rem; }
        .container { max-width: 820px; width: 100%; }
        header { text-align: center; margin-bottom: 2rem; }
        header h1 { font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #10b981 0%, #38bdf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }
        header p { color: var(--text-muted); font-size: 1.05rem; }
        .card { background: var(--card); border: 1px solid var(--card-border); border-radius: 16px; padding: 2rem; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5); margin-bottom: 1.5rem; }
        .upload-zone { border: 2px dashed #475569; border-radius: 12px; padding: 2.5rem 1.5rem; text-align: center; cursor: pointer; transition: all 0.25s ease; background: rgba(30, 41, 59, 0.5); }
        .upload-zone:hover, .upload-zone.dragover { border-color: var(--accent); background: rgba(16, 185, 129, 0.05); }
        .upload-icon { font-size: 3rem; margin-bottom: 1rem; }
        .file-input { display: none; }
        .preview-box { display: none; text-align: center; margin-top: 1.5rem; }
        .preview-img { max-height: 280px; max-width: 100%; border-radius: 12px; border: 2px solid var(--card-border); }
        .btn { display: inline-block; background: var(--accent); color: #022c22; font-weight: 700; padding: 0.8rem 1.8rem; border-radius: 8px; border: none; cursor: pointer; font-size: 1rem; margin-top: 1rem; transition: all 0.2s; box-shadow: 0 4px 14px var(--accent-glow); }
        .btn:hover { transform: translateY(-2px); filter: brightness(1.1); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .results-box { display: none; margin-top: 1.5rem; }
        .result-item { margin-bottom: 1rem; }
        .result-label { display: flex; justify-content: space-between; font-weight: 600; margin-bottom: 0.3rem; font-size: 0.95rem; }
        .bar-container { background: #334155; border-radius: 9999px; height: 12px; overflow: hidden; }
        .bar-fill { height: 100%; background: linear-gradient(90deg, #10b981, #06b6d4); border-radius: 9999px; transition: width 0.6s ease; }
        .remedy-card { background: rgba(15, 23, 42, 0.6); border-left: 4px solid var(--accent); border-radius: 8px; padding: 1rem 1.25rem; margin-top: 1.5rem; }
        .remedy-card.danger { border-left-color: var(--danger); }
        .remedy-title { font-weight: 700; margin-bottom: 0.4rem; font-size: 1rem; color: #38bdf8; }
        .remedy-text { font-size: 0.92rem; color: var(--text-muted); line-height: 1.5; }
        footer { margin-top: 2rem; text-align: center; color: var(--text-muted); font-size: 0.85rem; }
        footer a { color: var(--accent); text-decoration: none; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🌿 Plant Disease Detection AI</h1>
            <p>ResNet-18 Deep Learning Classifier & Agricultural Diagnostic System</p>
        </header>

        <div class="card">
            <div class="upload-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📸</div>
                <h3>Upload Plant Leaf Photograph</h3>
                <p style="color: var(--text-muted); margin-top: 0.5rem; font-size: 0.9rem;">Drag and drop image here or click to browse (JPG, PNG)</p>
                <input type="file" id="fileInput" class="file-input" accept="image/*" onchange="handleFile(this.files[0])">
            </div>

            <div class="preview-box" id="previewBox">
                <img id="previewImg" class="preview-img" alt="Leaf Preview">
                <div>
                    <button class="btn" id="analyzeBtn" onclick="analyzeLeaf()">Diagnose Disease</button>
                </div>
            </div>

            <div class="results-box" id="resultsBox">
                <h3 style="margin-bottom: 1rem;">Diagnosis Results</h3>
                <div id="predictionsList"></div>
                <div class="remedy-card" id="remedyCard">
                    <div class="remedy-title" id="remedyTitle">Agricultural Recommendation</div>
                    <div class="remedy-text" id="remedyText"></div>
                </div>
            </div>
        </div>

        <footer>
            Built with PyTorch & ResNet-18 • Developed by <a href="https://github.com/aditya-121105">Kumarkhaniya Aditya</a>
        </footer>
    </div>

    <script>
        let currentFile = null;
        const dropZone = document.getElementById('dropZone');

        ['dragenter', 'dragover'].forEach(name => {
            dropZone.addEventListener(name, (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
        });
        ['dragleave', 'drop'].forEach(name => {
            dropZone.addEventListener(name, (e) => { e.preventDefault(); dropZone.classList.remove('dragover'); });
        });
        dropZone.addEventListener('drop', (e) => {
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
        });

        function handleFile(file) {
            if (!file) return;
            currentFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                document.getElementById('previewImg').src = e.target.result;
                document.getElementById('previewBox').style.display = 'block';
                document.getElementById('resultsBox').style.display = 'none';
            };
            reader.readAsDataURL(file);
        }

        async function analyzeLeaf() {
            if (!currentFile) return;
            const btn = document.getElementById('analyzeBtn');
            btn.disabled = true;
            btn.innerText = 'Analyzing Leaf...';

            const reader = new FileReader();
            reader.onload = async () => {
                const base64Data = reader.result.split(',')[1];
                try {
                    const response = await fetch('/api/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: base64Data })
                    });
                    const data = await response.json();
                    renderResults(data);
                } catch (err) {
                    alert('Error connecting to backend: ' + err.message);
                } finally {
                    btn.disabled = false;
                    btn.innerText = 'Diagnose Disease';
                }
            };
            reader.readAsDataURL(currentFile);
        }

        function renderResults(data) {
            const list = document.getElementById('predictionsList');
            list.innerHTML = '';
            (data.predictions || []).forEach(p => {
                const isHealthy = p.class_name.toLowerCase().includes('healthy');
                const color = isHealthy ? 'linear-gradient(90deg, #10b981, #06b6d4)' : 'linear-gradient(90deg, #ef4444, #f59e0b)';
                list.innerHTML += `
                    <div class="result-item">
                        <div class="result-label">
                            <span>${p.class_name}</span>
                            <span>${p.confidence.toFixed(1)}%</span>
                        </div>
                        <div class="bar-container">
                            <div class="bar-fill" style="width: ${p.confidence}%; background: ${color}"></div>
                        </div>
                    </div>
                `;
            });

            const remedyCard = document.getElementById('remedyCard');
            const remedyTitle = document.getElementById('remedyTitle');
            const remedyText = document.getElementById('remedyText');

            const top = (data.predictions || [])[0];
            if (top) {
                const isHealthy = top.class_name.toLowerCase().includes('healthy');
                remedyCard.className = 'remedy-card ' + (isHealthy ? '' : 'danger');
                remedyTitle.innerText = isHealthy ? '🌱 Plant Status: Healthy Foliage' : '⚠️ Treatment Recommendation: ' + top.class_name;
                remedyText.innerText = data.recommendation || 'Consult local agricultural extension office for pesticide advice.';
            }

            document.getElementById('resultsBox').style.display = 'block';
        }
    </script>
</body>
</html>
"""

# Global model instance
APP_MODEL = None
APP_CLASSES = DEFAULT_PLANT_CLASSES


class AppRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "model_loaded": APP_MODEL is not None}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                payload = json.loads(body.decode("utf-8"))
                b64_image = payload.get("image", "")
                image_bytes = base64.b64decode(b64_image)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                # Run inference
                results = predict_single_image(
                    model=APP_MODEL,
                    image_path=io.BytesIO(image_bytes),
                    device=Config.device,
                    class_names=APP_CLASSES,
                    top_k=3
                )

                top_class, top_conf = results[0]
                rec = "No recommendation found."
                for k, v in DISEASE_TIPS.items():
                    if k.lower() in top_class.lower():
                        rec = v
                        break

                response = {
                    "predictions": [{"class_name": c, "confidence": conf} for c, conf in results],
                    "recommendation": rec
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="Plant Disease Detection Web App")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server (default: 8000)")
    parser.add_argument("--model", type=str, default=Config.model_save_path, help="Path to PyTorch model weights (.pth)")
    args = parser.parse_args()

    global APP_MODEL, APP_CLASSES
    print("Loading PyTorch model...")
    APP_MODEL = PlantDiseaseModel(num_classes=len(APP_CLASSES), pretrained=False, freeze_backbone=False)

    if os.path.exists(args.model):
        ckpt = torch.load(args.model, map_location=Config.device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            APP_MODEL.load_state_dict(ckpt["model_state_dict"])
            if ckpt.get("classes"):
                APP_CLASSES = ckpt["classes"]
        else:
            APP_MODEL.load_state_dict(ckpt)
        print(f"Loaded trained checkpoint: {args.model}")
    else:
        print("Initialized baseline model for demo.")

    APP_MODEL = APP_MODEL.to(Config.device)
    APP_MODEL.eval()

    server = HTTPServer(("0.0.0.0", args.port), AppRequestHandler)
    print(f"\n🌱 Plant Disease Web App running at: http://localhost:{args.port}")
    print("Press Ctrl+C to terminate.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        server.server_close()


if __name__ == "__main__":
    main()
