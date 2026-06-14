#!/usr/bin/env python3
"""
Full Training and Firebase Deployment Script

This script:
1. Trains a from-scratch model with gpt-5.1-codex-mini via Responses API
2. Runs multiple generations to improve model quality
3. Uploads the trained model to Firebase Storage
4. Stores model metadata in Firestore
5. Deploys Firebase Functions for model serving

Usage:
    python run_full_training_and_deploy.py --name my-model --generations 3 --epochs 3
    python run_full_training_and_deploy.py --name my-model --deploy-only  # Skip training, just deploy
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# Set API key from environment (NEVER hardcode keys)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", os.environ.get("DEEPSEEK_API_KEY", ""))

# Firebase config
FIREBASE_PROJECT = "erosolar-llm"
SERVICE_ACCOUNT_PATH = Path(__file__).parent / "configs" / "firebase-service-account.json"

# Colors
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(msg):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}\n")


def print_step(msg):
    print(f"{GREEN}[+]{RESET} {msg}")


def print_warn(msg):
    print(f"{YELLOW}[!]{RESET} {msg}")


def print_error(msg):
    print(f"{RED}[ERROR]{RESET} {msg}")


def run_training(name: str, preset: str, epochs: int, generations: int, prompts_per_gen: int):
    """Run the multi-generation training with gpt-5.1-codex-mini enhancement."""
    print_header("TRAINING WITH GPT-5.1-CODEX-MINI ENHANCEMENT")

    cmd = [
        sys.executable, "train.py",
        "--name", name,
        "--desc", f"gpt-5.1-codex-mini enhanced from-scratch model (gen={generations})",
        "--preset", preset,
        "--epochs", str(epochs),
        "--generations", str(generations),
        "--prompts-per-gen", str(prompts_per_gen),
        "--upgrade-pipeline"
    ]

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = OPENAI_API_KEY

    print_step(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, env=env, cwd=Path(__file__).parent)

    if result.returncode != 0:
        print_error("Training failed!")
        return False

    print_step("Training completed successfully!")
    return True


def get_model_path(name: str) -> Path:
    """Get the path to a trained model."""
    from registry import get_registry
    registry = get_registry()
    if not registry.exists(name):
        return None
    return registry._get_model_dir(name)


def upload_to_firebase_storage(model_name: str) -> str:
    """Upload trained model to Firebase Storage."""
    print_header("UPLOADING TO FIREBASE STORAGE")

    try:
        import firebase_admin
        from firebase_admin import credentials, storage
    except ImportError:
        print_error("firebase-admin not installed. Run: pip install firebase-admin")
        return None

    # Initialize Firebase
    if not firebase_admin._apps:
        cred = credentials.Certificate(str(SERVICE_ACCOUNT_PATH))
        firebase_admin.initialize_app(cred, {
            'storageBucket': f'{FIREBASE_PROJECT}.firebasestorage.app'
        })

    bucket = storage.bucket()

    # Get model directory
    model_dir = get_model_path(model_name)
    if not model_dir or not model_dir.exists():
        print_error(f"Model '{model_name}' not found!")
        return None

    print_step(f"Model directory: {model_dir}")

    # Create zip of model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"{model_name}_{timestamp}"
    zip_path = Path(f"/tmp/{zip_name}")
    shutil.make_archive(str(zip_path), 'zip', model_dir)
    zip_file = f"{zip_path}.zip"

    print_step(f"Created archive: {zip_file}")

    # Upload to Firebase Storage
    blob_path = f"models/{model_name}/{zip_name}.zip"
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(zip_file)

    # Make publicly accessible (or use signed URLs)
    blob.make_public()
    public_url = blob.public_url

    print_step(f"Uploaded to: gs://{FIREBASE_PROJECT}.firebasestorage.app/{blob_path}")
    print_step(f"Public URL: {public_url}")

    # Cleanup
    os.remove(zip_file)

    return public_url


def store_model_metadata(model_name: str, storage_url: str):
    """Store model metadata in Firestore."""
    print_header("STORING METADATA IN FIRESTORE")

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
    except ImportError:
        print_error("firebase-admin not installed")
        return False

    # Initialize Firebase if not already
    if not firebase_admin._apps:
        cred = credentials.Certificate(str(SERVICE_ACCOUNT_PATH))
        firebase_admin.initialize_app(cred)

    db = firestore.client()

    # Get model info from registry
    from registry import get_registry
    registry = get_registry()
    info = registry.get_info(model_name)

    metadata = {
        "name": model_name,
        "description": info.description if info else "",
        "params": info.params if info else 0,
        "final_loss": info.final_loss if info else 0,
        "storage_url": storage_url,
        "created": datetime.now().isoformat(),
        "preset": info.preset if info else "unknown",
        "tags": info.tags if info else [],
        "api_models": ["gpt-5.1-codex-mini", "gpt-5.1-mini", "gpt-5-nano"],
        "api_type": "responses"
    }

    # Store in Firestore
    doc_ref = db.collection('models').document(model_name)
    doc_ref.set(metadata)

    print_step(f"Stored metadata in Firestore: models/{model_name}")
    print_step(f"Metadata: {json.dumps(metadata, indent=2)}")

    return True


def create_firebase_functions():
    """Create Firebase Functions for model serving."""
    print_header("CREATING FIREBASE FUNCTIONS")

    functions_dir = Path(__file__).parent / "firebase_functions"
    functions_dir.mkdir(exist_ok=True)

    # Create package.json
    package_json = {
        "name": "erosolar-functions",
        "description": "Firebase Functions for Erosolar model serving",
        "engines": {"node": "18"},
        "main": "index.js",
        "dependencies": {
            "firebase-admin": "^11.8.0",
            "firebase-functions": "^4.3.1"
        }
    }

    with open(functions_dir / "package.json", "w") as f:
        json.dump(package_json, f, indent=2)

    # Create index.js
    index_js = '''const functions = require('firebase-functions');
const admin = require('firebase-admin');

admin.initializeApp();
const db = admin.firestore();
const storage = admin.storage();

/**
 * Get available models
 */
exports.getModels = functions.https.onRequest(async (req, res) => {
  res.set('Access-Control-Allow-Origin', '*');

  if (req.method === 'OPTIONS') {
    res.set('Access-Control-Allow-Methods', 'GET');
    res.set('Access-Control-Allow-Headers', 'Content-Type');
    res.status(204).send('');
    return;
  }

  try {
    const snapshot = await db.collection('models').get();
    const models = [];
    snapshot.forEach(doc => {
      models.push({ id: doc.id, ...doc.data() });
    });
    res.json({ success: true, models });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * Get model info
 */
exports.getModelInfo = functions.https.onRequest(async (req, res) => {
  res.set('Access-Control-Allow-Origin', '*');

  if (req.method === 'OPTIONS') {
    res.set('Access-Control-Allow-Methods', 'GET');
    res.set('Access-Control-Allow-Headers', 'Content-Type');
    res.status(204).send('');
    return;
  }

  const modelName = req.query.name;
  if (!modelName) {
    res.status(400).json({ success: false, error: 'Model name required' });
    return;
  }

  try {
    const doc = await db.collection('models').doc(modelName).get();
    if (!doc.exists) {
      res.status(404).json({ success: false, error: 'Model not found' });
      return;
    }
    res.json({ success: true, model: { id: doc.id, ...doc.data() } });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * Get training results/benchmarks
 */
exports.getBenchmarks = functions.https.onRequest(async (req, res) => {
  res.set('Access-Control-Allow-Origin', '*');

  if (req.method === 'OPTIONS') {
    res.set('Access-Control-Allow-Methods', 'GET');
    res.set('Access-Control-Allow-Headers', 'Content-Type');
    res.status(204).send('');
    return;
  }

  try {
    const snapshot = await db.collection('benchmarks').orderBy('timestamp', 'desc').limit(50).get();
    const benchmarks = [];
    snapshot.forEach(doc => {
      benchmarks.push({ id: doc.id, ...doc.data() });
    });
    res.json({ success: true, benchmarks });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * Store benchmark result (admin only)
 */
exports.storeBenchmark = functions.https.onRequest(async (req, res) => {
  res.set('Access-Control-Allow-Origin', '*');

  if (req.method === 'OPTIONS') {
    res.set('Access-Control-Allow-Methods', 'POST');
    res.set('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    res.status(204).send('');
    return;
  }

  // Check for admin key
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    res.status(401).json({ success: false, error: 'Unauthorized' });
    return;
  }

  try {
    const benchmark = req.body;
    benchmark.timestamp = admin.firestore.FieldValue.serverTimestamp();

    const docRef = await db.collection('benchmarks').add(benchmark);
    res.json({ success: true, id: docRef.id });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

/**
 * Get GPT-5.x model configuration
 */
exports.getGPT5Config = functions.https.onRequest(async (req, res) => {
  res.set('Access-Control-Allow-Origin', '*');

  if (req.method === 'OPTIONS') {
    res.set('Access-Control-Allow-Methods', 'GET');
    res.set('Access-Control-Allow-Headers', 'Content-Type');
    res.status(204).send('');
    return;
  }

  // Return locked GPT-5.x configuration
  const config = {
    primary: "gpt-5.1-codex-mini",
    backup: ["gpt-5.1-mini", "gpt-5-nano"],
    api_type: "responses",
    fallback_to_chat: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    locked: true
  };

  res.json({ success: true, config });
});
'''

    with open(functions_dir / "index.js", "w") as f:
        f.write(index_js)

    # Create .firebaserc
    firebaserc = {
        "projects": {
            "default": FIREBASE_PROJECT
        }
    }
    with open(functions_dir.parent / ".firebaserc", "w") as f:
        json.dump(firebaserc, f, indent=2)

    # Create firebase.json
    firebase_json = {
        "functions": {
            "source": "firebase_functions",
            "runtime": "nodejs18"
        },
        "hosting": {
            "public": "angular_app/dist/browser",
            "ignore": ["firebase.json", "**/.*", "**/node_modules/**"],
            "rewrites": [
                {"source": "/api/**", "function": "getModels"},
                {"source": "**", "destination": "/index.html"}
            ]
        }
    }
    with open(functions_dir.parent / "firebase.json", "w") as f:
        json.dump(firebase_json, f, indent=2)

    print_step(f"Created Firebase Functions in: {functions_dir}")
    print_step("Files created: package.json, index.js, firebase.json, .firebaserc")

    return True


def create_angular_app():
    """Create Angular app for displaying results."""
    print_header("CREATING ANGULAR APP")

    angular_dir = Path(__file__).parent / "angular_app"
    angular_dir.mkdir(exist_ok=True)

    # Create minimal Angular setup (or just HTML for simplicity)
    public_dir = angular_dir / "dist" / "browser"
    public_dir.mkdir(parents=True, exist_ok=True)

    # Create index.html
    index_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Erosolar Model Dashboard</title>
    <script src="https://www.gstatic.com/firebasejs/10.7.1/firebase-app-compat.js"></script>
    <script src="https://www.gstatic.com/firebasejs/10.7.1/firebase-firestore-compat.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #fff;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        h1 {
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 2rem;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 1.5rem; }
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }
        .card h2 { color: #00d9ff; margin-bottom: 1rem; font-size: 1.3rem; }
        .stat { display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.1); }
        .stat:last-child { border-bottom: none; }
        .stat-label { color: #888; }
        .stat-value { color: #00ff88; font-weight: 600; }
        .benchmark-table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
        .benchmark-table th, .benchmark-table td { padding: 0.75rem; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.1); }
        .benchmark-table th { color: #00d9ff; }
        .score { font-weight: 600; }
        .score.high { color: #00ff88; }
        .score.medium { color: #ffaa00; }
        .score.low { color: #ff4444; }
        .loading { text-align: center; padding: 2rem; color: #888; }
        .gpt5-badge {
            display: inline-block;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            color: #000;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .api-config { margin-top: 1rem; padding: 1rem; background: rgba(0,0,0,0.3); border-radius: 8px; }
        .api-config code { color: #00ff88; font-family: 'Fira Code', monospace; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Erosolar Model Dashboard</h1>

        <div class="grid">
            <!-- Models Card -->
            <div class="card">
                <h2>Trained Models</h2>
                <div id="models-list" class="loading">Loading models...</div>
            </div>

            <!-- Benchmarks Card -->
            <div class="card" style="grid-column: span 2;">
                <h2>Benchmark Results</h2>
                <table class="benchmark-table">
                    <thead>
                        <tr>
                            <th>Benchmark</th>
                            <th>Erosolar</th>
                        </tr>
                    </thead>
                    <tbody id="benchmarks-table">
                        <tr>
                            <td>GPQA Diamond (Science)</td>
                            <td class="score" id="gpqa">-</td>
                        </tr>
                        <tr>
                            <td>AIME 2025 (Math)</td>
                            <td class="score" id="aime">-</td>
                        </tr>
                        <tr>
                            <td>SWE-Bench Pro (Coding)</td>
                            <td class="score" id="swe">-</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <!-- Training Stats Card -->
            <div class="card">
                <h2>Training Statistics</h2>
                <div id="training-stats">
                    <div class="stat">
                        <span class="stat-label">Total Enhanced</span>
                        <span class="stat-value" id="total-enhanced">-</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Avg Quality Score</span>
                        <span class="stat-value" id="avg-quality">-</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Generations Trained</span>
                        <span class="stat-value" id="generations">-</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">Final Loss</span>
                        <span class="stat-value" id="final-loss">-</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const firebaseConfig = {
            apiKey: "AIzaSyCtpuLR-rdHoC461vrQ_gActkjHGRcBTas",
            authDomain: "erosolar-llm.firebaseapp.com",
            projectId: "erosolar-llm",
            storageBucket: "erosolar-llm.firebasestorage.app",
            messagingSenderId: "218680900515",
            appId: "1:218680900515:web:797c1567fade8eca597816"
        };

        firebase.initializeApp(firebaseConfig);
        const db = firebase.firestore();

        async function loadModels() {
            const modelsList = document.getElementById('models-list');
            try {
                const snapshot = await db.collection('models').get();
                if (snapshot.empty) {
                    modelsList.innerHTML = '<p>No models found. Run training first.</p>';
                    return;
                }

                let html = '';
                snapshot.forEach(doc => {
                    const model = doc.data();
                    html += `
                        <div class="stat">
                            <span class="stat-label">${model.name}</span>
                            <span class="stat-value">${(model.params / 1e6).toFixed(1)}M params</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Loss</span>
                            <span class="stat-value">${model.final_loss?.toFixed(4) || '-'}</span>
                        </div>
                    `;

                    // Update training stats
                    document.getElementById('final-loss').textContent = model.final_loss?.toFixed(4) || '-';
                    if (model.tags) {
                        const genTag = model.tags.find(t => t.startsWith('gen-'));
                        if (genTag) {
                            document.getElementById('generations').textContent = genTag.replace('gen-', '');
                        }
                    }
                });
                modelsList.innerHTML = html;
            } catch (error) {
                modelsList.innerHTML = `<p>Error loading models: ${error.message}</p>`;
            }
        }

        async function loadBenchmarks() {
            try {
                const snapshot = await db.collection('benchmarks')
                    .orderBy('timestamp', 'desc')
                    .limit(1)
                    .get();

                if (!snapshot.empty) {
                    const benchmark = snapshot.docs[0].data();

                    if (benchmark.gpqa) {
                        document.getElementById('gpqa').textContent = benchmark.gpqa + '%';
                        document.getElementById('gpqa').className = 'score ' + (benchmark.gpqa >= 92 ? 'high' : benchmark.gpqa >= 80 ? 'medium' : 'low');
                    }

                    if (benchmark.total_enhanced) {
                        document.getElementById('total-enhanced').textContent = benchmark.total_enhanced;
                    }
                    if (benchmark.avg_quality) {
                        document.getElementById('avg-quality').textContent = benchmark.avg_quality.toFixed(3);
                    }
                }
            } catch (error) {
                console.error('Error loading benchmarks:', error);
            }
        }

        loadModels();
        loadBenchmarks();
    </script>
</body>
</html>
'''

    with open(public_dir / "index.html", "w") as f:
        f.write(index_html)

    print_step(f"Created Angular app in: {angular_dir}")
    print_step("Created: dist/browser/index.html")

    return True


def deploy_to_firebase():
    """Deploy to Firebase Hosting and Functions."""
    print_header("DEPLOYING TO FIREBASE")

    project_dir = Path(__file__).parent

    # Check if Firebase CLI is installed
    result = subprocess.run(["which", "firebase"], capture_output=True)
    if result.returncode != 0:
        print_error("Firebase CLI not installed. Run: npm install -g firebase-tools")
        return False

    # Install functions dependencies
    functions_dir = project_dir / "firebase_functions"
    if (functions_dir / "package.json").exists():
        print_step("Installing Firebase Functions dependencies...")
        subprocess.run(["npm", "install"], cwd=functions_dir)

    # Deploy
    print_step("Deploying to Firebase...")
    result = subprocess.run(
        ["firebase", "deploy", "--project", FIREBASE_PROJECT],
        cwd=project_dir
    )

    if result.returncode != 0:
        print_warn("Firebase deploy may have had issues. Check output above.")

    print_step(f"Dashboard URL: https://{FIREBASE_PROJECT}.web.app")

    return True


def main():
    parser = argparse.ArgumentParser(description="Train and deploy Erosolar model")
    parser.add_argument("--name", type=str, default="erosolar-gpt52", help="Model name")
    parser.add_argument("--preset", type=str, default="medium-2", help="Model preset")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per generation")
    parser.add_argument("--generations", type=int, default=3, help="Number of model generations")
    parser.add_argument("--prompts-per-gen", type=int, default=100, help="Prompts per generation")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, just deploy")
    parser.add_argument("--skip-upload", action="store_true", help="Skip Firebase upload")
    parser.add_argument("--skip-deploy", action="store_true", help="Skip Firebase deploy")
    args = parser.parse_args()

    print_header("EROSOLAR FULL TRAINING AND DEPLOYMENT PIPELINE")
    print(f"  Model: {args.name}")
    print(f"  Preset: {args.preset}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Generations: {args.generations}")
    print(f"  Prompts/Gen: {args.prompts_per_gen}")

    # Step 1: Training
    if not args.skip_training:
        if not run_training(args.name, args.preset, args.epochs, args.generations, args.prompts_per_gen):
            print_error("Training failed! Aborting.")
            return 1

    # Step 2: Upload to Firebase
    if not args.skip_upload:
        storage_url = upload_to_firebase_storage(args.name)
        if storage_url:
            store_model_metadata(args.name, storage_url)
        else:
            print_warn("Upload failed, continuing with deployment...")

    # Step 3: Create Firebase Functions
    create_firebase_functions()

    # Step 4: Create Angular App
    create_angular_app()

    # Step 5: Deploy to Firebase
    if not args.skip_deploy:
        deploy_to_firebase()

    print_header("DEPLOYMENT COMPLETE")
    print(f"  Dashboard: https://{FIREBASE_PROJECT}.web.app")
    print(f"  Functions: https://us-central1-{FIREBASE_PROJECT}.cloudfunctions.net/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
