"""
Cloud Run Manager - Unified deployment for Erosolar LLM
Manages Angular frontend and Python backend deployments to Google Cloud Run.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

# Project paths
PROJECT_ROOT = Path(__file__).parent
ANGULAR_DIR = PROJECT_ROOT / "angular-chat"
DATA_STORE = PROJECT_ROOT / "data_store"


@dataclass
class CloudRunConfig:
    """Configuration for Cloud Run deployment."""
    project_id: str = "erosolar-prod"
    region: str = "us-central1"
    service_name: str = "erosolar-chat"
    memory: str = "512Mi"
    cpu: str = "1"
    min_instances: int = 0
    max_instances: int = 10
    port: int = 8080
    allow_unauthenticated: bool = True


class CloudRunManager:
    """
    Unified Cloud Run deployment manager for Erosolar.

    Handles:
    - Angular frontend deployment
    - Python backend deployment
    - Service management (start, stop, scale)
    - Deployment verification
    """

    def __init__(self, config: Optional[CloudRunConfig] = None):
        self.config = config or CloudRunConfig()
        self.angular_dir = ANGULAR_DIR
        self.project_root = PROJECT_ROOT

    def _run_cmd(self, cmd: List[str], cwd: Optional[Path] = None, capture: bool = True) -> subprocess.CompletedProcess:
        """Run a command and return result."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                capture_output=capture,
                text=True
            )
            return result
        except Exception as e:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"Error: {e}")
            raise

    def check_gcloud_auth(self) -> bool:
        """Check if gcloud is authenticated."""
        result = self._run_cmd(["gcloud", "auth", "list", "--format=json"])
        if result.returncode != 0:
            return False
        try:
            accounts = json.loads(result.stdout)
            return len(accounts) > 0 and any(a.get("status") == "ACTIVE" for a in accounts)
        except:
            return False

    def set_project(self) -> bool:
        """Set the GCP project."""
        result = self._run_cmd([
            "gcloud", "config", "set", "project", self.config.project_id
        ])
        return result.returncode == 0

    # =========================================================================
    # Angular Frontend Deployment
    # =========================================================================

    def build_angular(self) -> bool:
        """Build Angular frontend for production."""
        print(f"\n{'='*60}")
        print("Building Angular Frontend")
        print(f"{'='*60}")

        if not self.angular_dir.exists():
            print(f"Error: Angular directory not found: {self.angular_dir}")
            return False

        # Install dependencies
        print("Installing dependencies...")
        result = self._run_cmd(["npm", "install"], cwd=self.angular_dir)
        if result.returncode != 0:
            print(f"npm install failed: {result.stderr}")
            return False

        # Build production
        print("Building production bundle...")
        result = self._run_cmd(["npm", "run", "build"], cwd=self.angular_dir)
        if result.returncode != 0:
            print(f"Build failed: {result.stderr}")
            return False

        print("Angular build complete")
        return True

    def deploy_angular(self, build: bool = True) -> Optional[str]:
        """
        Deploy Angular frontend to Cloud Run.

        Args:
            build: Whether to build before deploying

        Returns:
            Service URL if successful, None otherwise
        """
        print(f"\n{'='*60}")
        print("Deploying Angular to Cloud Run")
        print(f"{'='*60}")

        if build and not self.build_angular():
            return None

        # Check for Dockerfile
        dockerfile = self.angular_dir / "Dockerfile"
        if not dockerfile.exists():
            print("Creating Dockerfile for Angular...")
            self._create_angular_dockerfile()

        # Deploy to Cloud Run
        print(f"Deploying to {self.config.service_name}...")
        result = self._run_cmd([
            "gcloud", "run", "deploy", self.config.service_name,
            "--source", str(self.angular_dir),
            "--region", self.config.region,
            "--platform", "managed",
            "--memory", self.config.memory,
            "--cpu", self.config.cpu,
            "--min-instances", str(self.config.min_instances),
            "--max-instances", str(self.config.max_instances),
            "--port", str(self.config.port),
            "--allow-unauthenticated" if self.config.allow_unauthenticated else "--no-allow-unauthenticated"
        ])

        if result.returncode != 0:
            print(f"Deployment failed: {result.stderr}")
            return None

        # Get service URL
        url = self.get_service_url()
        if url:
            print(f"\nDeployed successfully: {url}")
        return url

    def _create_angular_dockerfile(self):
        """Create Dockerfile for Angular deployment."""
        dockerfile_content = """FROM nginx:alpine
COPY dist/erosolar-chat/browser /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
"""
        nginx_conf = """server {
    listen 8080;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml text/javascript;
}
"""
        (self.angular_dir / "Dockerfile").write_text(dockerfile_content)
        (self.angular_dir / "nginx.conf").write_text(nginx_conf)
        print("Created Dockerfile and nginx.conf")

    # =========================================================================
    # Service Management
    # =========================================================================

    def get_service_url(self) -> Optional[str]:
        """Get the URL of the deployed service."""
        result = self._run_cmd([
            "gcloud", "run", "services", "describe", self.config.service_name,
            "--region", self.config.region,
            "--format", "value(status.url)"
        ])
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None

    def get_service_status(self) -> Dict[str, Any]:
        """Get detailed status of the Cloud Run service."""
        result = self._run_cmd([
            "gcloud", "run", "services", "describe", self.config.service_name,
            "--region", self.config.region,
            "--format", "json"
        ])
        if result.returncode != 0:
            return {"error": "Service not found", "exists": False}

        try:
            data = json.loads(result.stdout)
            return {
                "exists": True,
                "url": data.get("status", {}).get("url"),
                "ready": data.get("status", {}).get("conditions", [{}])[0].get("status") == "True",
                "latest_revision": data.get("status", {}).get("latestReadyRevisionName"),
                "traffic": data.get("status", {}).get("traffic", [])
            }
        except json.JSONDecodeError:
            return {"error": "Failed to parse response", "exists": False}

    def scale_service(self, min_instances: int, max_instances: int) -> bool:
        """Scale the Cloud Run service."""
        result = self._run_cmd([
            "gcloud", "run", "services", "update", self.config.service_name,
            "--region", self.config.region,
            "--min-instances", str(min_instances),
            "--max-instances", str(max_instances)
        ])
        return result.returncode == 0

    def delete_service(self, confirm: bool = False) -> bool:
        """Delete the Cloud Run service."""
        if not confirm:
            print("Use confirm=True to actually delete the service")
            return False

        result = self._run_cmd([
            "gcloud", "run", "services", "delete", self.config.service_name,
            "--region", self.config.region,
            "--quiet"
        ])
        return result.returncode == 0

    def list_revisions(self) -> List[Dict[str, Any]]:
        """List all revisions of the service."""
        result = self._run_cmd([
            "gcloud", "run", "revisions", "list",
            "--service", self.config.service_name,
            "--region", self.config.region,
            "--format", "json"
        ])
        if result.returncode != 0:
            return []

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return []

    # =========================================================================
    # Training Data Integration
    # =========================================================================

    def get_training_data_stats(self) -> Dict[str, Any]:
        """Get statistics about training data in data_store."""
        stats = {
            "files": [],
            "total_examples": 0,
            "total_size_bytes": 0
        }

        seen_files = set()
        for pattern in ["*_training_data.jsonl", "*_training.jsonl"]:
            for data_file in DATA_STORE.glob(pattern):
                if data_file.name in seen_files:
                    continue
                seen_files.add(data_file.name)

                try:
                    with open(data_file) as f:
                        count = sum(1 for _ in f)
                    size = data_file.stat().st_size

                    stats["files"].append({
                        "name": data_file.name,
                        "examples": count,
                        "size_bytes": size
                    })
                    stats["total_examples"] += count
                    stats["total_size_bytes"] += size
                except Exception as e:
                    print(f"Error reading {data_file}: {e}")

        return stats

    # =========================================================================
    # Full Deployment Pipeline
    # =========================================================================

    def full_deploy(self, build_angular: bool = True) -> Dict[str, Any]:
        """
        Run full deployment pipeline.

        Returns:
            Deployment results including URLs and status
        """
        results = {
            "success": False,
            "angular_url": None,
            "training_data": None,
            "errors": []
        }

        print(f"\n{'='*60}")
        print("EROSOLAR FULL DEPLOYMENT PIPELINE")
        print(f"{'='*60}")

        # Check auth
        if not self.check_gcloud_auth():
            results["errors"].append("Not authenticated with gcloud")
            print("Error: Run 'gcloud auth login' first")
            return results

        # Set project
        if not self.set_project():
            results["errors"].append(f"Failed to set project: {self.config.project_id}")
            return results

        # Get training data stats
        print("\nChecking training data...")
        results["training_data"] = self.get_training_data_stats()
        print(f"  Total examples: {results['training_data']['total_examples']:,}")
        print(f"  Files: {len(results['training_data']['files'])}")

        # Deploy Angular
        url = self.deploy_angular(build=build_angular)
        if url:
            results["angular_url"] = url
            results["success"] = True
        else:
            results["errors"].append("Angular deployment failed")

        return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command line interface for Cloud Run management."""
    import argparse

    parser = argparse.ArgumentParser(description="Erosolar Cloud Run Manager")
    parser.add_argument("command", choices=["deploy", "status", "scale", "delete", "stats"],
                       help="Command to execute")
    parser.add_argument("--project", default="erosolar-prod", help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="Cloud Run region")
    parser.add_argument("--service", default="erosolar-chat", help="Service name")
    parser.add_argument("--no-build", action="store_true", help="Skip Angular build")
    parser.add_argument("--min-instances", type=int, default=0, help="Minimum instances")
    parser.add_argument("--max-instances", type=int, default=10, help="Maximum instances")
    parser.add_argument("--confirm", action="store_true", help="Confirm destructive operations")

    args = parser.parse_args()

    config = CloudRunConfig(
        project_id=args.project,
        region=args.region,
        service_name=args.service,
        min_instances=args.min_instances,
        max_instances=args.max_instances
    )

    manager = CloudRunManager(config)

    if args.command == "deploy":
        results = manager.full_deploy(build_angular=not args.no_build)
        if results["success"]:
            print(f"\n✓ Deployment successful: {results['angular_url']}")
        else:
            print(f"\n✗ Deployment failed: {results['errors']}")
            sys.exit(1)

    elif args.command == "status":
        status = manager.get_service_status()
        print(json.dumps(status, indent=2))

    elif args.command == "scale":
        success = manager.scale_service(args.min_instances, args.max_instances)
        if success:
            print(f"✓ Scaled to {args.min_instances}-{args.max_instances} instances")
        else:
            print("✗ Scaling failed")
            sys.exit(1)

    elif args.command == "delete":
        if manager.delete_service(confirm=args.confirm):
            print("✓ Service deleted")
        else:
            print("✗ Delete failed (use --confirm to actually delete)")
            sys.exit(1)

    elif args.command == "stats":
        stats = manager.get_training_data_stats()
        print(f"\nTraining Data Statistics:")
        print(f"  Total examples: {stats['total_examples']:,}")
        print(f"  Total size: {stats['total_size_bytes'] / 1024 / 1024:.2f} MB")
        print(f"\nFiles:")
        for f in stats["files"]:
            print(f"  - {f['name']}: {f['examples']:,} examples ({f['size_bytes'] / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
