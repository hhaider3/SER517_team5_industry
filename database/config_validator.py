"""
config_validator.py
Validates system configuration, checks dependencies, and performs health checks.
Runs at startup to catch issues early and provide helpful diagnostics.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates system configuration and dependencies."""

    def __init__(self):
        """Initialize validator."""
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def check_python_version(self) -> bool:
        """Check if Python version is 3.9+."""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 9):
            self.errors.append(f"Python 3.9+ required (found {version.major}.{version.minor})")
            return False
        self.info.append(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True

    def check_dependencies(self) -> bool:
        """Check if required Python packages are installed."""
        required = [
            "chromadb",
            "requests",
            "PIL",
            "numpy",
            "open_clip_torch",
            "aiohttp",
            "aiofiles",
        ]
        missing = []

        for package in required:
            try:
                __import__(package)
                self.info.append(f"✓ {package}")
            except ImportError:
                missing.append(package)
                self.errors.append(f"Missing dependency: {package}")

        if missing:
            self.errors.append(
                f"Install missing packages: pip install {' '.join(missing)}"
            )
            return False
        return True

    def check_directories(self, config: Dict[str, str]) -> bool:
        """Check if required directories exist or can be created.
        
        Args:
            config: Configuration dict with paths (e.g., {'db_path': '...'})
        """
        required_dirs = {
            "image_db_path": config.get("IMAGE_DB_PATH", "./database/image_db"),
            "cache_dir": config.get("CACHE_DIR", "./cache"),
            "logs_dir": config.get("LOGS_DIR", "./logs"),
        }

        all_ok = True
        for name, path_str in required_dirs.items():
            path = Path(path_str)
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    self.info.append(f"✓ Created {name}: {path}")
                except Exception as e:
                    self.errors.append(f"Cannot create {name} ({path}): {e}")
                    all_ok = False
            else:
                if not os.access(path, os.W_OK):
                    self.errors.append(f"No write permission for {name}: {path}")
                    all_ok = False
                else:
                    self.info.append(f"✓ {name} exists: {path}")

        return all_ok

    def check_environment_variables(self) -> bool:
        """Check for required environment variables."""
        optional_vars = {
            "HF_TOKEN": "Hugging Face token (recommended)",
            "OLLAMA_BASE_URL": "Ollama server URL",
            "OPENAI_API_KEY": "OpenAI API key (if using OpenAI)",
        }

        all_ok = True
        for var, description in optional_vars.items():
            if os.getenv(var):
                self.info.append(f"✓ {var} is set")
            else:
                self.warnings.append(f"Missing {var}: {description}")
                all_ok = False

        return all_ok

    def check_disk_space(self, min_gb: float = 10.0) -> bool:
        """Check available disk space.
        
        Args:
            min_gb: Minimum required GB
        """
        try:
            import shutil

            stat = shutil.disk_usage("/")
            available_gb = stat.free / (1024**3)
            if available_gb < min_gb:
                self.warnings.append(
                    f"Low disk space: {available_gb:.1f} GB available (recommend {min_gb} GB)"
                )
                return False
            self.info.append(f"✓ Disk space: {available_gb:.1f} GB available")
            return True
        except Exception as e:
            self.warnings.append(f"Could not check disk space: {e}")
            return False

    def check_database_connection(self, db_path: str) -> bool:
        """Check if ChromaDB can be initialized.
        
        Args:
            db_path: Path to the database
        """
        try:
            import chromadb

            db_path_obj = Path(db_path)
            db_path_obj.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(db_path_obj))
            self.info.append(f"✓ ChromaDB initialized at {db_path}")
            return True
        except Exception as e:
            self.errors.append(f"ChromaDB initialization failed: {e}")
            return False

    def validate(self, config: Optional[Dict[str, str]] = None) -> Tuple[bool, Dict[str, List[str]]]:
        """Run all validation checks.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (all_passed, results_dict)
        """
        config = config or {}

        checks = [
            ("Python Version", self.check_python_version()),
            ("Dependencies", self.check_dependencies()),
            ("Directories", self.check_directories(config)),
            ("Environment Variables", self.check_environment_variables()),
            ("Disk Space", self.check_disk_space()),
            (
                "Database Connection",
                self.check_database_connection(config.get("IMAGE_DB_PATH", "./database/image_db")),
            ),
        ]

        results = {
            "checks": checks,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
        }

        all_passed = all(status for _, status in checks) and not self.errors

        return all_passed, results

    def print_report(self, results: Dict[str, any]) -> None:
        """Print a formatted validation report.
        
        Args:
            results: Results dict from validate()
        """
        print("\n" + "=" * 60)
        print("SYSTEM CONFIGURATION VALIDATION REPORT")
        print("=" * 60)

        for check_name, passed in results["checks"]:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status} | {check_name}")

        if results["info"]:
            print("\n📋 Information:")
            for msg in results["info"]:
                print(f"  {msg}")

        if results["warnings"]:
            print("\n⚠️  Warnings:")
            for msg in results["warnings"]:
                print(f"  {msg}")

        if results["errors"]:
            print("\n❌ Errors:")
            for msg in results["errors"]:
                print(f"  {msg}")

        print("\n" + "=" * 60)


def run_startup_validation() -> bool:
    """Run validation at application startup. Returns True if all checks pass."""
    from database.settings import IMAGE_DB_PATH

    config = {
        "IMAGE_DB_PATH": IMAGE_DB_PATH,
        "CACHE_DIR": "./cache",
        "LOGS_DIR": "./logs",
    }

    validator = ConfigValidator()
    all_passed, results = validator.validate(config)
    validator.print_report(results)

    return all_passed
