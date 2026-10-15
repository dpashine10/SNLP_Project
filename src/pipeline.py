"""
Entry point for ResearchDiscoveryPipeline.
Redirects to src/week2/pipeline.py for backward compatibility.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from src.week2.pipeline import ResearchDiscoveryPipeline, main

if __name__ == "__main__":
  main()
