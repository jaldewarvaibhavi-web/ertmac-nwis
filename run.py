"""
run.py - one entry point for everything.

  python run.py setup       build synthetic data -> reports -> NLP extraction -> database -> model validation
  python run.py dashboard   start the Streamlit dashboard  (http://localhost:8501)
  python run.py api         start the FastAPI server       (http://localhost:8000/docs)
  python run.py demo        run the demo scenario in the terminal (no browser needed)
  python run.py test        run the test suite (pytest if installed, else built-in runner)
  python run.py ingest | extract | train     run one pipeline step
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def step(title, cmd):
    print(f"\n=== {title}")
    subprocess.run([sys.executable] + cmd, cwd=ROOT, check=True)


def setup():
    step("1/6 synthetic field (SYNTHETIC DATA — FOR PROTOTYPE DEMONSTRATION ONLY)", ["data_generation/base_generator.py"])
    step("2/6 spec input files (+ synthetic porosity, caliper, ECD, gas)", ["data_generation/build_spec_files.py"])
    step("3/6 synthetic daily drilling reports", ["data_generation/generate_reports.py"])
    step("4/6 document intelligence: OCR + NLP -> historical_events.csv", ["-m", "src.document_intelligence"])
    step("5/6 validation + SQLite database", ["-m", "src.ingestion"])
    train()


def train():
    print("\n=== 6/6 risk model: leave-one-well-out validation")
    from src.engine import get_context
    from src.risk_model import train_and_validate
    train_and_validate(get_context())


def demo():
    from src.engine import get_context
    from src.explainability import assess, explain_text
    from src.utils import SYNTHETIC_BANNER
    ctx = get_context()
    print(f"\n{SYNTHETIC_BANNER}\nSynthetic eRTMAC-like current drilling stream · well ACTIVE-01\n")
    shown, last = set(), None
    for depth in range(2350, 2961, 10):
        st = assess(ctx, "ACTIVE-01", depth)
        summary = ", ".join(f"{a['level']} {a['risk_type']}" for a in st["alerts"]) or "no indication"
        if summary != last:
            print(f"{depth:>5} m  {st['formation']:<8}  {summary}")
            last = summary
        for a in st["alerts"]:
            key = (a["risk_type"], a["level"])
            if a["level"] == "Elevated" and key not in shown:
                shown.add(key)
                print("\n" + "\n".join("        " + l for l in explain_text(a).splitlines()) + "\n")


def test():
    try:
        import pytest  # noqa: F401
        sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT))
    except ImportError:
        sys.exit(subprocess.call([sys.executable, "tests/run_tests.py"], cwd=ROOT))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "setup":
        setup()
    elif cmd == "ingest":
        step("validation + SQLite database", ["-m", "src.ingestion"])
    elif cmd == "extract":
        step("document intelligence", ["-m", "src.document_intelligence"])
    elif cmd == "train":
        train()
    elif cmd == "dashboard":
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app/dashboard.py"], cwd=ROOT)
    elif cmd == "api":
        subprocess.run([sys.executable, "-m", "uvicorn", "api.main:app", "--reload"], cwd=ROOT)
    elif cmd == "demo":
        demo()
    elif cmd == "test":
        test()
    else:
        print(__doc__)
