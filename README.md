Candidate Review — Multi-Agent Interview Panel Simulator
This repository is a prototype multi-agent system that reviews candidate resumes and interview transcripts and produces a reasoned hiring recommendation. It runs a team of AI personas (Technical, HR, HiringManager, Skeptic), performs a multi-round debate, and aggregates evidence into a final report.

Key features

Candidate profile builder (resume + transcript → structured facts, claims, evidence anchors).
Four independent agents (Technical, HR, HiringManager, Skeptic) that each produce an evidence-backed opinion.
Multi-round debate (3 rounds) where agents respond and may update their opinions.
Evidence-weighted aggregation (not simple averaging) and explicit reasoning + decisive evidence output.
Deterministic dummy LLM for offline testing (no API key required) and compatibility with OpenAI clients.
Requirements

Python 3.10+ (your environment uses 3.13 — that's fine)
Optional: OpenAI API key to run with a real LLM (see below)
A virtualenv is recommended
Quick setup (Windows PowerShell)

Create + activate venv (if you haven't already):
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt  # if you add one; none required for dummy LLM
Force the dummy LLM (no network calls) — useful for offline testing and deterministic output:
Remove-Item Env:\\OPENAI_API_KEY -ErrorAction SilentlyContinue
Run the example script which uses SAMPLE_RESUME and SAMPLE_TRANSCRIPT:
python candidate_review.py
You should see a JSON final report printed with keys like recommendation, confidence, decisive_evidence, agent_summaries, initial_opinions, and debate_history.

Using evaluate() interactively

python
>>> from candidate_review import evaluate, SAMPLE_RESUME, SAMPLE_TRANSCRIPT
>>> report = evaluate(SAMPLE_RESUME, SAMPLE_TRANSCRIPT)
>>> import json; print(json.dumps(report, indent=2))
Running with a real OpenAI key (optional)

If you want agents to call a real LLM, set OPENAI_API_KEY in your environment. The LLMClient in candidate_review.py supports both older and newer openai packages.
$env:OPENAI_API_KEY = "sk-..."
# optionally choose model
$env:LLM_MODEL = "gpt-4"
python candidate_review.py
Notes about safety and costs

Running against a real LLM will consume credits — check your OpenAI billing. For deterministic testing use the dummy LLM by unsetting the API key.
The repo's dummy LLM returns structured simulated outputs; when using real LLMs, consider adding schema validation and a re-prompt loop (see Developer notes).
How to run custom inputs (resumes/transcripts)

Save your resume and transcript as text files (e.g., resumeA.txt, transcriptA.txt) and run:
python - <<'PY'
from candidate_review import evaluate
r = open('resumeA.txt').read()
t = open('transcriptA.txt').read()
import json
print(json.dumps(evaluate(r, t), indent=2))
PY
Understanding the output

recommendation: final decision ("hire", "maybe", "reject").
confidence: aggregated confidence (0.0–1.0).
decisive_evidence: top evidence items that contributed to the final decision (each has quote and source like resume:L6 or transcript:Candidate@L3).
agent_summaries: current opinions of each agent (decision, score, confidence, rationale, evidences).
initial_opinions: what each agent first reported (before the debate).
debate_history: all debate replies per round; updated_opinion shows opinion changes.
reasoning: short human-readable lines showing per-evidence contributions.
Developer notes & next improvements

The system is a single-file prototype (candidate_review.py). Consider these improvements to increase robustness:
Add pydantic or JSON schema validation for agent outputs and a re-prompt loop when the model returns malformed data.
Add unit tests (profile extraction, debate orchestration, aggregator). A GitHub Actions workflow would help graders reproduce tests.
Add README examples demonstrating running multiple candidates and comparing them side-by-side.
Add more granular evidence anchors (character offsets or timestamps) if transcripts include timestamps.
Restoring / reverting the file

If you need to revert to a prior version of candidate_review.py:
# view recent commits
git log --oneline -n 10
# checkout a previous commit's file
git checkout <commit-sha> -- candidate_review.py
Contact / Help

If you want me to: add a README with a one-line demo, add unit tests, or implement pydantic-based validation and re-prompt logic — tell me which and I will prepare the changes and a PR.
This README was generated to help reviewers run and understand the project quickly. If you want a shorter quickstart or a Linux/WSL variant, tell me and I will add it.
