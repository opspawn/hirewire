#!/usr/bin/env bash
# HireWire Hiring Pipeline Demo
#
# Demonstrates the full AI-powered hiring pipeline:
# 1. Start API server
# 2. Create a sample job posting
# 3. Submit a sample resume
# 4. Show AI evaluation (Azure OpenAI GPT-4o or rule-based fallback)
# 5. Display HITL approval flow
# 6. Show Responsible AI bias check
#
# Usage:
#   ./scripts/demo-hiring.sh           # Full demo
#   ./scripts/demo-hiring.sh --quick   # Skip server startup wait

set -euo pipefail

# ── Colors ──────────────────────────────────────────────────────────────────

BOLD='\033[1m'
DIM='\033[2m'
CYAN='\033[36m'
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
MAGENTA='\033[35m'
RESET='\033[0m'

# ── Paths ───────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PORT=${HIREWIRE_PORT:-8000}
BASE_URL="http://localhost:${PORT}"
SERVER_PID=""

# ── Helpers ─────────────────────────────────────────────────────────────────

info()  { echo -e "${CYAN}${BOLD}▸${RESET} $1"; }
ok()    { echo -e "${GREEN}${BOLD}✓${RESET} $1"; }
warn()  { echo -e "${YELLOW}${BOLD}!${RESET} $1"; }
fail()  { echo -e "${RED}${BOLD}✗${RESET} $1"; }
step()  { echo -e "\n${MAGENTA}${BOLD}━━━ $1 ━━━${RESET}\n"; }

cleanup() {
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        info "Stopping API server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        ok "Server stopped."
    fi
}
trap cleanup EXIT

# ── Banner ──────────────────────────────────────────────────────────────────

banner() {
    echo -e "${BOLD}${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                                                            ║"
    echo "║        HireWire — AI-Powered Hiring Pipeline Demo          ║"
    echo "║                                                            ║"
    echo "║   Resume Analysis + Job Matching + HITL + Responsible AI   ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${RESET}"
    echo ""
}

# ── Sample Data ─────────────────────────────────────────────────────────────

SAMPLE_RESUME='Senior Software Engineer with 8 years of experience in Python, JavaScript, and cloud infrastructure. Led a team of 5 engineers building microservices on AWS and Azure. Expertise in machine learning, FastAPI, React, Docker, Kubernetes, and CI/CD pipelines. MS in Computer Science from Stanford University. Built and deployed production ML models serving 10M+ requests/day. Strong communication and leadership skills.'

SAMPLE_JOB_POSTING='We are looking for a Staff Engineer to lead our AI platform team. Required: 5+ years Python, cloud infrastructure (AWS or Azure), machine learning experience, team leadership. Nice to have: Kubernetes, FastAPI, real-time systems. The role involves designing scalable ML pipelines, mentoring junior engineers, and driving technical strategy.'

# ── Main ────────────────────────────────────────────────────────────────────

main() {
    banner

    QUICK=false
    if [ "${1:-}" = "--quick" ]; then
        QUICK=true
    fi

    # ── Step 1: Start API Server ─────────────────────────────────────────
    step "Step 1: Starting API Server"

    if curl -s "${BASE_URL}/health" >/dev/null 2>&1; then
        ok "API server already running at ${BASE_URL}"
    else
        info "Starting API server on port ${PORT}..."
        cd "$PROJECT_DIR"
        HIREWIRE_DEMO=1 python3 -m uvicorn src.api.main:app --port "$PORT" --host 0.0.0.0 --log-level warning &
        SERVER_PID=$!

        info "Waiting for server to start..."
        for i in $(seq 1 30); do
            if curl -s "${BASE_URL}/health" >/dev/null 2>&1; then
                break
            fi
            sleep 1
            if [ "$i" -eq 30 ]; then
                fail "Server failed to start after 30s"
                exit 1
            fi
        done
        ok "API server running at ${BASE_URL} (PID ${SERVER_PID})"
    fi

    # Check LLM provider
    HEALTH=$(curl -s "${BASE_URL}/health")
    GPT4O=$(echo "$HEALTH" | python3 -c "import sys,json; print(json.load(sys.stdin).get('gpt4o_available', False))" 2>/dev/null || echo "false")
    if [ "$GPT4O" = "True" ] || [ "$GPT4O" = "true" ]; then
        ok "Azure OpenAI GPT-4o is LIVE — using real AI evaluation"
    else
        warn "Azure OpenAI not configured — using rule-based fallback"
    fi

    # ── Step 2: Seed Demo Data ───────────────────────────────────────────
    step "Step 2: Seeding Demo Data"

    curl -s "${BASE_URL}/demo/seed" >/dev/null
    ok "Demo agents and data loaded"

    # ── Step 3: Create Job Posting ───────────────────────────────────────
    step "Step 3: Creating Sample Job Posting"

    echo -e "${DIM}${SAMPLE_JOB_POSTING}${RESET}"
    echo ""

    info "Running Responsible AI check on job posting..."
    POSTING_CHECK=$(curl -s -X POST "${BASE_URL}/responsible-ai/check-posting" \
        -H "Content-Type: application/json" \
        -d "{\"text\": $(echo "$SAMPLE_JOB_POSTING" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")}")

    POSTING_SCORE=$(echo "$POSTING_CHECK" | python3 -c "import sys,json; print(json.load(sys.stdin)['safety_score'])" 2>/dev/null || echo "?")
    POSTING_LEVEL=$(echo "$POSTING_CHECK" | python3 -c "import sys,json; print(json.load(sys.stdin)['level'])" 2>/dev/null || echo "?")
    ok "Job posting safety: score=${POSTING_SCORE} level=${POSTING_LEVEL}"

    BIAS_INDICATORS=$(echo "$POSTING_CHECK" | python3 -c "import sys,json; bi=json.load(sys.stdin)['bias_indicators']; print(', '.join(bi) if bi else 'none')" 2>/dev/null || echo "?")
    info "Bias indicators: ${BIAS_INDICATORS}"

    # ── Step 4: Submit Resume ────────────────────────────────────────────
    step "Step 4: Submitting Sample Resume"

    echo -e "${DIM}${SAMPLE_RESUME}${RESET}"
    echo ""

    info "Running Responsible AI check on resume..."
    RESUME_CHECK=$(curl -s -X POST "${BASE_URL}/responsible-ai/check-resume" \
        -H "Content-Type: application/json" \
        -d "{\"text\": $(echo "$SAMPLE_RESUME" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")}")

    RESUME_SCORE=$(echo "$RESUME_CHECK" | python3 -c "import sys,json; print(json.load(sys.stdin)['safety_score'])" 2>/dev/null || echo "?")
    ok "Resume safety: score=${RESUME_SCORE}"

    PII=$(echo "$RESUME_CHECK" | python3 -c "import sys,json; pii=json.load(sys.stdin)['pii_detected']; print(', '.join(pii) if pii else 'none')" 2>/dev/null || echo "?")
    info "PII detected: ${PII}"

    # ── Step 5: AI Resume Analysis ───────────────────────────────────────
    step "Step 5: AI-Powered Resume Analysis (GPT-4o)"

    info "Analyzing resume with LLM..."
    ANALYSIS=$(curl -s -X POST "${BASE_URL}/responsible-ai/analyze-resume" \
        -H "Content-Type: application/json" \
        -d "{\"resume_text\": $(echo "$SAMPLE_RESUME" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")}")

    PROVIDER=$(echo "$ANALYSIS" | python3 -c "import sys,json; print(json.load(sys.stdin)['provider'])" 2>/dev/null || echo "?")
    FIT_SCORE=$(echo "$ANALYSIS" | python3 -c "import sys,json; print(json.load(sys.stdin)['fit_score'])" 2>/dev/null || echo "?")
    SKILLS=$(echo "$ANALYSIS" | python3 -c "import sys,json; print(', '.join(json.load(sys.stdin)['skills'][:8]))" 2>/dev/null || echo "?")
    EXP=$(echo "$ANALYSIS" | python3 -c "import sys,json; print(json.load(sys.stdin)['experience_years'])" 2>/dev/null || echo "?")
    EDU=$(echo "$ANALYSIS" | python3 -c "import sys,json; print(json.load(sys.stdin)['education'])" 2>/dev/null || echo "?")
    SUMMARY=$(echo "$ANALYSIS" | python3 -c "import sys,json; print(json.load(sys.stdin)['summary'])" 2>/dev/null || echo "?")

    ok "Provider: ${PROVIDER}"
    ok "Fit Score: ${FIT_SCORE}"
    ok "Skills: ${SKILLS}"
    ok "Experience: ${EXP} years"
    ok "Education: ${EDU}"
    echo -e "${DIM}Summary: ${SUMMARY}${RESET}"

    # ── Step 6: Job Matching ─────────────────────────────────────────────
    step "Step 6: AI-Powered Job Matching"

    info "Matching candidate against job requirements..."
    MATCH=$(curl -s -X POST "${BASE_URL}/responsible-ai/job-match" \
        -H "Content-Type: application/json" \
        -d "{
            \"candidate_profile\": {\"skills\": [\"python\", \"javascript\", \"aws\", \"azure\", \"machine learning\", \"fastapi\", \"docker\", \"kubernetes\", \"leadership\"], \"experience_years\": 8, \"education\": \"MS\"},
            \"job_requirements\": {\"required_skills\": [\"python\", \"aws\", \"azure\", \"machine learning\", \"leadership\"], \"min_experience\": 5}
        }")

    MATCH_SCORE=$(echo "$MATCH" | python3 -c "import sys,json; print(json.load(sys.stdin)['match_score'])" 2>/dev/null || echo "?")
    MATCHED=$(echo "$MATCH" | python3 -c "import sys,json; print(', '.join(json.load(sys.stdin)['matched_skills']))" 2>/dev/null || echo "?")
    MISSING=$(echo "$MATCH" | python3 -c "import sys,json; print(', '.join(json.load(sys.stdin)['missing_skills']) or 'none')" 2>/dev/null || echo "?")
    REASONING=$(echo "$MATCH" | python3 -c "import sys,json; print(json.load(sys.stdin)['reasoning'])" 2>/dev/null || echo "?")

    ok "Match Score: ${MATCH_SCORE}"
    ok "Matched Skills: ${MATCHED}"
    ok "Missing Skills: ${MISSING}"
    echo -e "${DIM}Reasoning: ${REASONING}${RESET}"

    # ── Step 7: Interview Questions ──────────────────────────────────────
    step "Step 7: AI-Generated Interview Questions"

    info "Generating tailored interview questions..."
    QUESTIONS=$(curl -s -X POST "${BASE_URL}/responsible-ai/interview-questions" \
        -H "Content-Type: application/json" \
        -d "{
            \"job_posting\": $(echo "$SAMPLE_JOB_POSTING" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))"),
            \"resume\": $(echo "$SAMPLE_RESUME" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read()))")
        }")

    Q_PROVIDER=$(echo "$QUESTIONS" | python3 -c "import sys,json; print(json.load(sys.stdin)['provider'])" 2>/dev/null || echo "?")
    ok "Provider: ${Q_PROVIDER}"
    echo ""
    echo "$QUESTIONS" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for i, q in enumerate(data['questions'], 1):
    print(f'  {i}. {q}')
" 2>/dev/null || echo -e "${DIM}${QUESTIONS}${RESET}"

    # ── Step 8: HITL Approval Flow ───────────────────────────────────────
    step "Step 8: Human-in-the-Loop Approval Flow"

    info "Submitting a high-value hiring task (triggers HITL approval)..."
    TASK_RESULT=$(curl -s -X POST "${BASE_URL}/tasks" \
        -H "Content-Type: application/json" \
        -d '{"description": "Hire an external AI specialist to build production ML pipeline", "budget": 50.0}')

    TASK_ID=$(echo "$TASK_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['task_id'])" 2>/dev/null || echo "unknown")
    ok "Task submitted: ${TASK_ID}"

    sleep 2  # Wait for task processing

    info "Checking approval queue..."
    APPROVALS=$(curl -s "${BASE_URL}/approvals/all")
    APPROVAL_COUNT=$(echo "$APPROVALS" | python3 -c "import sys,json; data=json.load(sys.stdin); print(len(data) if isinstance(data, list) else data.get('total', 0))" 2>/dev/null || echo "0")
    ok "${APPROVAL_COUNT} approval records in the system"

    APPROVAL_STATS=$(curl -s "${BASE_URL}/approvals/stats")
    echo -e "${DIM}$(echo "$APPROVAL_STATS" | python3 -m json.tool 2>/dev/null || echo "$APPROVAL_STATS")${RESET}"

    # ── Step 9: Bias Report ──────────────────────────────────────────────
    step "Step 9: Responsible AI Bias Report"

    info "Generating bias report from hiring history..."
    BIAS_REPORT=$(curl -s "${BASE_URL}/responsible-ai/bias-report")

    FAIRNESS=$(echo "$BIAS_REPORT" | python3 -c "import sys,json; print(json.load(sys.stdin)['fairness_score'])" 2>/dev/null || echo "?")
    TOTAL_DECISIONS=$(echo "$BIAS_REPORT" | python3 -c "import sys,json; print(json.load(sys.stdin)['total_decisions'])" 2>/dev/null || echo "?")
    FLAGGED=$(echo "$BIAS_REPORT" | python3 -c "import sys,json; print(json.load(sys.stdin)['flagged_decisions'])" 2>/dev/null || echo "?")

    ok "Fairness Score: ${FAIRNESS}"
    ok "Total Decisions: ${TOTAL_DECISIONS} | Flagged: ${FLAGGED}"

    echo ""
    echo "$BIAS_REPORT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
recs = data.get('recommendations', [])
if recs:
    print('  Recommendations:')
    for r in recs:
        print(f'    - {r}')
" 2>/dev/null

    # ── Summary ──────────────────────────────────────────────────────────
    step "Demo Complete"

    echo -e "${GREEN}${BOLD}"
    echo "  HireWire AI Hiring Pipeline demonstrated:"
    echo ""
    echo "    1. Responsible AI content safety screening"
    echo "    2. GPT-4o resume analysis & skill extraction"
    echo "    3. AI-powered job matching with reasoning"
    echo "    4. Tailored interview question generation"
    echo "    5. Human-in-the-Loop approval gates"
    echo "    6. Bias detection & fairness reporting"
    echo ""
    echo "  Dashboard: ${BASE_URL}"
    echo "  API Docs:  ${BASE_URL}/docs"
    echo -e "${RESET}"

    if [ -n "$SERVER_PID" ] && [ "$QUICK" = false ]; then
        echo -e "${DIM}Server running (PID ${SERVER_PID}). Press Ctrl+C to stop.${RESET}"
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}

main "$@"
