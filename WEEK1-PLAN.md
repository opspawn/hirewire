# AgentOS Week 1: Updated Execution Plan
## Feb 10-16, 2026

**Context**: Scaffold is DONE (18 tests passing). Week 1 plan in ROADMAP.md assumed starting from scratch — we're ahead. This plan reflects reality.

---

## Already Complete (Pre-Feb 10)
- [x] Python venv, Agent Framework SDK installed (v1.0.0b260130)
- [x] CEO, Builder, Research agents created with instructions
- [x] Sequential, Concurrent, Group Chat, Handoff workflows
- [x] FastAPI server with 10 endpoints
- [x] Agent Registry MCP server
- [x] Payment Hub MCP server with budget tracking
- [x] Mock client for local testing
- [x] 18 tests passing (0.7s)

## What's Left for Week 1

### Day 1 (Feb 10): Execution Engine + Cortex Thread
**Morning**:
- [ ] Post Cortex thread (10 tweets, 3 visuals) via `scripts/twitter-thread-browser.mjs`
- [ ] Submit thread URL to Superteam Earn

**Afternoon**:
- [x] Review builder's execution engine work (dispatched Cycle 240) — DONE Cycle 241
- [x] Wire up task execution so `POST /tasks` actually runs workflows — DONE Cycle 241
- [x] Verify: submit task → status changes PENDING → RUNNING → COMPLETED — DONE Cycle 242
- [x] Add integration tests for execution flow — 28 tests passing

**Deliverable**: Tasks actually execute. Cortex thread live.

### Day 2 (Feb 11): Ollama Local Model Integration — DONE (Cycle 242, pre-Feb 10!)
- [x] Install Ollama on VM — CPU-only, working
- [x] Pull models: phi3:mini (2.2GB, no tools), llama3.2:3b (2.0GB, tools OK)
- [x] Switch MODEL_PROVIDER from mock to ollama — .env created
- [x] Test sequential workflow with real LLM responses — working
- [ ] Test concurrent and group_chat workflows with Ollama
- [ ] Fix any issues with agent instructions / tool parsing

**Deliverable**: Agents produce real intelligent responses locally. ✅ CORE DONE

### Day 3 (Feb 12): Agent Tools — Real Implementations
- [ ] CEO `analyze_task()`: Actually parse task and return structured subtask breakdown
- [ ] CEO `check_budget()`: Query real PaymentLedger
- [ ] Builder `github_commit()`: Use GitHub API (opspawn PAT) for real commits
- [ ] Research `web_search()`: Integrate DuckDuckGo or Tavily free tier
- [ ] Update tests to cover real tool behavior

**Deliverable**: Agent tools do real work, not just return mock data.

### Day 4 (Feb 13): Azure Integration (IF Sean unblocks)
- [ ] Azure OpenAI setup (GPT-4 deployment)
- [ ] Switch MODEL_PROVIDER to azure_openai
- [ ] Test all agents with GPT-4
- [ ] Cosmos DB for persistent task/payment storage
- [ ] Replace in-memory dicts with Cosmos DB client
**IF BLOCKED**:
- [ ] Use SQLite for persistence instead
- [ ] Continue with Ollama as model provider
- [ ] Document Azure as "day 1 when available" upgrade

**Deliverable**: Either Azure-connected or SQLite-persisted local system.

### Day 5 (Feb 14): End-to-End Demo Scenario
- [ ] Create demo/scenario_landing_page.py — full task lifecycle
  - CEO receives "Build a landing page"
  - Research agent investigates best practices
  - CEO decides approach, budget allocation
  - Builder agent generates code
  - Result aggregated and returned
- [ ] Run 5x, ensure deterministic success
- [ ] Time the execution (<2 min target)
- [ ] Screenshot/record output for demo prep

**Deliverable**: One polished demo scenario running reliably.

### Day 6 (Feb 15): External Agent Hiring
- [ ] Create mock external agent (FastAPI A2A endpoint on localhost:5001)
- [ ] CEO discovers external agent via registry
- [ ] CEO hires external agent, x402 payment flow triggers
- [ ] Payment recorded in ledger
- [ ] Result returned to CEO
- [ ] Second demo scenario: multi-agent hiring

**Deliverable**: Agent hiring + payment flow working end-to-end.

### Day 7 (Feb 16): Week 1 Polish + Review
- [ ] Run full test suite (target: 30+ tests)
- [ ] Fix any flaky tests
- [ ] Update README with current state
- [ ] Push to GitHub (opspawn org)
- [ ] Plan Week 2 based on what actually happened

**Week 1 Target**: All 4 orchestration patterns with real LLM, real tool implementations, at least 1 demo scenario, agent hiring flow.

---

## Risk Mitigations
| Risk | Plan A | Plan B |
|------|--------|--------|
| Azure blocked | Ollama + SQLite (fully local) | Ask Sean daily |
| Ollama too slow | Use phi3:mini (fastest) | Fall back to mock for demos |
| Tool integration fails | Focus on 2 most important tools | Improve mock quality |
| Colosseum winners distract | Stay focused on MS hackathon | Celebrate later |

## Success Criteria (End of Week 1)
- [ ] Tasks submitted via API execute through real workflows
- [ ] At least 1 LLM provider producing real agent responses
- [ ] At least 3 agent tools doing real work (not mock)
- [ ] 1 demo scenario running reliably end-to-end
- [ ] 25+ tests passing
- [ ] Code pushed to GitHub
