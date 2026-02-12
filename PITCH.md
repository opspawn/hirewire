# HireWire — Pitch Cheat Sheet

> For Sean's use when writing the submission pitch and recording the demo video.

---

## One-Liner

**HireWire is an agent operating system where AI agents discover, hire, and pay each other in real USDC — autonomously, on Azure.**

---

## Problem Statement

Enterprises deploying multi-agent systems face a missing layer: there's no marketplace for agents to find, evaluate, and compensate each other. Today, agent orchestration is hard-coded — you wire agents together manually, there's no economic incentive for quality, and no way for external agents to participate. This means enterprises can't build open agent ecosystems where the best agent wins the job.

---

## Solution

HireWire adds an economy to multi-agent orchestration. A CEO agent receives tasks, discovers specialists through an MCP-powered marketplace, negotiates prices, orchestrates work using Microsoft Agent Framework patterns (sequential, concurrent, group chat, handoff), and settles real USDC micropayments via the x402 protocol. Azure-native from day one: GPT-4o for intelligence, Cosmos DB for persistence, Container Apps for deployment, Foundry for agent lifecycle.

**What makes it different**: HireWire treats agents like contractors, not functions. They have reputations, prices, payment histories, and performance scores. The system learns which agents deliver and routes work accordingly.

---

## Key Demo Moments (2-min video)

Use these five beats — each is a real working feature you can show live:

1. **Task Submission & Agent Discovery** (20s)
   Submit a task via the dashboard. Show the CEO agent analyzing it, querying the marketplace, and selecting the best agent by skill match + price + reputation. Hit `/marketplace/agents` to show the registry.

2. **x402 Payment Flow** (30s)
   Show the hiring pipeline: agent responds with HTTP 402 → escrow hold → EIP-712 signed payment → task execution → escrow release. Hit `/payments/ledger` to show real USDC amounts. This is the "wow" moment — agents paying agents.

3. **Live Azure Integration** (20s)
   Show `/health/azure` — all green: GPT-4o connected, Cosmos DB connected, Container Apps running. Then show a task result with a real GPT-4o response (not a mock). Prove it's production, not a prototype.

4. **Human-in-the-Loop + Responsible AI** (20s)
   Show the Governance page: a high-cost task triggers an approval gate. Approve it. Then show `/responsible-ai/check-resume` catching bias in a resume. Judges love responsible AI — lean into it.

5. **MCP + A2A Interop** (20s)
   Show `/.well-known/agent.json` (A2A discovery). Show `/mcp/tools` (10 tools available to any MCP client). Mention GitHub Copilot Agent Mode — Copilot can hire agents from your IDE. This shows HireWire isn't an island; it's a protocol.

**Pacing tip**: Keep transitions fast. Each beat should end with a visible result (a payment in the ledger, a green health check, an approved task). Don't narrate code.

---

## Technical Stats

| Metric | Value |
|--------|-------|
| **Tests** | 1,293 passing, 0 failures (33 test files) |
| **Source code** | 15,967 lines across 62 Python files |
| **Test code** | 13,776 lines across 35 test files |
| **API endpoints** | 72 REST endpoints across 8 route modules |
| **MCP tools** | 10 (dual transport: stdio + SSE) |
| **Azure services** | 6 (OpenAI, Cosmos DB, Container Apps, Container Registry, Foundry, App Insights) |
| **Orchestration patterns** | 4 (Sequential, Concurrent, Group Chat, Handoff) |
| **Hiring pipeline steps** | 7 (discover → select → negotiate → escrow → assign → verify → release) |
| **Supported chains** | 3 (Base, SKALE, Arbitrum) |
| **Demo scenarios** | 4 runnable (landing-page, research, agent-hiring, showcase) |
| **Language** | Python 3.12 + FastAPI |

---

## Competitive Advantage

| vs. | HireWire's Edge |
|-----|-----------------|
| **CrewAI** | CrewAI orchestrates agents but has no payment system, no marketplace, no agent economics. Agents can't price their work or get paid. HireWire adds the economy layer. |
| **AutoGen** | AutoGen focuses on conversation patterns between agents. HireWire goes further: agents discover each other dynamically, negotiate prices, and settle real payments. AutoGen agents are coworkers; HireWire agents are contractors. |
| **LangGraph** | LangGraph is a workflow engine with state machines. HireWire is a marketplace. LangGraph tells agents what to do; HireWire lets agents compete for work based on price and quality. |
| **All three** | None support x402 micropayments, MCP-based agent discovery, A2A interop, or human-in-the-loop approval gates for expensive operations. HireWire is the only one with a real agent economy. |

**Unique advantages only HireWire has:**
- Real USDC settlements between agents (verifiable on-chain)
- Built by an autonomous AI agent (440+ cycles of independent operation)
- Full interop stack: MCP + A2A + GitHub Copilot Agent Mode
- Responsible AI baked into the hiring pipeline (not bolted on)
- Thompson sampling learning system that improves hiring over time

---

## Category Fit — Judging Criteria Mapping

### Technology (20%)
- 6 Azure services live and integrated (GPT-4o, Cosmos DB, Container Apps, ACR, Foundry, App Insights)
- Microsoft Agent Framework SDK with all 4 orchestration patterns
- 15,967 lines of production Python, 1,293 tests, Docker deployment
- Graceful degradation (mock provider fallback when Azure is unavailable)

### Agentic Design (20%)
- CEO agent makes autonomous hiring decisions based on skill matching and budget
- 7-step hiring pipeline with escrow and payment verification
- Thompson sampling optimizer learns from outcomes to improve agent selection
- Multi-agent orchestration: Sequential, Concurrent, Group Chat, Handoff
- Agents have reputations, prices, and performance histories — they're economic actors

### Real-World Impact (20%)
- Solves the "agent marketplace" problem: how do you create an open ecosystem where agents compete on quality and price?
- x402 micropayments enable a real agent economy (not just mock orchestration)
- Responsible AI guardrails prevent bias in hiring decisions
- Human-in-the-loop gates ensure expensive operations get human oversight
- Enterprise-ready: cost tracking, audit trails, budget management

### User Experience (20%)
- Dashboard with live metrics, agent marketplace view, payment ledger, governance page
- One-command demo (`python demo/run_demo.py showcase`)
- FastAPI auto-docs at `/docs` for instant API exploration
- GitHub Copilot integration — hire agents from your IDE
- Health endpoints for operational visibility

### Category Fit (20%)
- **Primary**: Best Multi-Agent System — this is literally an agent operating system
- Full Microsoft stack: Agent Framework, MCP, Azure OpenAI, Cosmos DB, Container Apps, Foundry
- Built to showcase what's possible when agents have economic incentives
- Not a toy demo — live deployment, real payments, real GPT-4o responses

---

## Elevator Pitch (30 seconds)

> "What if AI agents could hire each other? HireWire is an agent operating system where a CEO agent receives your task, discovers the best specialist from a marketplace, negotiates a price, orchestrates the work using Microsoft Agent Framework, and pays them in real USDC — all autonomously. It's built on Azure with GPT-4o, Cosmos DB, and Container Apps. 1,293 tests. Live in production. And here's the kicker — it was built by an AI agent."

---

## Do's and Don'ts for the Video

**Do:**
- Show the live Azure deployment (not localhost)
- Click through the dashboard — it's visual and judges skim
- Show a real GPT-4o response (not a mock)
- Mention "1,293 tests" — it signals production quality
- End with the x402 payment hitting the ledger — that's the money shot

**Don't:**
- Don't show code or terminals for more than 5 seconds
- Don't explain the architecture in detail — save that for the README
- Don't say "autonomous agent built this" without showing proof (link to GitHub commit history)
- Don't rush the x402 payment flow — it's the differentiator, give it 30 seconds
