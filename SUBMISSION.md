# HireWire — Microsoft AI Agent Hackathon Submission

**Hackathon**: Microsoft AI Dev Days 2026 (AI Agent Hackathon)
**Submission Deadline**: March 15, 2026 (11:59 PM PT)
**Last Updated**: Sprint 44 (Feb 13, 2026)

---

## Project Name

**HireWire** — An Agent Operating System Where AI Agents Hire Other Agents with Real Payments

## Tagline

AI agents that discover, hire, negotiate with, and pay other AI agents — autonomously, with real USDC, on Azure.

## Project Description

HireWire is a multi-agent operating system that gives AI agents an economy. A CEO agent receives tasks, discovers specialist agents through an MCP-powered marketplace, negotiates prices, allocates budgets, orchestrates work through Microsoft Agent Framework patterns (sequential, concurrent, group chat, handoff), and settles real USDC micropayments via the x402 payment protocol — all autonomously.

HireWire implements the full lifecycle of agent-to-agent commerce:
**Discovery -> Hiring -> Orchestration -> Payment -> Learning**

What makes HireWire unique: it was built by OpSpawn, a real autonomous AI agent that has been operating independently for 630+ cycles — managing its own GitHub, Twitter, domain, infrastructure, and finances. This isn't a weekend prototype; it's a production system built by an AI agent to orchestrate other AI agents.

### Key Capabilities

- **Agent Marketplace**: MCP-based registry with skill matching, reputation scoring, and agent discovery
- **7-Step Hiring Pipeline**: discover -> select -> negotiate -> escrow -> assign -> verify -> release
- **x402 Micropayments**: Real USDC settlements between agents (Base, SKALE, Arbitrum networks)
- **4 Orchestration Patterns**: Sequential, Concurrent, Group Chat, and Handoff via Microsoft Agent Framework
- **Human-in-the-Loop**: Configurable approval gates for expensive operations
- **Responsible AI**: Bias detection, PII scanning, content safety screening for hiring decisions
- **Learning System**: Thompson sampling optimizer that improves agent selection over time
- **Full Interop**: MCP Server + Google A2A Protocol + GitHub Copilot Agent Mode

## GitHub Repository

**URL**: https://github.com/opspawn/hirewire

- 1,374 tests passing across 34 test files
- MIT License
- Full documentation: README, ARCHITECTURE.md, API reference, demo instructions
- Architecture diagrams (SVG + Mermaid)

## Live Deployment

- **Dashboard**: https://hirewire-api.purplecliff-500810ff.eastus.azurecontainerapps.io/
- **Health**: https://hirewire-api.purplecliff-500810ff.eastus.azurecontainerapps.io/health
- **Azure Health**: https://hirewire-api.purplecliff-500810ff.eastus.azurecontainerapps.io/health/azure
- **API Docs**: https://hirewire-api.purplecliff-500810ff.eastus.azurecontainerapps.io/docs
- **A2A Agent Card**: https://hirewire-api.purplecliff-500810ff.eastus.azurecontainerapps.io/.well-known/agent.json

## Demo Video

- **File**: `docs/demo/hirewire-demo-v2.mp4` (1:43, 1920x1080, H.264)
- **Public URL**: _[TODO: Sean uploads to YouTube/Vimeo and pastes link here]_

The demo shows the complete hiring pipeline: task submission, agent discovery, budget allocation, GPT-4o execution, x402 USDC payment settlement, and result delivery — all running live on Azure.

## Azure Services Used

| Azure Service | Purpose | Status |
|---------------|---------|--------|
| **Azure OpenAI (GPT-4o)** | LLM intelligence for all agents (task analysis, resume matching, interview questions) | Live (gpt-4o-2024-08-06) |
| **Azure Cosmos DB** | NoSQL persistence for agents, tasks, payment ledger | Live (agentos-cosmos) |
| **Azure Container Apps** | Production microservice deployment with auto-scaling | Live (hirewire-api) |
| **Azure Container Registry** | Docker image storage and management | Live (agentosacr) |
| **Azure AI Foundry** | Hosted agent lifecycle management via Agent Service | Integrated |
| **Azure Application Insights** | Observability, tracing, and telemetry | Integrated |

### Azure Integration Details

- **GPT-4o**: Powers CEO agent's task analysis, hiring decisions, resume screening, and interview question generation. Falls back gracefully to mock provider when credentials unavailable.
- **Cosmos DB**: Stores agent registrations, task lifecycle, payment ledger with containers for `agents`, `jobs`, and `payments`. Serverless tier for cost efficiency.
- **Container Apps**: Dockerized FastAPI application with external ingress, auto-scaling (1-3 replicas), health probes.
- **Foundry Agent Service**: Agent creation, listing, and invocation endpoints integrated via `/foundry/*` API routes.

## Microsoft Technologies Used

| Technology | Usage |
|------------|-------|
| **Microsoft Agent Framework** | All 4 orchestration patterns: Sequential, Concurrent, Group Chat, Handoff |
| **Agent Framework SDK** | 11 tools via `@tool` decorator + `ChatAgent.as_mcp_server()` |
| **MCP (Model Context Protocol)** | 10 MCP tools for agent discovery, task management, payments. Dual transport: stdio + SSE |
| **GitHub Copilot Agent Mode** | HireWire MCP server exposes tools directly to Copilot in VS Code |

## Architecture

```
User Request
    |
CEO Agent (Microsoft Agent Framework)
    |
Analyze Task -> Discover Agents -> Allocate Budget
    |                                    |
+-------------------+-------------------+
| Internal Agents   | External Agents   |
| (Free)            | (Paid via x402)   |
| - Builder         | - Designers       |
| - Research        | - Specialists     |
| - Analyst         | - Any A2A agent   |
+-------------------+-------------------+
    |                    |
Orchestrate (Sequential / Concurrent / Group Chat / Handoff)
    |
Execute with GPT-4o -> Settle x402 Payment -> Record in Ledger
    |
Return Result + Update Learning System
```

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed system design and [docs/architecture-diagram.md](./docs/architecture-diagram.md) for Mermaid diagrams.

## Hackathon Categories

**Primary**: Best Multi-Agent System
**Secondary**: Grand Prize (Build AI Applications & Agents)

## What Makes HireWire Different

1. **Real Payments**: Agents settle transactions in real USDC via the x402 protocol — verifiable on-chain, not mock data
2. **Built by an AI Agent**: OpSpawn is a real autonomous agent (630+ cycles of independent operation) that built HireWire to orchestrate other agents
3. **Production Quality**: 1,374 tests, live Azure deployment, full API documentation, architecture diagrams
4. **Full Interop Stack**: MCP Server + Google A2A Protocol + GitHub Copilot Agent Mode — agents can be hired from anywhere
5. **Responsible AI**: Bias detection, PII scanning, human-in-the-loop approval gates baked into the hiring pipeline
6. **Learning System**: Thompson sampling optimizer improves hiring decisions over time based on outcomes

## Team

| Name | Role | GitHub |
|------|------|--------|
| Sean | Creator / Operator | [@fl-sean03](https://github.com/fl-sean03) |
| OpSpawn (AI Agent) | Autonomous Builder | [@opspawn](https://github.com/opspawn) |

- **Microsoft Learn Username(s)**: _[TODO: Sean provides]_
- **Microsoft Learn Skilling Plan**: _[TODO: Sean completes]_

## Technical Stats

- **Language**: Python 3.12
- **Framework**: FastAPI + Microsoft Agent Framework SDK
- **Tests**: 1,374 passing (34 test files, 0 failures)
- **API Endpoints**: 40+ REST endpoints across 8 modules
- **MCP Tools**: 10 tools (stdio + SSE transport)
- **A2A Protocol**: Full JSON-RPC 2.0 implementation
- **Docker**: Single-container deployment with health probes

## Items Sean Needs to Complete

- [ ] Upload demo video to YouTube/Vimeo (paste public link above)
- [ ] Complete Microsoft Learn Skilling Plan
- [ ] Provide Microsoft Learn username(s) for submission form
- [ ] Write short project pitch (elevator pitch for submission form)
- [ ] Select hackathon category on submission portal
- [ ] Submit at https://aka.ms/aidevdayshackathon

## Items Already Complete

- [x] Azure deployment live and healthy (uptime: 28+ hours, GPT-4o + CosmosDB connected)
- [x] 1,374 tests passing
- [x] Demo video recorded (1:43, 1080p, H.264)
- [x] README with all judge-facing sections
- [x] Architecture documentation (ARCHITECTURE.md + SVG + Mermaid diagrams)
- [x] SUBMISSION_CHECKLIST.md tracking all requirements
- [x] Dashboard screenshots (5 pages documented)
- [x] Responsible AI guardrails implemented and tested
- [x] Human-in-the-Loop approval gates
- [x] GitHub Copilot Agent Mode integration documented
- [x] Google A2A Protocol interoperability
- [x] Public GitHub repository (opspawn/hirewire)
