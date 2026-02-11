# HireWire Submission Checklist

**Microsoft AI Agent Hackathon — AI Dev Days 2026**

## GitHub Repository
- [x] Repo public and polished: [github.com/opspawn/hirewire](https://github.com/opspawn/hirewire)
- [x] README with architecture diagram, features, API reference, demo instructions
- [x] All features documented: Agent Framework SDK, MCP, A2A, Marketplace, x402, HITL, RAI, Foundry
- [x] Test count updated (1293+ passing)
- [x] License (MIT)

## Azure Deployment
- [x] Azure Container Apps live and healthy
- [x] Dashboard URL: https://hirewire-api.purplecliff-500810ff.eastus.azurecontainerapps.io/
- [x] Health endpoint: /health
- [x] Azure OpenAI (GPT-4o), CosmosDB, Container Registry, Application Insights

## Demo
- [x] `demo/record_demo.py` runs all 8 stages end-to-end (--fast mode verified)
- [x] `demo/DEMO_TRANSCRIPT.md` — full output formatted for video script reference
- [x] Dashboard works standalone (no server needed) with demo data
- [ ] Demo video recorded and uploaded
- [x] Demo scenarios: landing-page, research, agent-hiring, showcase

## Tests
- [x] 1293 tests passing (`python3 -m pytest tests/ -q`)
- [x] 33 test files covering agents, workflows, marketplace, payments, API, Foundry, A2A, MCP

## Hackathon Requirements
- [ ] Microsoft Learn Skilling Plan completed (BLOCKED - Sean)
- [x] Architecture diagram (docs/architecture.svg)
- [x] Responsible AI guardrails (bias detection, PII scanning, content safety, HITL)
- [x] Uses Azure services (OpenAI, CosmosDB, Container Apps, Foundry, App Insights)

## Key Differentiators
- Real autonomous agent (OpSpawn) building agent infrastructure
- x402 micropayment protocol for agent-to-agent commerce
- Full hiring lifecycle: discover, negotiate, escrow, pay, learn
- Human-in-the-Loop approval gates for expensive operations
- 8 Azure services integrated
- 1293+ tests
