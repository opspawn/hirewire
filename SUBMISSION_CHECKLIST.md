# HireWire Submission Checklist

**Microsoft AI Agent Hackathon — AI Dev Days 2026**
**Submission Deadline: March 15, 2026 (11:59 PM PT)**
**Last verified: Sprint 42 (Feb 11, 2026)**

## GitHub Repository
- [x] Repo public and polished: [github.com/opspawn/hirewire](https://github.com/opspawn/hirewire)
- [x] README with architecture diagram, features, API reference, demo instructions
- [x] All features documented: Agent Framework SDK, MCP, A2A, Marketplace, x402, HITL, RAI, Foundry
- [x] Test count updated (1293+ passing)
- [x] License (MIT)
- [x] All README links verified (architecture.svg, demo images, ARCHITECTURE.md, architecture-diagram.md)

## Azure Deployment
- [x] Azure Container Apps live and healthy
- [x] Dashboard URL: https://hirewire-api.purplecliff-500810ff.eastus.azurecontainerapps.io/
- [x] Health endpoint: /health
- [x] Azure OpenAI (GPT-4o), CosmosDB, Container Registry, Application Insights

## Demo
- [x] `demo/record_demo.py` runs all 8 stages end-to-end (--fast mode verified)
- [x] `demo/DEMO_TRANSCRIPT.md` — full output formatted for video script reference
- [x] Dashboard works standalone with demo data (verified Sprint 42)
- [x] Demo video recorded: docs/demo/hirewire-demo-v2.mp4 (1:43, 1920x1080, H.264)
- [ ] **Demo video uploaded to YouTube/Vimeo (Sean)** — REQUIRED for submission
- [x] Demo scenarios: landing-page, research, agent-hiring, showcase

## Tests
- [x] 1293 tests passing (`python3 -m pytest tests/ -q`) — verified Sprint 42
- [x] 33 test files covering agents, workflows, marketplace, payments, API, Foundry, A2A, MCP
- [x] 8 azure-marked tests correctly deselected (require live credentials)

## Hackathon Requirements (from Official Rules)
- [ ] **Microsoft Learn Skilling Plan completed (Sean)** — Rules require: "Create a Microsoft Learn profile and complete the Microsoft Learn Skilling Plan(s) shared on the hackathon landing page." Specific plans not yet listed on landing page; check back closer to submission deadline.
- [ ] **Microsoft Learn usernames for all participants (Sean)** — Submission form requires team members' MS Learn usernames.
- [ ] **Project pitch/description text (Sean)** — Short elevator pitch for the submission form.
- [ ] **Hackathon category selection (Sean)** — Best fits: "Best Multi-Agent System" and/or "Best Azure Integration". Can also target Grand Prize: "Build AI Applications & Agents".
- [x] Architecture diagram (docs/architecture.svg)
- [x] Responsible AI guardrails (bias detection, PII scanning, content safety, HITL)
- [x] Uses Azure services (OpenAI, CosmosDB, Container Apps, Foundry, App Insights)
- [x] Public GitHub repository
- [x] Demo video < 2 minutes (1:43)

## Items NOT Requiring Sean
- [x] Test suite healthy (1293 passing, 0 failures)
- [x] README accurate and complete
- [x] Dashboard renders correctly with demo data
- [x] All demo assets present (SVG, PNGs, GIF, MP4)
- [x] Architecture docs complete

## Items Requiring Sean
- [ ] Upload demo video to YouTube/Vimeo (public link required)
- [ ] Complete Microsoft Learn Skilling Plan (when published)
- [ ] Provide Microsoft Learn username(s)
- [ ] Write project pitch/description for submission form
- [ ] Select hackathon category
- [ ] Submit on hackathon website (https://aka.ms/aidevdayshackathon)

## Key Differentiators
- Real autonomous agent (OpSpawn) building agent infrastructure
- x402 micropayment protocol for agent-to-agent commerce
- Full hiring lifecycle: discover, negotiate, escrow, pay, learn
- Human-in-the-Loop approval gates for expensive operations
- 8 Azure services integrated
- 1293+ tests
- GitHub Copilot Agent Mode integration via MCP
- Google A2A protocol interoperability
