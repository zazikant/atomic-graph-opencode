# Atomic Graph — AI-Powered Knowledge Graph Builder

Transform raw notes into interactive knowledge graphs using AI semantic reasoning powered by the OpenCode API (GLM 5.1).

## Features

- **AX DSPy-style Pipeline**: Extract → Link → Validate → Refine loop
- **1500-character Chunking**: Processes large inputs in 1500-char segments
- **OpenCode API**: Uses GLM 5.1 with `reasoning_effort: "none"` for reliable output
- **Interactive Graph Visualization**: ReactFlow-powered knowledge graph
- **JSON Import/Export**: Import graphs from JSON or AI-generated results
- **Vercel Compatible**: Deploy with a single environment variable

## Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Set Environment Variable

```bash
cp .env.example .env
# Edit .env and add your OpenCode API key
```

### 3. Run Development Server

```bash
npm run dev
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENCODE_API_KEY` | Yes* | Your OpenCode API key from https://opencode.ai/ |

*Can also be provided via the UI config bar if not set on the server.

## Deployment on Vercel

1. Fork or push this repo to GitHub
2. Import the repo in [Vercel](https://vercel.com)
3. Add `OPENCODE_API_KEY` in your Vercel project's Environment Variables
4. Deploy — that's it!

The `vercel.json` configures the API route for up to 60s timeout (requires Vercel Pro).

## API Endpoint

```
POST /api/opencode
Content-Type: application/json
```

```json
{
  "apiKey": "YOUR_OPENCODE_API_KEY",
  "model": "glm-5.1",
  "messages": [
    { "role": "system", "content": "<SYSTEM_PROMPT>" },
    { "role": "user", "content": "<USER_PROMPT>" }
  ],
  "temperature": 0.7,
  "max_tokens": 16384
}
```

### Config Endpoint

```
GET /api/config
```

Returns `{ "hasServerKey": true/false }` — tells the frontend if a server-side key is available.

## Pipeline Architecture

The AX Pipeline implements a DSPy-inspired orchestration pattern:

```
repeat up to N iterations (default: 3):
  1. EXTRACT  → nodes[]        (splits input into 1500-char chunks)
  2. LINK     → edges[]        (maps relationships between nodes)
  3. VALIDATE → score, issues[] (quality-aware self-critique)
  if score >= threshold (default: 0.75):
    break
  else:
  4. REFINE   → updated nodes[], edges[] (targeted fixes only)
    go back to step 3
```

### Key Design Decisions

- **1500-char chunking**: Keeps prompts within the OpenCode API's optimal processing range
- **reasoning_effort: "none"**: Disables GLM 5.1's internal reasoning so all `max_tokens` go to content output
- **Proxy route**: Server-side `/api/opencode` route bypasses CORS restrictions
- **Retry logic**: 3 retries with 15s delay for 429/5xx errors

## Output Format

```json
{
  "nodes": [
    {
      "id": "c1",
      "title": "Short Concept Title",
      "summary": "1-2 sentence explanation of why this matters.",
      "tags": ["tag1", "tag2"]
    }
  ],
  "edges": [
    {
      "source": "c1",
      "target": "c2",
      "label": "enables",
      "strength": 0.85
    }
  ]
}
```

## Tech Stack

- **Next.js 16** with App Router
- **React 19** with TypeScript
- **Tailwind CSS 4** with shadcn/ui components
- **ReactFlow** for graph visualization
- **Zustand** for state management
- **Dagre** for automatic graph layout
- **OpenCode API** (GLM 5.1) for AI processing
