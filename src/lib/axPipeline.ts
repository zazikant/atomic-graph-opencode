import { OpenCodeClient, calculateMaxTokens, getStageTemperature, estimateTokens } from "./opencodeClient";
import type {
  AtomicNode,
  ExtractResult,
  LinkResult,
  ValidationResult,
  PipelineResult,
  IterationLog,
  OpenCodeModel,
} from "./types";

/**
 * AX Pipeline — DSPy-like orchestration for Knowledge Graph generation
 *
 * This pipeline implements the same DSPy-inspired patterns from the
 * ax-opencode-translator project:
 *
 * - compileRefinePrompt: Like DSPy's Module.compile() — produces focused
 *   refinement prompts based on error history, not generic retries
 * - ErrorEntry tracking: Full error history for surgical, targeted fixes
 * - resumeFrom state machine: Deterministic pipeline progression
 * - Activity-style discrete steps: extract → link → validate → refine
 * - Stage-specific system prompts: Each stage has its own optimized prompt
 * - Stage-specific temperatures: extract=0.3, link=0.3, validate=0.1, refine=0.2
 * - Dynamic max_tokens: Calculated from input length (from repo 2)
 * - Echo detection: Detects when LLM echoes input unchanged
 *
 * Pipeline flow:
 * 1. Extract concepts (with 1500-char chunking for large inputs)
 * 2. Link relationships between concepts
 * 3. Validate graph quality (self-critique)
 * 4. If validation fails, refine with compiled error context (up to N iterations)
 */

// ─── Constants ─────────────────────────────────────────────────

/** Maximum characters per chunk for extraction.
 *  Set to 1500 as per the OpenCode API processing specification.
 *  Keeps prompts manageable and within the API's processing capacity. */
const CHUNK_CHAR_LIMIT = 1500;

// ─── DSPy-style Error Tracking ─────────────────────────────────

interface ErrorEntry {
  attempt: number;
  stage: "extract" | "link" | "validate" | "refine";
  error: string;
  issues?: string[];
}

// ─── Stage-Specific System Prompts (from ax-opencode-translator) ──────
// Like repo 2's translate/validate/refine prompts — each stage has its
// own specialized system prompt that focuses the LLM on its specific task.
// This is the core of the AX DSPy pattern: structured prompting per stage.

const EXTRACT_SYSTEM_PROMPT = `You are a semantic reasoning engine specialized in extracting atomic concepts from raw thinking.
You do NOT merely reformat or summarise — you REASON through the semantic space of ideas.
You surface implicit structure the writer already knows but didn't articulate.
You infer missing concepts, bridge gaps, and make hidden relationships explicit.

CRITICAL OUTPUT RULES:
- Always respond with valid JSON only. No markdown, no explanation, no code fences.
- Every response must be a single valid JSON object parseable by JSON.parse().
- Be CONCISE in summaries: 1-2 short sentences max per node.
- Keep titles short: 2-5 words.
- Use minimal tags: 1-3 per node.
- Do NOT repeat the input text verbatim in summaries — distill the core idea.
- Each concept must be ONE idea only — split compound ideas.
- Preserve the writer's original meaning faithfully.`;

const EXTRACT_RETRY_SYSTEM_PROMPT = `You are a semantic reasoning engine specialized in extracting atomic concepts. The previous attempt did not extract concepts properly.

CRITICAL: You MUST extract atomic concepts from the notes — do NOT echo the input text unchanged.

CRITICAL OUTPUT RULES:
- Always respond with valid JSON only. No markdown, no explanation, no code fences.
- Every response must be a single valid JSON object parseable by JSON.parse().
- Each concept = ONE idea only. Split compound ideas into separate concepts.
- Be CONCISE: short titles (2-5 words), brief summaries (1-2 sentences).
- Tags: 1-3 descriptive tags per node.
- Do NOT repeat the input text verbatim — distill the core idea.`;

const LINK_SYSTEM_PROMPT = `You are a semantic relationship mapper that discovers connections between concepts.
You find both direct and implicit relationships — the hidden structure that connects ideas.
You use SPECIFIC verbs for edge labels, never generic ones like "related to".

CRITICAL OUTPUT RULES:
- Always respond with valid JSON only. No markdown, no explanation, no code fences.
- Every response must be a single valid JSON object parseable by JSON.parse().
- Edge labels: use SPECIFIC verbs ("requires", "enables", "feeds into", "constrains", "extends"), NOT generic "related to".
- Keep labels to 1-3 words.
- Only create edges for REAL relationships. Do NOT fabricate connections.
- Return ONLY edges — do NOT repeat nodes.`;

const VALIDATE_SYSTEM_PROMPT = `You are a knowledge graph quality reviewer. Evaluate the provided graph and respond in JSON format.

Evaluate on these criteria (weight: semantic fidelity > atomicity > completeness > relationships > structure):
1. SEMANTIC FIDELITY: Does it preserve the writer's meaning? Minor wording changes don't lower score.
2. ATOMICITY: Is each concept truly one idea?
3. COMPLETENESS: Are implicit concepts captured?
4. RELATIONSHIP QUALITY: Specific edge labels ("requires") vs lazy ones ("related to")?
5. STRUCTURAL INTEGRITY: Orphan nodes? Missing cross-links?

Respond in this exact JSON format:
{
  "score": 0.85,
  "issues": ["issue1", "issue2"],
  "suggestions": ["suggestion1"]
}

Scoring:
- 0.90-1.00: Faithful representation, minor wording differences only
- 0.75-0.89: Minor gaps
- 0.50-0.74: Significant gaps
- 0.00-0.49: Major problems

Be FAIR — clear notes + good graph = high score. Don't invent reasons to lower it.
Only list issues affecting MEANING or STRUCTURE.`;

const REFINE_SYSTEM_PROMPT = `You are a knowledge graph refinement engine. Fix the identified issues while preserving what already works.
Output ONLY the corrected knowledge graph in JSON, nothing else.

CRITICAL OUTPUT RULES:
- Always respond with valid JSON only. No markdown, no explanation, no code fences.
- Every response must be a single valid JSON object parseable by JSON.parse().
- Preserve writer's wording where accurate — only fix SEMANTIC or STRUCTURAL problems.
- Missing bridge? INFER and add it.
- Generic edge? Replace with specific verb.
- Non-atomic concept? SPLIT into two and link.
- No speculative additions.
- Be CONCISE: short titles, brief summaries, 1-3 word edge labels.`;

// ─── compileRefinePrompt — pure function (DSPy-like) ──────────────
// Like DSPy's Module.compile() — produces focused prompts based on error history.
// This is the ONLY place refinement prompt construction happens.
// Pattern adopted from ax-opencode-translator's compileTranslatePrompt().

function compileRefinePrompt(
  issues: string[],
  errorHistory: ErrorEntry[],
  stage: "extract" | "link" | "validate" | "refine"
): string {
  // No errors yet — this is initial context
  if (errorHistory.length === 0 && issues.length === 0) {
    return `Initial ${stage} request for knowledge graph generation`;
  }

  // Build surgical context from error history (Mode B: surgical fix)
  const latestError = errorHistory.length > 0
    ? errorHistory[errorHistory.length - 1]
    : null;
  const previousErrors = errorHistory.slice(0, -1).map(e =>
    `  Attempt ${e.attempt} | ${e.stage}: ${e.error.substring(0, 200)}`
  ).join("\n");

  const issueContext = issues.length > 0
    ? `\nValidation issues to fix:\n${issues.map(i => `  - ${i}`).join("\n")}`
    : "";

  const errorContext = latestError
    ? `\nLatest error (attempt ${latestError.attempt}, stage ${latestError.stage}): ${latestError.error.substring(0, 300)}
${latestError.issues ? `Issues: ${latestError.issues.join(", ")}` : ""}`
    : "";

  const previousContext = previousErrors.length > 0
    ? `\nPrevious errors — do NOT repeat these patterns:\n${previousErrors}`
    : "";

  return `Refinement context for stage "${stage}":${issueContext}${errorContext}${previousContext}`;
}

// ─── Echo Detection ─────────────────────────────────────────────
// If the LLM returns the same text it was given (instead of extracting
// concepts), we detect it and retry with a more forceful prompt.
// Pattern from ax-opencode-translator's isEcho() function.

function isEcho(originalText: string, extractedText: string): boolean {
  const normalize = (s: string) => s.trim().toLowerCase().replace(/\s+/g, ' ');
  // Check if the extracted text is mostly the same as the input
  // (LLM sometimes just echoes input instead of extracting concepts)
  const normOrig = normalize(originalText);
  const normExtr = normalize(extractedText);
  // If the extracted text is >80% similar to the input, it's an echo
  if (normExtr.length < 10) return false; // Too short to be meaningful
  const overlap = normOrig.includes(normExtr.substring(0, Math.min(200, normExtr.length)));
  return overlap && normExtr.length > normOrig.length * 0.5;
}

// ─── AX Pipeline ───────────────────────────────────────────────

export class AXPipeline {
  private client: OpenCodeClient;
  private onIteration: (log: IterationLog) => void;
  private rawNotes: string = "";
  private errorHistory: ErrorEntry[] = [];

  constructor(
    apiKey: string,
    model: OpenCodeModel,
    onIteration: (log: IterationLog) => void
  ) {
    this.client = new OpenCodeClient(apiKey, model);
    this.onIteration = onIteration;
  }

  private emit(
    phase: IterationLog["phase"],
    iteration: number,
    score: number,
    passed: boolean,
    issues?: string[],
    detail?: string
  ) {
    this.onIteration({
      iteration,
      phase,
      score,
      passed,
      issues,
      detail,
      timestamp: Date.now(),
    });
  }

  /**
   * Wrap an async API call with rate-limit / transient error detection.
   * Records errors to errorHistory for DSPy-style compileRefinePrompt().
   */
  private async callWithRetryAwareness<T>(
    fn: () => Promise<T>,
    iteration: number,
    phaseLabel: string,
    stage: ErrorEntry["stage"]
  ): Promise<T> {
    try {
      return await fn();
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);

      // Track error in history for DSPy-style compiled prompts
      this.errorHistory.push({
        attempt: iteration,
        stage,
        error: msg,
      });

      const isRateLimit =
        msg.includes("rate limit") ||
        msg.includes("Rate limit") ||
        msg.includes("429") ||
        msg.includes("retry attempt") ||
        msg.includes("too many requests");

      const isTransient =
        msg.includes("server error") ||
        msg.includes("Server Error") ||
        msg.includes("502") ||
        msg.includes("503") ||
        msg.includes("504") ||
        msg.includes("temporary");

      if (isRateLimit) {
        this.emit(
          "retrying",
          iteration,
          0,
          false,
          undefined,
          `Rate limited during ${phaseLabel} — all proxy retries exhausted. Please wait and try again.`
        );
      } else if (isTransient) {
        this.emit(
          "retrying",
          iteration,
          0,
          false,
          undefined,
          `Transient server error during ${phaseLabel} — all proxy retries exhausted. Try again shortly.`
        );
      }

      throw error;
    }
  }

  // ─── Chunking Utilities ────────────────────────────────────────

  /**
   * Split raw notes into sensible chunks at paragraph/sentence boundaries.
   * Each chunk is at most CHUNK_CHAR_LIMIT (1500) characters.
   * Tries to split at paragraph breaks first, then sentence boundaries.
   */
  private splitIntoChunks(text: string): string[] {
    if (text.length <= CHUNK_CHAR_LIMIT) return [text];

    const chunks: string[] = [];
    let remaining = text;

    while (remaining.length > 0) {
      if (remaining.length <= CHUNK_CHAR_LIMIT) {
        chunks.push(remaining);
        break;
      }

      // Try to split at paragraph break
      let splitIndex = remaining.lastIndexOf("\n\n", CHUNK_CHAR_LIMIT);
      if (splitIndex < CHUNK_CHAR_LIMIT * 0.3) {
        // Try newline
        splitIndex = remaining.lastIndexOf("\n", CHUNK_CHAR_LIMIT);
      }
      if (splitIndex < CHUNK_CHAR_LIMIT * 0.3) {
        // Try sentence boundary (. ! ?)
        const sentenceMatch = remaining
          .slice(0, CHUNK_CHAR_LIMIT)
          .match(/[.!?]\s+/g);
        if (sentenceMatch) {
          const lastSentence = sentenceMatch[sentenceMatch.length - 1];
          splitIndex =
            remaining.slice(0, CHUNK_CHAR_LIMIT).lastIndexOf(lastSentence) +
            lastSentence.length;
        }
      }
      if (splitIndex < CHUNK_CHAR_LIMIT * 0.3) {
        // Hard split at limit
        splitIndex = CHUNK_CHAR_LIMIT;
      }

      chunks.push(remaining.slice(0, splitIndex).trim());
      remaining = remaining.slice(splitIndex).trim();
    }

    return chunks.filter((c) => c.length > 0);
  }

  /**
   * Deduplicate nodes by title similarity.
   */
  private deduplicateNodes(nodes: AtomicNode[]): AtomicNode[] {
    const seen = new Map<string, AtomicNode>();

    for (const node of nodes) {
      const key = node.title.toLowerCase().trim();

      if (seen.has(key)) {
        const existing = seen.get(key)!;
        const mergedTags = [...new Set([...existing.tags, ...node.tags])];
        existing.tags = mergedTags;
        if ((node.summary || "").length > (existing.summary || "").length) {
          existing.summary = node.summary;
        }
        if ((node.content || "").length > (existing.content || "").length) {
          existing.content = node.content;
        }
      } else {
        seen.set(key, { ...node });
      }
    }

    return Array.from(seen.values()).map((node, i) => ({
      ...node,
      id: `c${i + 1}`,
    }));
  }

  // ─── Step 1: EXTRACT — reason through semantic space ──────────

  private async extractChunk(
    chunk: string,
    chunkIndex: number,
    totalChunks: number,
    isRetry: boolean = false
  ): Promise<ExtractResult> {
    const contextHint =
      totalChunks > 1
        ? `\n\n[This is part ${chunkIndex + 1} of ${totalChunks} of the input. Focus on concepts in this section. Use tags to link related ideas across sections.]`
        : "";

    // Use retry-specific prompt when echo detected (from ax-opencode-translator pattern)
    const systemPrompt = isRetry ? EXTRACT_RETRY_SYSTEM_PROMPT : EXTRACT_SYSTEM_PROMPT;

    const prompt = `Extract atomic concepts from these notes. Each concept = ONE idea only.

Rules:
- Identify explicit AND implicit concepts (what's assumed but not named)
- Infer "glue" concepts that connect ideas but are left unsaid
- Title: 2-5 words. Summary: 1-2 SHORT sentences explaining WHY it matters.
- Preserve the writer's intent. Minor wording differences are NOT new concepts.
- Tags: 1-3 descriptive tags for grouping.
- Be CONCISE — do NOT repeat input text verbatim.

Return JSON: { "nodes": [{ "id": "c1", "title": "...", "summary": "...", "tags": ["..."] }] }

Raw notes:${contextHint}
${chunk}`;

    // AX DSPy: stage-specific temperature and dynamic max_tokens
    const temperature = getStageTemperature('extract');
    const maxTokens = calculateMaxTokens(chunk, 'extract');

    const result = await this.client.chatJSON<ExtractResult>(
      prompt,
      systemPrompt,
      { temperature, max_tokens: maxTokens }
    );

    if (!result.nodes || !Array.isArray(result.nodes)) {
      throw new Error("Extract step returned invalid nodes array");
    }

    result.nodes = result.nodes.map((node, i) => ({
      id: node.id || `c${i + 1}`,
      title: node.title || `Concept ${i + 1}`,
      summary: node.summary || "",
      tags: Array.isArray(node.tags) ? node.tags : [],
      content: node.content || node.summary || "",
    }));

    return result;
  }

  private async extract(
    rawNotes: string,
    previousResult: LinkResult | null,
    issues: string[]
  ): Promise<ExtractResult> {
    if (previousResult) {
      return this.extractRefinement(rawNotes, previousResult, issues);
    }

    const chunks = this.splitIntoChunks(rawNotes);

    if (chunks.length === 1) {
      return this.extractChunk(rawNotes, 0, 1);
    }

    this.emit(
      "chunking",
      1,
      0,
      false,
      undefined,
      `Processing ${chunks.length} sections of your notes (1500 chars each)…`
    );

    const allNodes: AtomicNode[] = [];
    let succeededChunks = 0;

    for (let i = 0; i < chunks.length; i++) {
      try {
        this.emit(
          "extracting",
          1,
          0,
          false,
          undefined,
          `Extracting section ${i + 1} of ${chunks.length}…`
        );

        const chunkResult = await this.callWithRetryAwareness(
          () => this.extractChunk(chunks[i], i, chunks.length),
          1,
          `Extract (section ${i + 1}/${chunks.length})`,
          "extract"
        );

        // Echo detection per chunk (from ax-opencode-translator pattern)
        if (isEcho(chunks[i], JSON.stringify(chunkResult))) {
          console.log(`[AX Pipeline] Echo detected in chunk ${i + 1} — retrying with forceful prompt`);
          try {
            const retryResult = await this.extractChunk(chunks[i], i, chunks.length, true);
            retryResult.nodes.forEach((node) => {
              node.id = `s${i + 1}_${node.id}`;
            });
            allNodes.push(...retryResult.nodes);
            succeededChunks++;
            continue;
          } catch {
            // Retry failed, use original result
          }
        }

        chunkResult.nodes.forEach((node) => {
          node.id = `s${i + 1}_${node.id}`;
        });

        allNodes.push(...chunkResult.nodes);
        succeededChunks++;
      } catch (error) {
        console.warn(
          `[AX Pipeline] Chunk ${i + 1}/${chunks.length} failed:`,
          error instanceof Error ? error.message : error
        );
        this.emit(
          "retrying",
          1,
          0,
          false,
          undefined,
          `Section ${i + 1} failed — continuing with remaining sections`
        );
      }
    }

    if (allNodes.length === 0) {
      throw new Error(
        `All ${chunks.length} sections failed to extract. Please try again.`
      );
    }

    const deduplicated = this.deduplicateNodes(allNodes);

    this.emit(
      "extracting",
      1,
      0,
      false,
      undefined,
      `Extracted ${deduplicated.length} concepts from ${succeededChunks}/${chunks.length} sections`
    );

    return { nodes: deduplicated };
  }

  /**
   * Refinement extraction using DSPy-compiled prompt from error history.
   * Like ax-opencode-translator's compileTranslatePrompt() — builds a
   * surgical fix prompt instead of a generic retry.
   */
  private async extractRefinement(
    rawNotes: string,
    previousResult: LinkResult | null,
    issues: string[]
  ): Promise<ExtractResult> {
    // Compile DSPy-style refinement context
    const fixContext = compileRefinePrompt(issues, this.errorHistory, "extract");
    console.log(`[AX Pipeline] Extract refinement context: ${fixContext.substring(0, 200)}`);

    const nodesCompact = previousResult?.nodes
      ?.slice(0, 50)
      .map((n) => `${n.id}: ${n.title}`)
      .join("\n");

    const prompt = `Refine concepts — fix ONLY the validation issues, nothing else.

PREVIOUS CONCEPTS:
${nodesCompact}

ORIGINAL NOTES:
${rawNotes.slice(0, 2000)}${rawNotes.length > 2000 ? "\n…" : ""}

Fix rules:
- Add truly missing concepts (not rephrased ones)
- Add bridge concepts that connect clusters
- Split genuinely non-atomic concepts
- Keep all existing valid concepts unchanged
- Be CONCISE: short titles, brief summaries
- ${fixContext}

Return JSON: { "nodes": [{ "id": "c1", "title": "...", "summary": "...", "tags": ["..."] }] }`;

    // AX DSPy: use REFINE system prompt with refinement temperature
    const temperature = getStageTemperature('refine');
    const maxTokens = calculateMaxTokens(rawNotes, 'extract');

    const result = await this.client.chatJSON<ExtractResult>(
      prompt,
      REFINE_SYSTEM_PROMPT,
      { temperature, max_tokens: maxTokens }
    );

    if (!result.nodes || !Array.isArray(result.nodes)) {
      throw new Error("Extract refinement returned invalid nodes array");
    }

    result.nodes = result.nodes.map((node, i) => ({
      id: node.id || `c${i + 1}`,
      title: node.title || `Concept ${i + 1}`,
      summary: node.summary || "",
      tags: Array.isArray(node.tags) ? node.tags : [],
      content: node.content || node.summary || "",
    }));

    return result;
  }

  // ─── Step 2: LINK — surface hidden relationships ──────────────

  private async link(extracted: ExtractResult): Promise<LinkResult> {
    const nodeSummary = extracted.nodes
      .map((n) => `${n.id}: ${n.title}`)
      .join("\n");

    const prompt = `Map relationships between these atomic concepts.

Find both direct and implicit relationships:
- Direct: A enables B, A requires B, A is a subtype of B
- Implicit: A and B connected through unstated C
- Causal: A leads to B which enables C

Edge labels: use SPECIFIC verbs ("requires", "enables", "feeds into", "constrains", "extends"), NOT generic "related to".
Keep labels to 1-3 words.

Strength (0.0-1.0):
- 0.9+: definitionally true
- 0.7-0.9: strongly implied
- 0.4-0.7: inferred bridge
- 0.0-0.4: speculative

Only create edges for REAL relationships. Do NOT fabricate connections.
Return ONLY edges — do NOT repeat nodes.

Return JSON: { "edges": [{ "source": "nodeId", "target": "nodeId", "label": "verb", "strength": 0.8 }] }

Nodes (id: title):
${nodeSummary}`;

    // AX DSPy: LINK stage-specific temperature and dynamic max_tokens
    const temperature = getStageTemperature('link');
    const maxTokens = calculateMaxTokens(nodeSummary, 'link');

    const result = await this.client.chatJSON<LinkResult>(
      prompt,
      LINK_SYSTEM_PROMPT,
      { temperature, max_tokens: maxTokens }
    );

    const nodeIds = new Set(extracted.nodes.map((n) => n.id));
    const validEdges = (result.edges || [])
      .filter(
        (e) =>
          e.source &&
          e.target &&
          nodeIds.has(e.source) &&
          nodeIds.has(e.target) &&
          e.source !== e.target
      )
      .map((e) => ({
        source: e.source,
        target: e.target,
        label: e.label || "relates to",
        strength:
          typeof e.strength === "number"
            ? Math.min(1, Math.max(0, e.strength))
            : 0.5,
      }));

    return {
      nodes: extracted.nodes,
      edges: validEdges,
    };
  }

  // ─── Step 3: VALIDATE — quality-aware self-critique ───────────

  private async validate(graph: LinkResult): Promise<ValidationResult> {
    const graphCompact = {
      nodes: graph.nodes.map((n) => ({ id: n.id, title: n.title, summary: n.summary, tags: n.tags })),
      edges: graph.edges.map((e) => ({ source: e.source, target: e.target, label: e.label, strength: e.strength })),
    };

    const notesForValidation = this.rawNotes.length > 3000
      ? this.rawNotes.slice(0, 3000) + "\n… (notes truncated)"
      : this.rawNotes;

    const prompt = `Evaluate this knowledge graph for quality and semantic fidelity.

Axes (weight: semantic fidelity > atomicity > completeness > relationships > structure):
1. SEMANTIC FIDELITY: Does it preserve the writer's meaning? Minor wording changes don't lower score.
2. ATOMICITY: Is each concept truly one idea?
3. COMPLETENESS: Are implicit concepts captured?
4. RELATIONSHIP QUALITY: Specific edge labels ("requires") vs lazy ones ("related to")?
5. STRUCTURAL INTEGRITY: Orphan nodes? Missing cross-links?

Scoring:
- 0.90-1.00: Faithful representation, minor wording differences only
- 0.75-0.89: Minor gaps
- 0.50-0.74: Significant gaps
- 0.00-0.49: Major problems

Be FAIR — clear notes + good graph = high score. Don't invent reasons to lower it.

ORIGINAL NOTES:
${notesForValidation}

Graph: ${JSON.stringify(graphCompact)}

Return JSON: { "score": 0.85, "issues": ["..."], "suggestions": ["..."] }
Only list issues affecting MEANING or STRUCTURE.`;

    // AX DSPy: VALIDATE stage uses lowest temperature for objective assessment
    const temperature = getStageTemperature('validate');
    const maxTokens = calculateMaxTokens('', 'validate'); // Returns 1024 for validate

    const result = await this.client.chatJSON<ValidationResult>(
      prompt,
      VALIDATE_SYSTEM_PROMPT,
      { temperature, max_tokens: maxTokens }
    );

    return {
      score:
        typeof result.score === "number"
          ? Math.min(1, Math.max(0, result.score))
          : 0,
      issues: Array.isArray(result.issues) ? result.issues : [],
      suggestions: Array.isArray(result.suggestions) ? result.suggestions : [],
    };
  }

  // ─── Step 4: REFINE — targeted fixes using DSPy-compiled prompt ──

  private async refine(
    graph: LinkResult,
    issues: string[]
  ): Promise<LinkResult> {
    // Compile DSPy-style refinement context from error history
    const fixContext = compileRefinePrompt(issues, this.errorHistory, "refine");
    console.log(`[AX Pipeline] Refinement context: ${fixContext.substring(0, 200)}`);

    const graphCompact = {
      nodes: graph.nodes.map((n) => ({ id: n.id, title: n.title, summary: n.summary, tags: n.tags })),
      edges: graph.edges.map((e) => ({ source: e.source, target: e.target, label: e.label, strength: e.strength })),
    };

    const prompt = `Fix ONLY these specific issues in the knowledge graph.

Rules:
- Preserve writer's wording where accurate
- Only fix SEMANTIC or STRUCTURAL problems, not style
- Missing bridge? INFER and add it.
- Generic edge? Replace with specific verb.
- Non-atomic concept? SPLIT into two and link.
- No speculative additions.
- Be CONCISE: short titles, brief summaries, 1-3 word edge labels.
- ${fixContext}

Return the FULL corrected graph (both nodes and edges). Keep unchanged items as-is.

Return JSON: { "nodes": [{ "id": "c1", "title": "...", "summary": "...", "tags": ["..."] }], "edges": [{ "source": "id", "target": "id", "label": "verb", "strength": 0.8 }] }

Issues:
${issues.map((i) => `- ${i}`).join("\n")}

Current graph: ${JSON.stringify(graphCompact)}`;

    // AX DSPy: REFINE stage uses slightly higher temperature for creative fixes
    const temperature = getStageTemperature('refine');
    const graphStr = JSON.stringify(graphCompact);
    const maxTokens = calculateMaxTokens(graphStr, 'refine');

    const result = await this.client.chatJSON<LinkResult>(
      prompt,
      REFINE_SYSTEM_PROMPT,
      { temperature, max_tokens: maxTokens }
    );

    const nodeIds = new Set((result.nodes || []).map((n) => n.id));
    const validEdges = (result.edges || [])
      .filter(
        (e) =>
          e.source &&
          e.target &&
          nodeIds.has(e.source) &&
          nodeIds.has(e.target) &&
          e.source !== e.target
      )
      .map((e) => ({
        source: e.source,
        target: e.target,
        label: e.label || "relates to",
        strength:
          typeof e.strength === "number"
            ? Math.min(1, Math.max(0, e.strength))
            : 0.5,
      }));

    const nodes = (result.nodes || graph.nodes).map((node, i) => ({
      id: node.id || graph.nodes[i]?.id || `c${i + 1}`,
      title: node.title || graph.nodes[i]?.title || `Concept ${i + 1}`,
      summary: node.summary || graph.nodes[i]?.summary || "",
      tags: Array.isArray(node.tags) ? node.tags : graph.nodes[i]?.tags || [],
      content:
        node.content || node.summary || graph.nodes[i]?.content || "",
    }));

    return {
      nodes,
      edges: validEdges.length > 0 ? validEdges : graph.edges,
    };
  }

  // ─── Fast Pipeline (like repo 2's runFastTranslation) ────────────
  // Extract + Link only, skip validate/refine.
  // Cuts pipeline from 3-5 LLM calls to just 2 (extract + link).
  // ~30-60s total instead of 90-150s.

  async runFast(
    rawNotes: string
  ): Promise<PipelineResult> {
    this.rawNotes = rawNotes;
    this.errorHistory = [];

    // ─── Stage 1: EXTRACT ──────────────────────────────────────────
    this.emit("extracting", 1, 0, false);
    let extracted: ExtractResult;
    try {
      extracted = await this.callWithRetryAwareness(
        () => this.extract(rawNotes, null, []),
        1,
        "Extract",
        "extract"
      );

      // Echo detection
      if (isEcho(rawNotes, JSON.stringify(extracted))) {
        console.log("[AX Pipeline] Echo detected — retrying with forceful prompt");
        try {
          extracted = await this.callWithRetryAwareness(
            () => this.extract(rawNotes, null, []),
            1,
            "Extract (echo retry)",
            "extract"
          );
        } catch {
          // Retry failed, use original result
        }
      }
    } catch (extractError) {
      throw extractError;
    }

    // ─── Stage 2: LINK ─────────────────────────────────────────────
    this.emit("linking", 1, 0, false);
    let result: LinkResult;
    try {
      result = await this.callWithRetryAwareness(
        () => this.link(extracted),
        1,
        "Link",
        "link"
      );
    } catch (linkError) {
      console.warn("[AX Pipeline] Link failed, using nodes without edges");
      result = { nodes: extracted.nodes, edges: [] };
    }

    this.emit("complete", 1, 0.85, true, undefined, "Fast mode — graph generated");

    return {
      result,
      score: 0.85, // Estimated — no validation step
      attempts: 1,
      iterations: [],
    };
  }

  // ─── Main Pipeline Runner (resumeFrom state machine) ───────────
  // Like ax-opencode-translator's resumeFrom pattern — deterministic
  // pipeline progression with state tracking.
  //
  // KEY FIX: Results are carried FORWARD between stages, not re-computed.
  // This matches repo 2's pattern where translate → validate → refine
  // passes the same result through without re-running earlier stages.
  //
  // Flow: extract → link → validate → (if fail) refine → validate → done

  async run(
    rawNotes: string,
    iterations: number,
    threshold: number
  ): Promise<PipelineResult> {
    this.rawNotes = rawNotes;
    this.errorHistory = []; // Reset error history for new pipeline run
    let result: LinkResult | null = null;
    let score = 0;
    let attempt = 0;
    let currentIssues: string[] = [];
    const iterLogs: IterationLog[] = [];

    // State machine: extract → link → validate → refine → done
    type PipelineStage = "extract" | "link" | "validate" | "refine" | "done";
    let resumeFrom: PipelineStage = "extract";

    // ─── Stage 1: EXTRACT (runs once, result carried forward) ──────
    if (resumeFrom === "extract") {
      attempt++;
      this.emit("extracting", attempt, 0, false);
      let extracted: ExtractResult;
      try {
        extracted = await this.callWithRetryAwareness(
          () => this.extract(rawNotes, null, []),
          attempt,
          "Extract",
          "extract"
        );

        // Echo detection (from ax-opencode-translator pattern)
        if (isEcho(rawNotes, JSON.stringify(extracted))) {
          console.log("[AX Pipeline] Echo detected in extract — retrying with forceful prompt");
          try {
            const retryExtracted = await this.callWithRetryAwareness(
              () => this.extract(rawNotes, null, []),
              attempt,
              "Extract (echo retry)",
              "extract"
            );
            extracted = retryExtracted;
          } catch {
            // Retry failed, use original result
          }
        }

        resumeFrom = "link";
      } catch (extractError) {
        throw extractError;
      }

      // ─── Stage 2: LINK (runs once after extract, result carried) ──
      if (resumeFrom === "link") {
        this.emit("linking", attempt, 0, false);
        try {
          result = await this.callWithRetryAwareness(
            () => this.link(extracted),
            attempt,
            "Link",
            "link"
          );
          resumeFrom = "validate";
        } catch (linkError) {
          console.warn("[AX Pipeline] Link failed, using nodes without edges");
          result = { nodes: extracted.nodes, edges: [] };
          resumeFrom = "validate";
        }
      }
    }

    // ─── Validate → Refine loop (carries result forward like repo 2) ──
    // This matches ax-opencode-translator's pattern:
    //   translate → validate → (if fail) refine → revalidate → done
    // Here: extract+link already done → validate → (if fail) refine → validate → done

    while (attempt < iterations && score < threshold && resumeFrom !== "done") {
      // ─── Stage 3: VALIDATE ─────────────────────────────────────
      if (resumeFrom === "validate") {
        this.emit("validating", attempt, 0, false);
        try {
          const validation = await this.callWithRetryAwareness(
            () => this.validate(result!),
            attempt,
            "Validate",
            "validate"
          );
          score = validation.score;
          currentIssues = validation.issues;

          const passed = score >= threshold;
          this.emit("validating", attempt, score, passed, validation.issues);

          iterLogs.push({
            iteration: attempt,
            phase: passed ? "complete" : "validating",
            score,
            passed,
            issues: validation.issues,
            timestamp: Date.now(),
          });

          if (passed) {
            resumeFrom = "done";
          } else if (attempt < iterations) {
            resumeFrom = "refine";
          } else {
            resumeFrom = "done";
          }
        } catch (validateError) {
          console.warn("[AX Pipeline] Validate failed, using default score");
          score = 0.7;
          currentIssues = ["Validation step failed — using estimated score"];

          iterLogs.push({
            iteration: attempt,
            phase: "validating",
            score,
            passed: score >= threshold,
            issues: currentIssues,
            timestamp: Date.now(),
          });

          if (score >= threshold) {
            resumeFrom = "done";
          } else if (attempt < iterations) {
            resumeFrom = "refine";
          } else {
            resumeFrom = "done";
          }
        }
      }

      // ─── Stage 4: REFINE (DSPy-compiled prompt) ────────────────
      if (resumeFrom === "refine") {
        attempt++;
        this.emit("refining", attempt, score, false);
        try {
          const refined = await this.callWithRetryAwareness(
            () => this.refine(result!, currentIssues),
            attempt,
            "Refine",
            "refine"
          );
          result = refined;
          // After refine, loop back to validate (like repo 2's revalidate)
          resumeFrom = "validate";
        } catch (refineError) {
          console.warn("[AX Pipeline] Refine failed, keeping current result");
          // Keep current result, loop back to validate
          resumeFrom = "validate";
        }
      }
    }

    this.emit("complete", attempt, score, score >= threshold);

    return {
      result,
      score,
      attempts: attempt,
      iterations: iterLogs,
    };
  }
}
