/**
 * OpenCode API Client — OpenAI Chat Completions interface
 *
 * Calls the GLM 5.1 model via the OpenAI-compatible chat completions endpoint
 * through our Next.js proxy route to avoid browser CORS restrictions.
 *
 * Base URL: https://opencode.ai/zen/go
 * Default model: glm-5.1
 * Endpoint: /v1/chat/completions (OpenAI-compatible)
 * Auth: Authorization: Bearer header
 *
 * GLM 5.1 is a thinking/reasoning model. We disable thinking via
 * reasoning_effort: "none" — this turns off internal reasoning so ALL
 * max_tokens go to content output. This prevents the "empty content" bug
 * where reasoning consumes all tokens. Quality is maintained through the
 * detailed system prompt in axPipeline.ts.
 */

import type { OpenCodeModel } from "./types";

const DEFAULT_MODEL: OpenCodeModel = "glm-5.1";

// ─── Token Estimation (from ax-opencode-translator) ──────────
// Rough: 1 token ≈ 4 chars for English, 2 chars for CJK

export function estimateTokens(text: string): number {
  const cjkChars = (text.match(/[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]/g) || []).length;
  const otherChars = text.length - cjkChars;
  return Math.ceil(cjkChars / 2 + otherChars / 4);
}

/**
 * Calculate max_tokens for the output based on input length.
 * With reasoning_effort: "none", thinking is OFF — all tokens go to content.
 * No need to reserve tokens for reasoning, so minimum can be lower.
 *
 * For knowledge graph extraction, output can be 2-3× the input
 * (notes → structured graph with nodes and edges).
 * For validation, output is small JSON (~1024 is enough).
 *
 * Minimum 2048, maximum 8192.
 */
export function calculateMaxTokens(inputText: string, stage: 'extract' | 'link' | 'validate' | 'refine' = 'extract'): number {
  // Validation returns small JSON — fixed 1024 is enough
  if (stage === 'validate') return 1024;

  const inputTokens = estimateTokens(inputText);
  // Knowledge graph extraction produces structured output ≈ 2× input
  const multiplier = stage === 'extract' ? 2 : 1.5;
  const outputTokens = Math.ceil(inputTokens * multiplier);
  // With reasoning_effort "none", no tokens are consumed by reasoning
  return Math.max(2048, Math.min(8192, outputTokens));
}

/**
 * Stage-specific temperature settings from ax-opencode-translator.
 * Lower temperature = more deterministic/focused output.
 * Higher temperature = more creative/diverse output.
 */
export function getStageTemperature(stage: 'extract' | 'link' | 'validate' | 'refine'): number {
  switch (stage) {
    case 'extract': return 0.3;  // Focused, deterministic concept extraction
    case 'link': return 0.3;     // Focused, precise relationship mapping
    case 'validate': return 0.1; // Very deterministic — objective quality assessment
    case 'refine': return 0.2;   // Slightly creative — targeted fixes
    default: return 0.3;
  }
}

export class OpenCodeClient {
  private apiKey: string;
  private model: OpenCodeModel;

  constructor(apiKey: string, model: OpenCodeModel) {
    this.apiKey = apiKey;
    this.model = model;
  }

  updateConfig(apiKey: string, model: OpenCodeModel) {
    this.apiKey = apiKey;
    this.model = model;
  }

  /**
   * Send a chat completion request through our server-side proxy.
   * The proxy forwards to OpenCode API, bypassing CORS.
   * Handles reasoning models that return output in reasoning_content.
   *
   * The proxy retries up to 3 times on 429/5xx errors with 15s delay.
   * If all retries are exhausted, throws a descriptive error.
   *
   * AX DSPy-style: temperature and max_tokens are stage-specific.
   * Each pipeline stage has its own optimal temperature for quality.
   */
  async chat(
    userPrompt: string,
    systemPrompt?: string,
    options?: { temperature?: number; max_tokens?: number }
  ): Promise<string> {
    const messages: { role: "system" | "user" | "assistant"; content: string }[] = [];

    if (systemPrompt) {
      messages.push({ role: "system", content: systemPrompt });
    }

    messages.push({ role: "user", content: userPrompt });

    const response = await fetch("/api/opencode", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        apiKey: this.apiKey,
        model: this.model,
        messages,
        temperature: options?.temperature ?? 0.3,
        max_tokens: options?.max_tokens ?? 4096,
      }),
    });

    if (!response.ok) {
      let errorMsg = `API error (${response.status})`;
      let retryable = false;
      let attemptsExhausted = false;

      try {
        const errBody = await response.json();
        errorMsg = errBody.error || errBody.details || errorMsg;
        retryable = errBody.retryable ?? false;
        attemptsExhausted = errBody.attemptsExhausted ?? false;
      } catch {
        // couldn't parse error body
      }

      // Provide user-friendly error messages based on common scenarios
      if (attemptsExhausted) {
        throw new Error(
          `OpenCode API rate limit reached — all 3 retry attempts failed after 15-second delays. Please wait a moment and try again.`
        );
      }

      if (response.status === 401) {
        throw new Error(
          "Invalid API key. Please check your OpenCode API key in the config bar and try again."
        );
      }

      if (response.status === 403) {
        throw new Error(
          "Access denied — your API key does not have permission to use this model."
        );
      }

      if (response.status === 429) {
        throw new Error(
          "Rate limited by OpenCode — too many requests. Please wait a moment before trying again."
        );
      }

      if (response.status >= 500) {
        throw new Error(
          `OpenCode server error (${response.status}): ${errorMsg}. This is usually temporary — try again in a moment.`
        );
      }

      throw new Error(errorMsg);
    }

    const data = await response.json();
    const message = data.choices?.[0]?.message;

    if (!message) {
      throw new Error("OpenCode API returned no message in response");
    }

    // GLM 5.1 returns reasoning in reasoning_content and the final answer in content.
    // If content is empty (token limit hit during reasoning), fall back to reasoning_content.
    const content = message.content || message.reasoning_content;

    if (!content) {
      throw new Error(
        "OpenCode API returned an empty response (both content and reasoning_content are null)"
      );
    }

    return content as string;
  }

  /**
   * Call the LLM and attempt to parse the response as JSON.
   * Handles common LLM output quirks (markdown code blocks, extra text,
   * reasoning model outputs that may include chain-of-thought before JSON).
   *
   * Recovery chain:
   * 1. Try normal JSON parse with balanced bracket extraction
   * 2. Try truncation recovery (close brackets, recover partial data)
   * 3. If recovery returns partial data, return it (caller decides how to handle)
   * 4. If all recovery fails, throw the original parse error
   */
  async chatJSON<T>(
    userPrompt: string,
    systemPrompt?: string,
    options?: { temperature?: number; max_tokens?: number }
  ): Promise<T> {
    const raw = await this.chat(userPrompt, systemPrompt, options);
    try {
      return parseLLMJson<T>(raw);
    } catch (parseError) {
      // If the response was truncated (common with reasoning models),
      // try to recover what we can rather than failing entirely.
      const recovered = recoverTruncatedJSON<T>(raw);
      if (recovered !== null) {
        console.warn(`[OpenCodeClient] Recovered truncated JSON (${raw.length} chars input, recovered keys: ${Object.keys(recovered as object).join(', ')})`);
        return recovered;
      }

      // Enhanced error message with context about why recovery failed
      const errMsg = parseError instanceof Error ? parseError.message : String(parseError);
      throw new Error(
        `${errMsg}\nRecovery attempted on ${raw.length}-character response but could not extract valid JSON. This usually means the output was truncated mid-structure.`
      );
    }
  }
}

/**
 * Robust JSON parser for LLM outputs.
 * Strips markdown code fences, extracts JSON objects,
 * and handles reasoning model outputs that may contain
 * chain-of-thought text before/after the JSON.
 */
export function parseLLMJson<T>(raw: string): T {
  // Strip markdown code fences: ```json ... ``` or ``` ... ```
  let cleaned = raw.trim();

  const fenceMatch = cleaned.match(/```(?:json)?\s*\n?([\s\S]*?)\n?\s*```/);
  if (fenceMatch) {
    cleaned = fenceMatch[1].trim();
  }

  // Strategy 1: Try direct parse of the full text
  try {
    return JSON.parse(cleaned) as T;
  } catch {
    // Continue to more sophisticated parsing
  }

  // Strategy 2: Extract balanced JSON using bracket matching.
  const jsonStr = extractBalancedJSON(cleaned);
  if (jsonStr) {
    try {
      return JSON.parse(jsonStr) as T;
    } catch {
      // Continue to fixes
    }
  }

  // Strategy 3: Fix common issues like trailing commas
  if (jsonStr) {
    const fixed = jsonStr.replace(/,\s*([}\]])/g, "$1");
    try {
      return JSON.parse(fixed) as T;
    } catch {
      // Continue
    }
  }

  // Strategy 4: Try the greedy regex as a last resort before failing
  const greedyMatch = cleaned.match(/(\{[\s\S]*\}|\[[\s\S]*\])/);
  if (greedyMatch) {
    try {
      return JSON.parse(greedyMatch[1]) as T;
    } catch {
      // Greedy match didn't work either
    }
  }

  throw new Error(
    `Failed to parse LLM JSON response. Output length: ${raw.length} chars.\nFirst 300 chars: ${raw.slice(0, 300)}\nLast 300 chars: ${raw.slice(-300)}`
  );
}

/**
 * Extract a balanced JSON object or array from text that may contain
 * extra content before or after the JSON.
 */
function extractBalancedJSON(text: string): string | null {
  const startCurly = text.indexOf("{");
  const startSquare = text.indexOf("[");

  let startIdx: number;
  let openCh: string;
  let closeCh: string;

  if (startCurly === -1 && startSquare === -1) return null;
  if (startCurly === -1) {
    startIdx = startSquare;
    openCh = "[";
    closeCh = "]";
  } else if (startSquare === -1) {
    startIdx = startCurly;
    openCh = "{";
    closeCh = "}";
  } else {
    startIdx = Math.min(startCurly, startSquare);
    openCh = startIdx === startCurly ? "{" : "[";
    closeCh = startIdx === startCurly ? "}" : "]";
  }

  let depth = 0;
  let inString = false;
  let escape = false;

  for (let i = startIdx; i < text.length; i++) {
    const ch = text[i];

    if (escape) {
      escape = false;
      continue;
    }

    if (ch === "\\" && inString) {
      escape = true;
      continue;
    }

    if (ch === '"') {
      inString = !inString;
      continue;
    }

    if (inString) continue;

    if (ch === openCh || ch === "{" || ch === "[") {
      if (ch === openCh && depth === 0) {
        depth = 1;
      } else if (depth > 0) {
        depth++;
      }
    } else if (ch === closeCh || ch === "}" || ch === "]") {
      if (depth > 0) {
        depth--;
        if (depth === 0) {
          return text.slice(startIdx, i + 1);
        }
      }
    }
  }

  return null;
}

/**
 * Attempt to recover useful data from a truncated JSON response.
 */
export function recoverTruncatedJSON<T>(raw: string): T | null {
  let cleaned = raw.trim();

  const fenceMatch = cleaned.match(/```(?:json)?\s*\n?([\s\S]*?)\n?\s*```/);
  if (fenceMatch) {
    cleaned = fenceMatch[1].trim();
  }

  const jsonStart = cleaned.search(/\{/);
  if (jsonStart === -1) return null;
  cleaned = cleaned.slice(jsonStart);

  const attempt1 = closeOpenBrackets(cleaned);
  if (attempt1 !== null) {
    return attempt1 as T;
  }

  const attempt2 = truncateToLastCompleteElement(cleaned);
  if (attempt2 !== null) {
    return attempt2 as T;
  }

  const attempt3 = recoverFieldsIndividually(cleaned);
  if (attempt3 !== null) {
    return attempt3 as T;
  }

  return null;
}

function closeOpenBrackets(cleaned: string): Record<string, unknown> | null {
  let curlyDepth = 0, squareDepth = 0;
  let inString = false, escape = false;

  for (const ch of cleaned) {
    if (escape) { escape = false; continue; }
    if (ch === "\\" && inString) { escape = true; continue; }
    if (ch === '"') { inString = !inString; continue; }
    if (inString) continue;
    if (ch === "{") curlyDepth++;
    else if (ch === "}") curlyDepth--;
    else if (ch === "[") squareDepth++;
    else if (ch === "]") squareDepth--;
  }

  if (curlyDepth === 0 && squareDepth === 0) return null;

  let attempt = findLastSafeTruncation(cleaned);
  if (attempt === null) return null;

  let newCurly = 0, newSquare = 0;
  let inStr = false, esc = false;
  for (const ch of attempt) {
    if (esc) { esc = false; continue; }
    if (ch === "\\" && inStr) { esc = true; continue; }
    if (ch === '"') { inStr = !inStr; continue; }
    if (inStr) continue;
    if (ch === "{") newCurly++;
    else if (ch === "}") newCurly--;
    else if (ch === "[") newSquare++;
    else if (ch === "]") newSquare--;
  }

  for (let i = 0; i < newSquare; i++) attempt += "]";
  for (let i = 0; i < newCurly; i++) attempt += "}";

  attempt = attempt.replace(/,\s*([}\]])/g, "$1");

  try {
    return JSON.parse(attempt) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function findLastSafeTruncation(text: string): string | null {
  const stringState: boolean[] = new Array(text.length).fill(false);
  let isStr = false, isEsc = false;
  for (let j = 0; j < text.length; j++) {
    if (isEsc) { isEsc = false; stringState[j] = isStr; continue; }
    if (text[j] === "\\" && isStr) { isEsc = true; stringState[j] = isStr; continue; }
    if (text[j] === '"') isStr = !isStr;
    stringState[j] = isStr;
  }

  const endsInString = stringState[text.length - 1];

  if (endsInString) {
    let stringStartQuote = -1;
    for (let j = text.length - 1; j >= 0; j--) {
      if (text[j] === '"') {
        const stateBefore = j > 0 ? stringState[j - 1] : false;
        const stateAfter = stringState[j];
        if (!stateBefore && stateAfter) {
          stringStartQuote = j;
          break;
        }
      }
    }

    if (stringStartQuote >= 0) {
      let beforeQuote = stringStartQuote - 1;
      while (beforeQuote >= 0 && /\s/.test(text[beforeQuote])) beforeQuote--;

      if (beforeQuote >= 0) {
        const prevChar = text[beforeQuote];
        if (prevChar === ":") {
          let objOpenIdx = -1;
          let depth = 0;
          for (let k = beforeQuote; k >= 0; k--) {
            if (text[k] === "}" && !stringState[k]) depth++;
            else if (text[k] === "{" && !stringState[k]) {
              if (depth === 0) { objOpenIdx = k; break; }
              depth--;
            }
          }
          if (objOpenIdx >= 0) {
            let beforeObj = objOpenIdx - 1;
            while (beforeObj >= 0 && /\s/.test(text[beforeObj])) beforeObj--;
            if (beforeObj >= 0 && text[beforeObj] === ",") {
              return text.slice(0, beforeObj).replace(/,\s*$/, "");
            }
            if (beforeObj >= 0 && text[beforeObj] === "[") {
              return text.slice(0, beforeObj + 1);
            }
            if (beforeObj >= 0 && text[beforeObj] === "{") {
              return text.slice(0, beforeObj + 1);
            }
            return text.slice(0, objOpenIdx).replace(/,\s*$/, "");
          }
          return text.slice(0, beforeQuote).replace(/,\s*$/, "");
        }
        if (prevChar === "," || prevChar === "[") {
          return text.slice(0, beforeQuote + (prevChar === "[" ? 1 : 0)).replace(/,\s*$/, "");
        }
      }
      return text.slice(0, stringStartQuote).replace(/,\s*$/, "");
    }
  }

  let i = text.length - 1;
  while (i >= 0 && /\s/.test(text[i])) i--;
  if (i < 0) return null;

  const ch = text[i];

  if ((ch === "}" || ch === "]") && !stringState[i]) {
    let end = i;
    let k = i - 1;
    while (k >= 0 && /\s/.test(text[k])) k--;
    if (k >= 0 && text[k] === ",") end = k;
    return text.slice(0, end + 1).replace(/,\s*$/, "");
  }

  if (ch === '"' && !stringState[i]) {
    let result = text.slice(0, i + 1);
    result = result.replace(/,\s*$/, "");
    return result;
  }

  if (/\d/.test(ch)) {
    let numEnd = i;
    let j = i;
    while (j >= 0 && /[0-9.eE+\-]/.test(text[j])) j--;
    let beforeNum = j;
    while (beforeNum >= 0 && /\s/.test(text[beforeNum])) beforeNum--;
    if (beforeNum >= 0 && (text[beforeNum] === ":" || text[beforeNum] === "," || text[beforeNum] === "[")) {
      let result = text.slice(0, numEnd + 1);
      result = result.replace(/,\s*$/, "");
      return result;
    }
  }

  for (let j = text.length - 1; j >= 0; j--) {
    if ((text[j] === "}" || text[j] === "]") && !stringState[j]) {
      let result = text.slice(0, j + 1);
      result = result.replace(/,\s*$/, "");
      return result;
    }
  }

  return null;
}

function truncateToLastCompleteElement(cleaned: string): Record<string, unknown> | null {
  const stringState: boolean[] = new Array(cleaned.length).fill(false);
  let isStr = false, isEsc = false;
  for (let j = 0; j < cleaned.length; j++) {
    if (isEsc) { isEsc = false; stringState[j] = isStr; continue; }
    if (cleaned[j] === "\\" && isStr) { isEsc = true; stringState[j] = isStr; continue; }
    if (cleaned[j] === '"') isStr = !isStr;
    stringState[j] = isStr;
  }

  let lastCompleteObjEnd = -1;
  for (let j = cleaned.length - 1; j >= 0; j--) {
    if (cleaned[j] === "}" && !stringState[j]) { lastCompleteObjEnd = j; break; }
  }

  let lastCompleteArrEnd = -1;
  for (let j = cleaned.length - 1; j >= 0; j--) {
    if (cleaned[j] === "]" && !stringState[j]) { lastCompleteArrEnd = j; break; }
  }

  let bestIdx = Math.max(lastCompleteObjEnd, lastCompleteArrEnd);
  if (bestIdx <= 0) return null;

  let truncated = cleaned.slice(0, bestIdx + 1);
  truncated = truncated.replace(/,\s*$/, "");

  let curly = 0, square = 0;
  let inStr = false, esc = false;
  for (const ch of truncated) {
    if (esc) { esc = false; continue; }
    if (ch === "\\" && inStr) { esc = true; continue; }
    if (ch === '"') { inStr = !inStr; continue; }
    if (inStr) continue;
    if (ch === "{") curly++;
    else if (ch === "}") curly--;
    else if (ch === "[") square++;
    else if (ch === "]") square--;
  }

  for (let i = 0; i < square; i++) truncated += "]";
  for (let i = 0; i < curly; i++) truncated += "}";
  truncated = truncated.replace(/,\s*([}\]])/g, "$1");

  try {
    return JSON.parse(truncated) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function recoverFieldsIndividually(cleaned: string): Record<string, unknown> | null {
  try {
    const result: Record<string, unknown> = {};

    const nodesMatch = cleaned.match(/"nodes"\s*:\s*\[/);
    if (nodesMatch) {
      const nodesStart = cleaned.indexOf("[", cleaned.indexOf('"nodes"'));
      const recoveredNodes = recoverArrayElements(cleaned, nodesStart);
      if (recoveredNodes && recoveredNodes.length > 0) result.nodes = recoveredNodes;
    }

    const edgesMatch = cleaned.match(/"edges"\s*:\s*\[/);
    if (edgesMatch) {
      const edgesStart = cleaned.indexOf("[", cleaned.indexOf('"edges"'));
      const recoveredEdges = recoverArrayElements(cleaned, edgesStart);
      if (recoveredEdges && recoveredEdges.length > 0) result.edges = recoveredEdges;
    }

    const scoreMatch = cleaned.match(/"score"\s*:\s*([\d.]+)/);
    if (scoreMatch) result.score = parseFloat(scoreMatch[1]);

    const issuesMatch = cleaned.match(/"issues"\s*:\s*\[/);
    if (issuesMatch) {
      const issuesStart = cleaned.indexOf("[", cleaned.indexOf('"issues"'));
      const recoveredIssues = recoverArrayElements(cleaned, issuesStart);
      if (recoveredIssues && recoveredIssues.length > 0) result.issues = recoveredIssues;
    }

    const suggestionsMatch = cleaned.match(/"suggestions"\s*:\s*\[/);
    if (suggestionsMatch) {
      const suggestionsStart = cleaned.indexOf("[", cleaned.indexOf('"suggestions"'));
      const recoveredSuggestions = recoverArrayElements(cleaned, suggestionsStart);
      if (recoveredSuggestions && recoveredSuggestions.length > 0) result.suggestions = recoveredSuggestions;
    }

    if (Object.keys(result).length > 0) return result;
  } catch {
    // Recovery failed
  }

  return null;
}

function recoverArrayElements(text: string, arrayStartIndex: number): unknown[] | null {
  if (arrayStartIndex === -1) return null;

  const elements: unknown[] = [];
  let depth = 0;
  let objStart = -1;
  let inString = false;
  let escape = false;

  for (let i = arrayStartIndex; i < text.length; i++) {
    const ch = text[i];

    if (escape) { escape = false; continue; }
    if (ch === "\\" && inString) { escape = true; continue; }
    if (ch === '"') { inString = !inString; continue; }
    if (inString) continue;

    if (ch === "{") {
      if (depth === 0) objStart = i;
      depth++;
    } else if (ch === "}") {
      depth--;
      if (depth === 0 && objStart !== -1) {
        const objStr = text.slice(objStart, i + 1);
        try { elements.push(JSON.parse(objStr)); } catch { /* skip */ }
        objStart = -1;
      }
    }
  }

  return elements;
}
