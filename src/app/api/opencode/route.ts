import { NextRequest, NextResponse } from "next/server";

const OPENCODE_BASE_URL = "https://opencode.ai/zen/go";

// ─── API Key Resolution ─────────────────────────────────────
// Priority: client-provided key > OPENCODE_API_KEY env var
// For Vercel deployment: set OPENCODE_API_KEY in your project's
// Environment Variables dashboard and no client config is needed.

// ─── Retry Configuration ────────────────────────────────────
const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 15_000; // 15 seconds between retry attempts

/**
 * Status codes that warrant a retry (transient / rate-limit errors).
 */
const RETRYABLE_STATUS_CODES = new Set([429, 500, 502, 503, 504]);

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function describeStatus(status: number): string {
  switch (status) {
    case 400:
      return "Bad Request — the prompt or parameters are invalid";
    case 401:
      return "Unauthorized — check your OpenCode API key";
    case 403:
      return "Forbidden — your API key does not have access to this model";
    case 404:
      return "Not Found — the model endpoint does not exist";
    case 429:
      return "Rate Limited — too many requests, retrying…";
    case 500:
      return "Server Error — OpenCode experienced an internal error";
    case 502:
      return "Bad Gateway — OpenCode's upstream is unreachable";
    case 503:
      return "Service Unavailable — OpenCode is temporarily offline";
    case 504:
      return "Gateway Timeout — OpenCode took too long to respond";
    default:
      return `HTTP ${status}`;
  }
}

/**
 * Server-side proxy for OpenCode API.
 *
 * Why: The browser blocks direct requests to opencode.ai
 * due to CORS. This API route forwards requests from the browser
 * to OpenCode's servers, bypassing CORS.
 *
 * Features:
 * - Retry up to 3 times on transient errors (429, 5xx) with 15-second delay
 * - Graceful error handling with descriptive messages
 * - reasoning_effort: "none" disables thinking so all tokens go to output
 * - Passes through retry metadata so the client can display status
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { apiKey, model, messages, temperature, max_tokens } = body;

    // Resolve API key: client override > server env var
    const effectiveApiKey = apiKey || process.env.OPENCODE_API_KEY;

    if (!effectiveApiKey) {
      return NextResponse.json(
        { error: "API key is required. Set OPENCODE_API_KEY environment variable or provide it in the UI." },
        { status: 400 }
      );
    }

    if (!model) {
      return NextResponse.json(
        { error: "Model is required" },
        { status: 400 }
      );
    }

    if (!messages || !Array.isArray(messages)) {
      return NextResponse.json(
        { error: "Messages array is required" },
        { status: 400 }
      );
    }

    // Build request body with reasoning_effort: "none" to disable thinking
    // This ensures all max_tokens go to content output, preventing the
    // "empty content" bug where reasoning consumes all tokens.
    const requestBody = JSON.stringify({
      model,
      messages,
      temperature: temperature ?? 0.7,
      max_tokens: max_tokens ?? 16384,
      reasoning_effort: "none",
      stream: false,
    });

    // ─── Retry Loop ───────────────────────────────────────────
    let lastError: string | null = null;
    let lastStatus = 0;

    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
      let apiResponse: Response;

      try {
        apiResponse = await fetch(`${OPENCODE_BASE_URL}/v1/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${effectiveApiKey}`,
          },
          body: requestBody,
        });
      } catch (fetchError) {
        const msg =
          fetchError instanceof Error
            ? fetchError.message
            : "Network error contacting OpenCode API";

        lastError = `Network error: ${msg}`;
        lastStatus = 0;

        if (attempt < MAX_RETRIES) {
          console.warn(
            `[OpenCode Proxy] Attempt ${attempt}/${MAX_RETRIES} failed — ${msg}. Retrying in ${RETRY_DELAY_MS / 1000}s…`
          );
          await sleep(RETRY_DELAY_MS);
          continue;
        }
        break;
      }

      // ─── Success ───────────────────────────────────────────
      if (apiResponse.ok) {
        const data = await apiResponse.json();

        // Handle reasoning models: prefer content, fall back to reasoning_content
        const message = data.choices?.[0]?.message;
        if (message && !message.content && message.reasoning_content) {
          message.content = message.reasoning_content;
        }

        // Attach retry metadata
        if (attempt > 1) {
          data._retryMeta = {
            attempts: attempt,
            succeeded: true,
          };
        }

        return NextResponse.json(data);
      }

      // ─── Non-retryable error ──────────────────────────────
      const errorText = await apiResponse.text().catch(() => "Unknown error");
      lastStatus = apiResponse.status;
      lastError = errorText.slice(0, 500);

      if (!RETRYABLE_STATUS_CODES.has(apiResponse.status)) {
        console.error(
          `[OpenCode Proxy] Non-retryable error (${apiResponse.status}): ${lastError}`
        );
        return NextResponse.json(
          {
            error: describeStatus(apiResponse.status),
            details: lastError,
            retryable: false,
          },
          { status: apiResponse.status }
        );
      }

      // ─── Retryable error ───────────────────────────────────
      console.warn(
        `[OpenCode Proxy] Attempt ${attempt}/${MAX_RETRIES} — ${describeStatus(apiResponse.status)}. ${
          attempt < MAX_RETRIES
            ? `Retrying in ${RETRY_DELAY_MS / 1000}s…`
            : "No more retries."
        }`
      );

      if (attempt < MAX_RETRIES) {
        await sleep(RETRY_DELAY_MS);
      }
    }

    // ─── All retries exhausted ───────────────────────────────
    return NextResponse.json(
      {
        error: lastStatus
          ? `${describeStatus(lastStatus)} — all ${MAX_RETRIES} attempts failed`
          : `Network error — all ${MAX_RETRIES} attempts failed`,
        details: lastError,
        retryable: true,
        attemptsExhausted: true,
      },
      { status: lastStatus || 502 }
    );
  } catch (error) {
    console.error("[OpenCode Proxy] Internal error:", error);
    return NextResponse.json(
      {
        error: "Internal proxy error",
        details: error instanceof Error ? error.message : "Unknown error",
        retryable: false,
      },
      { status: 500 }
    );
  }
}
