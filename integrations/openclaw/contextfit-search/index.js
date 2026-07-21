import { definePluginEntry, delegateCompactionToRuntime } from "openclaw/plugin-sdk/core";

const METHODS = ["hybrid", "exact", "bm25", "semantic", "sid", "graph"];

function readString(params, key, fallback) {
  const value = params?.[key];
  if (typeof value !== "string") return fallback;
  const trimmed = value.trim();
  return trimmed.length ? trimmed : fallback;
}

function readBool(params, key, fallback) {
  const value = params?.[key];
  return typeof value === "boolean" ? value : fallback;
}

function readInt(params, key, fallback, min, max) {
  const value = params?.[key];
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, Math.trunc(n)));
}

function cleanBaseUrl(raw) {
  const value = typeof raw === "string" && raw.trim() ? raw.trim() : "http://127.0.0.1:8765";
  return value.replace(/\/+$/, "");
}

function getChunks(payload) {
  return Array.isArray(payload?.results) ? payload.results
    : Array.isArray(payload?.chunks) ? payload.chunks
    : Array.isArray(payload) ? payload
    : [];
}

function compactResult(result) {
  const source = result.source || result.path || result.file || result.metadata?.source || result.id;
  return {
    source,
    score: result.score,
    method: result.method,
    chunk_id: result.chunk_id ?? result.chunkId ?? result.id,
    line_start: result.line_start ?? result.start_line ?? result.startLine ?? result.line,
    line_end: result.line_end ?? result.end_line ?? result.endLine,
    spans: result.spans ?? result.matches ?? undefined,
    tmd_rows: result.tmd_rows ?? result.tmdRows ?? undefined,
    text: result.text ?? result.preview ?? result.content ?? undefined
  };
}

function summarize(payload) {
  const results = getChunks(payload);
  if (!results.length) return "ContextFit found no matching results.";
  return results.slice(0, 5).map((raw, index) => {
    const r = compactResult(raw);
    const where = [r.source, r.line_start ? `:${r.line_start}${r.line_end && r.line_end !== r.line_start ? `-${r.line_end}` : ""}` : ""].join("");
    const rows = Array.isArray(r.tmd_rows) && r.tmd_rows.length
      ? `\n  TMD rows: ${r.tmd_rows.map((row) => row.id || row.row_id || row.key || JSON.stringify(row)).join(", ")}`
      : "";
    const spans = Array.isArray(r.spans) && r.spans.length
      ? `\n  Spans: ${r.spans.slice(0, 3).map((span) => span.text || span.match || JSON.stringify(span)).join(" | ")}`
      : "";
    const preview = typeof r.text === "string" && r.text.trim()
      ? `\n  Preview: ${r.text.trim().replace(/\s+/g, " ").slice(0, 500)}`
      : "";
    const score = typeof r.score === "number" ? ` score=${r.score.toFixed(4)}` : "";
    return `${index + 1}. ${where || "(unknown source)"}${score}${rows}${spans}${preview}`;
  }).join("\n");
}

function contextFitConfig(api) {
  const cfg = api.pluginConfig || {};
  const engine = cfg.engine && typeof cfg.engine === "object" ? cfg.engine : {};
  return {
    baseUrl: cleanBaseUrl(engine.baseUrl || cfg.baseUrl),
    accessKey: readString(engine, "accessKey", readString(cfg, "accessKey", "")),
    defaultKb: readString(engine, "defaultKb", readString(cfg, "defaultKb", "memory")),
    autoRecall: readBool(engine, "autoRecall", true),
    topK: readInt(engine, "topK", 5, 1, 20),
    method: METHODS.includes(engine.method) ? engine.method : "hybrid",
    includeText: readBool(engine, "includeText", false),
    returnSpans: readBool(engine, "returnSpans", true),
    maxPromptChars: readInt(engine, "maxPromptChars", 2000, 100, 12000),
    maxAdditionChars: readInt(engine, "maxAdditionChars", 4000, 500, 20000)
  };
}

async function queryContextFit(cfg, query, overrides = {}) {
  const body = {
    kb: overrides.kb || cfg.defaultKb || "memory",
    query,
    method: overrides.method || cfg.method || "hybrid",
    top_k: overrides.topK || cfg.topK || 5,
    include_text: overrides.includeText ?? cfg.includeText ?? false,
    return_spans: overrides.returnSpans ?? cfg.returnSpans ?? true
  };
  const response = await fetch(`${cfg.baseUrl}/query`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...(cfg.accessKey ? { "x-contextfit-access-key": cfg.accessKey } : {})
    },
    body: JSON.stringify(body)
  });
  const text = await response.text();
  let payload;
  try { payload = text ? JSON.parse(text) : null; } catch { payload = { raw: text }; }
  if (!response.ok) throw new Error(`ContextFit query failed (${response.status}): ${text.slice(0, 1000)}`);
  return { body, payload, results: getChunks(payload).map(compactResult) };
}

function extractMessageText(message) {
  const content = message?.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) return content.map((part) => {
    if (typeof part === "string") return part;
    if (typeof part?.text === "string") return part.text;
    if (typeof part?.content === "string") return part.content;
    return "";
  }).filter(Boolean).join("\n");
  return "";
}

function findPrompt(params) {
  if (typeof params.prompt === "string" && params.prompt.trim()) return params.prompt.trim();
  const messages = Array.isArray(params.messages) ? params.messages : [];
  for (let i = messages.length - 1; i >= 0; i--) {
    const role = messages[i]?.role || messages[i]?.speaker;
    if (role && role !== "user") continue;
    const text = extractMessageText(messages[i]).trim();
    if (text) return text;
  }
  return "";
}

function formatEngineRecall(results, request) {
  if (!results.length) return "";
  const lines = [
    "ContextFit auto-recall candidates (handle-first; expand/cite source before relying on details):"
  ];
  for (const [i, r] of results.slice(0, request.top_k || 5).entries()) {
    const loc = `${r.source || "unknown"}${r.line_start ? `#L${r.line_start}${r.line_end && r.line_end !== r.line_start ? `-L${r.line_end}` : ""}` : ""}`;
    const score = typeof r.score === "number" ? ` score=${r.score.toFixed(3)}` : "";
    const rows = Array.isArray(r.tmd_rows) && r.tmd_rows.length ? ` rows=${r.tmd_rows.map(row => row.row_id || row.id || row.key).filter(Boolean).join(",")}` : "";
    const matches = Array.isArray(r.spans) && r.spans.length ? ` matches=${r.spans.slice(0, 5).map(s => s.match || s.text).filter(Boolean).join("|")}` : "";
    lines.push(`${i + 1}. ${loc}${score}${rows}${matches}`.slice(0, 900));
  }
  return lines.join("\n");
}

function estimateTokens(messages, addition = "") {
  const messageChars = (Array.isArray(messages) ? messages : []).reduce((sum, m) => sum + JSON.stringify(m).length, 0);
  return Math.ceil((messageChars + addition.length) / 4);
}

const ContextFitSearchParams = {
  type: "object",
  additionalProperties: false,
  required: ["query"],
  properties: {
    query: { type: "string", description: "Search query string." },
    kb: { type: "string", description: "ContextFit KB name. Defaults to plugin config defaultKb or memory." },
    method: { type: "string", enum: METHODS, description: "Search method. Use hybrid by default; exact for token-exact lookups." },
    top_k: { type: "number", description: "Number of results to return (1-50). Defaults to 5.", minimum: 1, maximum: 50 },
    include_text: { type: "boolean", description: "Include result text/previews. Defaults to true." },
    return_spans: { type: "boolean", description: "Return proof spans/matches where supported. Defaults to true." },
    base_url: { type: "string", description: "Override ContextFit server URL for this call." }
  }
};

export default definePluginEntry({
  id: "contextfit-search",
  name: "ContextFit Search",
  description: "ContextFit search tool and context engine for OpenClaw workspace knowledge bases.",
  kind: "context-engine",
  register(api) {
    api.registerTool({
      name: "contextfit_search",
      label: "ContextFit Search",
      description: "Search OpenClaw workspace knowledge bases through ContextFit. Prefer method='hybrid' for most KB lookups, method='exact' for exact purchase/order IDs, names, and phrases. Returns source paths, proof spans, and TMD rows when available.",
      parameters: ContextFitSearchParams,
      async execute(_toolCallId, rawParams) {
        const params = rawParams && typeof rawParams === "object" ? rawParams : {};
        const query = readString(params, "query");
        if (!query) throw new Error("contextfit_search requires query.");
        const cfg = contextFitConfig(api);
        if (params.base_url) cfg.baseUrl = cleanBaseUrl(params.base_url);
        const method = METHODS.includes(params.method) ? params.method : "hybrid";
        let result;
        try {
          result = await queryContextFit(cfg, query, {
            kb: readString(params, "kb", cfg.defaultKb),
            method,
            topK: readInt(params, "top_k", 5, 1, 50),
            includeText: readBool(params, "include_text", true),
            returnSpans: readBool(params, "return_spans", true)
          });
        } catch (error) {
          throw new Error(`ContextFit query failed: ${error?.message || error}`);
        }
        return {
          content: [{ type: "text", text: summarize(result.payload) }],
          details: { status: "ok", request: result.body, baseUrl: cfg.baseUrl, results: result.results, raw: result.payload }
        };
      }
    });

    const createContextFitEngine = (engineId) => ({
      info: {
        id: engineId,
        name: "ContextFit Context Engine",
        version: "0.1.1",
        ownsCompaction: false
      },
      async ingest() {
        return { ingested: false };
      },
      async ingestBatch(params) {
        return { ingestedCount: Array.isArray(params.messages) ? 0 : 0 };
      },
      async assemble(params) {
        const cfg = contextFitConfig(api);
        let systemPromptAddition = "";
        if (cfg.autoRecall) {
          const prompt = findPrompt(params).slice(0, cfg.maxPromptChars);
          if (prompt) {
            try {
              const result = await queryContextFit(cfg, prompt, { includeText: cfg.includeText, returnSpans: cfg.returnSpans });
              systemPromptAddition = formatEngineRecall(result.results, result.body).slice(0, cfg.maxAdditionChars);
            } catch (error) {
              api.logger?.warn?.(`ContextFit auto-recall skipped: ${error?.message || error}`);
              systemPromptAddition = "ContextFit auto-recall is configured but unavailable for this turn; continue without it.";
            }
          }
        }
        return {
          messages: params.messages,
          estimatedTokens: estimateTokens(params.messages, systemPromptAddition),
          promptAuthority: "preassembly_may_overflow",
          systemPromptAddition
        };
      },
      async compact(params) {
        return delegateCompactionToRuntime(params);
      }
    });

    api.registerContextEngine("contextfit", () => createContextFitEngine("contextfit"));
    api.registerContextEngine("contextfit-search", () => createContextFitEngine("contextfit-search"));
  }
});
