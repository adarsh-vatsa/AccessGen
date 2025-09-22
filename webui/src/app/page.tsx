"use client";
import React from "react";

export default function Home() {
  return (
    <div className="font-sans grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20">
      <main className="flex flex-col gap-[24px] row-start-2 items-center sm:items-start w-full max-w-4xl">
        <h1 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">IAM Policy Generator UI</h1>
        <p className="text-sm text-neutral-700 dark:text-neutral-300 max-w-prose">
          Send a natural language requirement to the local Python pipeline and view the generated IAM policy and test configuration.
        </p>
        <QueryForm />
      </main>
    </div>
  );
}

function QueryForm() {
  const [query, setQuery] = React.useState("");
  const [services, setServices] = React.useState<string[]>([]);
  const [threshold, setThreshold] = React.useState<string>("");
  const [maxActions, setMaxActions] = React.useState<string>("15");
  const [expand, setExpand] = React.useState<boolean>(true);
  const [raw, setRaw] = React.useState<boolean>(false);
  const [mode, setMode] = React.useState<"vector" | "raw" | "adversarial">("vector");
  const [rounds, setRounds] = React.useState<string>("1");
  const [judgeModel, setJudgeModel] = React.useState<string>("");
  const [proModel, setProModel] = React.useState<string>("");
  const [conModel, setConModel] = React.useState<string>("");
  const [draftModel, setDraftModel] = React.useState<string>("");
  const [useDSPy, setUseDSPy] = React.useState<boolean>(false);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [result, setResult] = React.useState<any>(null);

  function toggleService(s: string) {
    setServices((prev) => (prev.includes(s) ? prev.filter((x) => x !== s) : [...prev, s]));
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const payload: any = { query };
      if (services.length > 0) payload.services = services;
      if (threshold) payload.threshold = parseFloat(threshold);
      if (maxActions) payload.maxActions = parseInt(maxActions, 10);
      if (!expand) payload.noExpand = true;
      if (raw) payload.raw = true;
      payload.mode = mode;
      if (mode === "adversarial") {
        if (rounds) payload.rounds = Math.max(1, Math.min(3, parseInt(rounds, 10) || 1));
        if (judgeModel) payload.judgeModel = judgeModel;
        if (proModel) payload.proModel = proModel;
        if (conModel) payload.conModel = conModel;
        if (draftModel) payload.draftModel = draftModel;
        if (useDSPy) payload.useDSPy = true;
      }

      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || "Request failed");
      setResult(json);
    } catch (err: any) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <form onSubmit={onSubmit} className="w-full space-y-4">
      <label className="block text-sm font-medium text-neutral-900 dark:text-neutral-100">Query</label>
      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        className="w-full h-28 rounded-md border border-neutral-300 dark:border-neutral-700 p-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-600 bg-white dark:bg-neutral-900 text-neutral-900 dark:text-neutral-100 placeholder-neutral-500 dark:placeholder-neutral-400 shadow-sm"
        placeholder="e.g., EC2 instances need to upload logs to S3 and read them back"
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="block text-sm font-medium mb-1 text-neutral-900 dark:text-neutral-100">Services</label>
          <div className="flex gap-3 text-sm">
            {(["s3", "ec2", "iam"] as const).map((s) => (
              <label key={s} className="flex items-center gap-2">
                <input type="checkbox" checked={services.includes(s)} onChange={() => toggleService(s)} />
                <span className="uppercase">{s}</span>
              </label>
            ))}
          </div>
          <p className="text-xs text-neutral-600 dark:text-neutral-400 mt-1">Leave empty for auto-detect.</p>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1 text-neutral-900 dark:text-neutral-100">Score threshold</label>
          <input
            type="number"
            step="0.0001"
            value={threshold}
            onChange={(e) => setThreshold(e.target.value)}
            className="w-full rounded-md border border-neutral-300 dark:border-neutral-700 p-2 text-sm bg-white dark:bg-neutral-900 text-neutral-900 dark:text-neutral-100 placeholder-neutral-500 dark:placeholder-neutral-400 shadow-sm"
            placeholder="0.0005"
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1 text-neutral-900 dark:text-neutral-100">Max actions</label>
          <input
            type="number"
            value={maxActions}
            onChange={(e) => setMaxActions(e.target.value)}
            className="w-full rounded-md border border-neutral-300 dark:border-neutral-700 p-2 text-sm bg-white dark:bg-neutral-900 text-neutral-900 dark:text-neutral-100 placeholder-neutral-500 dark:placeholder-neutral-400 shadow-sm"
            placeholder="15"
            min={1}
            max={50}
          />
        </div>
      </div>

      <label className="inline-flex items-center gap-2 text-sm text-neutral-900 dark:text-neutral-100">
        <input type="checkbox" checked={expand} onChange={(e) => setExpand(e.target.checked)} />
        Enable query expansion
      </label>

      <label className="inline-flex items-center gap-2 text-sm text-neutral-900 dark:text-neutral-100">
        <input type="checkbox" checked={raw} onChange={(e) => setRaw(e.target.checked)} />
        RAW mode (no vector search context)
      </label>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="block text-sm font-medium mb-1 text-neutral-900 dark:text-neutral-100">Mode</label>
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as any)}
            className="w-full rounded-md border border-neutral-300 dark:border-neutral-700 p-2 text-sm bg-white dark:bg-neutral-900 text-neutral-900 dark:text-neutral-100 shadow-sm"
          >
            <option value="vector">Vector</option>
            <option value="raw">RAW</option>
            <option value="adversarial">Adversarial (Pro)</option>
          </select>
        </div>
        {mode === "adversarial" && (
          <div>
            <label className="block text-sm font-medium mb-1 text-neutral-900 dark:text-neutral-100">Rounds (1–3)</label>
            <input
              type="number"
              min={1}
              max={3}
              value={rounds}
              onChange={(e) => setRounds(e.target.value)}
              className="w-full rounded-md border border-neutral-300 dark:border-neutral-700 p-2 text-sm bg-white dark:bg-neutral-900 text-neutral-900 dark:text-neutral-100 shadow-sm"
            />
          </div>
        )}
      </div>

      {mode === "adversarial" && (
        <details className="rounded-md border border-neutral-300 dark:border-neutral-700 p-3">
          <summary className="cursor-pointer text-sm font-medium text-neutral-900 dark:text-neutral-100">Advanced Models</summary>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-3">
            <div>
              <label className="block text-xs mb-1">Draft model</label>
              <input list="modelOptions" value={draftModel} onChange={(e) => setDraftModel(e.target.value)} className="w-full rounded-md border border-neutral-300 dark:border-neutral-700 p-2 text-sm bg-white dark:bg-neutral-900" placeholder="models/gemini-2.5-pro" />
            </div>
            <div>
              <label className="block text-xs mb-1">Judge model</label>
              <input list="modelOptions" value={judgeModel} onChange={(e) => setJudgeModel(e.target.value)} className="w-full rounded-md border border-neutral-300 dark:border-neutral-700 p-2 text-sm bg-white dark:bg-neutral-900" placeholder="models/gemini-2.5-flash" />
            </div>
            <div>
              <label className="block text-xs mb-1">Proponent model</label>
              <input list="modelOptions" value={proModel} onChange={(e) => setProModel(e.target.value)} className="w-full rounded-md border border-neutral-300 dark:border-neutral-700 p-2 text-sm bg-white dark:bg-neutral-900" placeholder="" />
            </div>
            <div>
              <label className="block text-xs mb-1">Opponent model</label>
              <input list="modelOptions" value={conModel} onChange={(e) => setConModel(e.target.value)} className="w-full rounded-md border border-neutral-300 dark:border-neutral-700 p-2 text-sm bg-white dark:bg-neutral-900" placeholder="" />
            </div>
            <div className="md:col-span-2">
              <label className="inline-flex items-center gap-2 text-sm">
                <input type="checkbox" checked={useDSPy} onChange={(e) => setUseDSPy(e.target.checked)} />
                Use DSPy debate (if installed)
              </label>
            </div>
            {/* Shared datalist for quick model selection, still allows free text */}
            <datalist id="modelOptions">
              {/* Google Gemini generation */}
              <option value="models/gemini-2.5-pro" />
              <option value="models/gemini-2.5-flash" />
              <option value="models/gemini-2.0-flash-exp" />
              {/* OpenAI common */}
              <option value="gpt-4o" />
              <option value="gpt-4o-mini" />
              <option value="gpt-4.1" />
              <option value="gpt-4.1-mini" />
              {/* Others (Anthropic/OpenRouter style - optional examples) */}
              <option value="claude-3-5-sonnet" />
              <option value="claude-3-haiku" />
            </datalist>
          </div>
        </details>
      )}

      <div className="flex items-center gap-3">
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="px-4 py-2 rounded-md bg-blue-600 hover:bg-blue-700 text-white disabled:opacity-50 shadow-sm"
        >
          {loading ? "Generating..." : "Generate"}
        </button>
        {error && <span className="text-red-600 text-sm">{error}</span>}
      </div>

      {result && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <JsonCard title="IAM Policy" data={result.iam_policy} />
          <JsonCard title="Test Config" data={result.test_config} />
          <JsonCard title="Metadata" data={result.metadata} />
          <ProvenanceCard result={result} />
          {result.metadata?.mode === "adversarial" && (
            <div className="rounded-lg border border-neutral-200 dark:border-neutral-700 p-4 bg-white dark:bg-neutral-900 shadow-sm w-full md:col-span-2">
              <h2 className="font-semibold mb-2 text-neutral-900 dark:text-neutral-100">Debate & Registry</h2>
              <div className="text-xs space-y-2 text-neutral-900 dark:text-neutral-100">
                <div>
                  <strong>Debate summary:</strong> {result.metadata.debate?.summary || ""}
                </div>
                <div>
                  <strong>Mismatches:</strong> {(result.metadata.registry?.mismatches || []).join(", ")}
                </div>
                <div>
                  <strong>Replacements:</strong> {JSON.stringify(result.metadata.registry?.replacements || {})}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </form>
  );
}

function JsonCard({ title, data }: { title: string; data: any }) {
  return (
    <div className="rounded-lg border border-neutral-200 dark:border-neutral-700 p-4 bg-white dark:bg-neutral-900 shadow-sm w-full">
      <h2 className="font-semibold mb-2 text-neutral-900 dark:text-neutral-100">{title}</h2>
      <pre className="text-xs overflow-auto max-h-[60vh] whitespace-pre-wrap text-neutral-900 dark:text-neutral-100">
        {JSON.stringify(data, null, 2)}
      </pre>
    </div>
  );
}

function ProvenanceCard({ result }: { result: any }) {
  const policy = result?.iam_policy;
  const prov = result?.metadata?.registry?.provenance || {};
  const [open, setOpen] = React.useState(true);
  if (!policy || !Array.isArray(policy?.Statement)) return null;

  // Collect unique actions from policy
  const actions: string[] = [];
  for (const st of policy.Statement) {
    let acts = st?.Action;
    if (!acts) continue;
    if (typeof acts === "string") acts = [acts];
    for (const a of acts) {
      if (typeof a === "string" && !actions.includes(a)) actions.push(a);
    }
  }
  if (actions.length === 0) return null;

  const truncate = (s: string, n = 140) => (s && s.length > n ? s.slice(0, n - 1) + "…" : s || "");

  return (
    <div className="rounded-lg border border-neutral-200 dark:border-neutral-700 p-4 bg-white dark:bg-neutral-900 shadow-sm w-full md:col-span-2">
      <div className="flex items-center justify-between">
        <h2 className="font-semibold text-neutral-900 dark:text-neutral-100">Provenance</h2>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="text-xs px-2 py-1 rounded border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-50 dark:hover:bg-neutral-800"
        >
          {open ? "Collapse" : "Expand"}
        </button>
      </div>
      {open && (
        <div className="mt-3 text-xs space-y-2 text-neutral-900 dark:text-neutral-100">
          {actions.map((act) => {
            const meta = prov[act] || {};
            const svc = meta?.service || (typeof act === "string" ? act.split(":")[0] : "");
            const src = meta?.source || {};
            const table = src?.table ?? "actions";
            const row = src?.row_index ?? "?";
            const url = src?.url || "";
            const why = truncate(meta?.description || "");
            return (
              <div key={act} className="flex items-start justify-between gap-3 border border-neutral-200 dark:border-neutral-700 rounded p-2">
                <div>
                  <div className="font-mono font-semibold">{act}</div>
                  <div className="opacity-80">Service: {svc}</div>
                  <div className="opacity-80">Table: {table} • Row: {row}</div>
                  {why && (
                    <div className="mt-1 opacity-90"><span className="font-semibold">Why:</span> {why}</div>
                  )}
                </div>
                {url ? (
                  <a className="text-blue-600 hover:underline break-all" href={url} target="_blank" rel="noreferrer">
                    source
                  </a>
                ) : (
                  <span className="opacity-60">no source url</span>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
