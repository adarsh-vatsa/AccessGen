import { NextRequest } from "next/server";
import { spawn } from "child_process";
import fs from "node:fs";
import path from "node:path";

type GenerateBody = {
  query?: string;
  rounds?: number;
  judgeModel?: string;
  proModel?: string;
  conModel?: string;
  draftModel?: string;
  useDSPy?: boolean;
};

const REPO_ROOT = "/Users/zeitgeist/Documents/SPT/vector_search/Alpha/01_source_fetcher";
function resolvePythonBin(): string {
  if (process.env.PYTHON_BIN) return process.env.PYTHON_BIN;
  const venv = path.join(REPO_ROOT, ".venv");
  const posix = path.join(venv, "bin", "python");
  const win = path.join(venv, "Scripts", "python.exe");
  if (fs.existsSync(posix)) return posix;
  if (fs.existsSync(win)) return win;
  return "python3";
}
const PYTHON_BIN = resolvePythonBin();

export async function POST(req: NextRequest) {
  const body = (await req.json()) as GenerateBody;
  const query = (body.query || "").trim();
  const rounds = typeof body.rounds === "number" ? String(Math.max(1, Math.min(3, body.rounds))) : "1";
  const judgeModel = typeof body.judgeModel === "string" ? body.judgeModel : "";
  const proModel = typeof body.proModel === "string" ? body.proModel : "";
  const conModel = typeof body.conModel === "string" ? body.conModel : "";
  const draftModel = typeof body.draftModel === "string" ? body.draftModel : "";
  const useDSPy = body.useDSPy ? "1" : "0";

  const norm = (m?: string) => {
    if (!m) return "";
    const mm = m.trim();
    if (/^gemini/i.test(mm) && !mm.startsWith("models/")) return `models/${mm}`;
    return mm;
  };

  const pyCode = `
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path("${REPO_ROOT}")))
from src.stream_adversarial import run_stream
run_stream()
`;

  const stream = new ReadableStream({
    start(controller) {
      const venvDir = path.join(REPO_ROOT, ".venv");
      const env = {
        ...process.env,
        QUERY: query,
        ROUNDS: rounds,
        JUDGE_MODEL: norm(judgeModel),
        PRO_MODEL: norm(proModel),
        CON_MODEL: norm(conModel),
        DRAFT_MODEL: norm(draftModel),
        USE_DSPY: useDSPy,
        GENAI_FORCE_LEGACY: "1",
        VIRTUAL_ENV: fs.existsSync(venvDir) ? venvDir : (process.env.VIRTUAL_ENV || ""),
        PATH: fs.existsSync(path.join(venvDir, "bin"))
          ? `${path.join(venvDir, "bin")}:${process.env.PATH || ""}`
          : process.env.PATH,
        PWD: REPO_ROOT,
      } as NodeJS.ProcessEnv;

      const child = spawn(PYTHON_BIN, ["-c", pyCode], { cwd: REPO_ROOT, env });
      const enc = new TextEncoder();
      // kick off SSE immediately
      try { controller.enqueue(enc.encode(": connected\n\n")); } catch {}
      const abort = () => {
        try { child.kill("SIGTERM"); } catch {}
        try { controller.close(); } catch {}
      };
      try { (req as any).signal?.addEventListener?.("abort", abort); } catch {}
      child.stdout.on("data", (chunk) => controller.enqueue(enc.encode(chunk.toString())));
      child.stderr.on("data", (chunk) => {
        // forward stderr as comment lines to keep SSE alive
        const lines = chunk.toString().split(/\r?\n/).filter(Boolean);
        for (const l of lines) {
          controller.enqueue(enc.encode(`: ${l}\n\n`));
        }
      });
      child.on("close", () => controller.close());
      child.on("error", () => controller.close());
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
