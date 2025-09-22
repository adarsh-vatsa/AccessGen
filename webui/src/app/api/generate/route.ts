import { NextRequest } from "next/server";
import { spawn } from "child_process";

type GenerateBody = {
  query?: string;
  services?: string[] | null;
  threshold?: number;
  maxActions?: number;
  model?: string;
  noExpand?: boolean;
  raw?: boolean;
  mode?: "vector" | "raw" | "adversarial";
  rounds?: number;
  judgeModel?: string;
  proModel?: string;
  conModel?: string;
  draftModel?: string;
  useDSPy?: boolean;
};

const REPO_ROOT = "/Users/zeitgeist/Documents/SPT/vector_search/Alpha/01_source_fetcher";

export async function POST(req: NextRequest) {
  try {
    const body = (await req.json()) as GenerateBody;
    const query = (body.query || "").trim();
    if (!query) {
      return new Response(JSON.stringify({ error: "Missing 'query'" }), { status: 400 });
    }

    const services = Array.isArray(body.services) ? body.services : null;
    const threshold = typeof body.threshold === "number" ? String(body.threshold) : undefined;
    const maxActions = typeof body.maxActions === "number" ? String(body.maxActions) : undefined;
    const model = typeof body.model === "string" ? body.model : undefined;
    const useExpand = body.noExpand ? "0" : "1";
    const useRaw = body.raw ? "1" : "0";
    const mode = body.mode || (body.raw ? "raw" : "vector");
    const rounds = typeof body.rounds === "number" ? String(Math.max(1, Math.min(3, body.rounds))) : undefined;
    const judgeModel = typeof body.judgeModel === "string" ? body.judgeModel : "";
    const proModel = typeof body.proModel === "string" ? body.proModel : "";
    const conModel = typeof body.conModel === "string" ? body.conModel : "";
    const draftModel = typeof body.draftModel === "string" ? body.draftModel : "";
    const useDSPy = body.useDSPy ? "1" : "0";

    const pyCode = `
import json, os, sys
from pathlib import Path

# Ensure imports work (add repo root and pinecone dir)
REPO_ROOT = Path("${REPO_ROOT}")
sys.path.insert(0, str(REPO_ROOT / "pinecone"))
sys.path.insert(0, str(REPO_ROOT))

from src.policy_generator import IAMPolicyGeneratorV2
from src.policy_generator_adv import generate_adversarial

query = os.getenv("QUERY") or ""
services_env = os.getenv("SERVICES")
services = None
if services_env:
    try:
        services = json.loads(services_env)
    except Exception:
        services = None

kwargs = {}
thr = os.getenv("THRESHOLD")
if thr:
    try:
        kwargs["score_threshold"] = float(thr)
    except Exception:
        pass
ma = os.getenv("MAX_ACTIONS")
if ma:
    try:
        kwargs["max_actions"] = int(ma)
    except Exception:
        pass
mdl = os.getenv("MODEL")
if mdl:
    kwargs["model"] = mdl

use_query_expansion = (os.getenv("USE_EXPAND", "1") == "1")
use_vector_search = (os.getenv("USE_RAW", "0") != "1")
mode = os.getenv("MODE") or ("raw" if not use_vector_search else "vector")

if mode == "adversarial":
    rounds = int(os.getenv("ROUNDS") or "1")
    models = {
        "draft": os.getenv("DRAFT_MODEL") or (os.getenv("MODEL") or "models/gemini-2.5-pro"),
        "judge": os.getenv("JUDGE_MODEL") or "models/gemini-2.5-flash",
        "pro": os.getenv("PRO_MODEL") or (os.getenv("JUDGE_MODEL") or "models/gemini-2.5-flash"),
        "con": os.getenv("CON_MODEL") or (os.getenv("JUDGE_MODEL") or "models/gemini-2.5-flash"),
    }
    res = generate_adversarial(query, rounds=rounds, models=models, settings={"use_dspy": (os.getenv("USE_DSPY", "0") == "1")})
    print(json.dumps(res))
else:
    gen = IAMPolicyGeneratorV2(use_query_expansion=use_query_expansion, target_services=services, use_vector_search=use_vector_search, **kwargs)
    res = gen.generate_policy(query, save_to_file=False)
    print(json.dumps(res))
`;

    const env = {
      ...process.env,
      QUERY: query,
      SERVICES: services ? JSON.stringify(services) : "",
      THRESHOLD: threshold || "",
      MAX_ACTIONS: maxActions || "",
      MODEL: model || "",
      USE_EXPAND: useExpand,
      USE_RAW: useRaw,
      MODE: mode,
      ROUNDS: rounds || "",
      JUDGE_MODEL: judgeModel,
      PRO_MODEL: proModel,
      CON_MODEL: conModel,
      DRAFT_MODEL: draftModel,
      USE_DSPY: useDSPy,
      // Ensure Python can see repo root for any relative file IO
      PWD: REPO_ROOT,
    } as NodeJS.ProcessEnv;

    const result = await new Promise<{ status: number; stdout: string; stderr: string }>((resolve) => {
      const child = spawn("python3", ["-c", pyCode], {
        cwd: REPO_ROOT,
        env,
      });
      let stdout = "";
      let stderr = "";
      child.stdout.on("data", (d) => (stdout += d.toString()));
      child.stderr.on("data", (d) => (stderr += d.toString()));
      child.on("close", (code) => resolve({ status: code ?? 1, stdout, stderr }));
      child.on("error", (err) => resolve({ status: 1, stdout: "", stderr: String(err) }));
    });

    // Try to parse direct JSON, otherwise return error with logs
    try {
      const json = JSON.parse(result.stdout);
      return new Response(JSON.stringify(json), { status: 200, headers: { "content-type": "application/json" } });
    } catch {
      return new Response(
        JSON.stringify({ error: "Generator failed", details: { status: result.status, stdout: result.stdout, stderr: result.stderr } }),
        { status: 500, headers: { "content-type": "application/json" } }
      );
    }
  } catch (e: any) {
    return new Response(JSON.stringify({ error: e?.message || String(e) }), { status: 500 });
  }
}
