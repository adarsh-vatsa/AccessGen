import { NextRequest } from "next/server";
import { spawn } from "child_process";

const REPO_ROOT = "/Users/zeitgeist/Documents/SPT/vector_search/Alpha/01_source_fetcher";

export async function GET(_req: NextRequest) {
  const py = `
import json, os
out = {
  "cwd": os.getcwd(),
  "has_gemini": bool(os.getenv("GEMINI_API_KEY")),
  "has_pinecone": bool(os.getenv("PINECONE_API_KEY")),
}
print(json.dumps(out))
`;
  const res = await new Promise<{ code: number; stdout: string; stderr: string }>((resolve) => {
    const child = spawn("python3", ["-c", py], { cwd: REPO_ROOT, env: process.env });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (d) => (stdout += d.toString()));
    child.stderr.on("data", (d) => (stderr += d.toString()));
    child.on("close", (code) => resolve({ code: code ?? 1, stdout, stderr }));
    child.on("error", (err) => resolve({ code: 1, stdout: "", stderr: String(err) }));
  });
  if (res.code !== 0) {
    return new Response(JSON.stringify({ error: "failed", details: res }), { status: 500 });
  }
  return new Response(res.stdout, { status: 200, headers: { "content-type": "application/json" } });
}

