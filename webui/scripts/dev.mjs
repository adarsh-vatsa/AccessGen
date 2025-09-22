#!/usr/bin/env node
import { createServer } from "http";
import { spawn } from "child_process";

const BASE = parseInt(process.env.PORT || "3010", 10);
const MAX = BASE + 100; // search up to 100 ports

function checkPort(port) {
  return new Promise((resolve) => {
    const srv = createServer(() => {});
    srv.once("error", () => resolve(false));
    srv.once("listening", () => srv.close(() => resolve(true)));
    srv.listen(port, "127.0.0.1");
  });
}

async function findOpenPort() {
  for (let p = BASE; p <= MAX; p++) {
    // eslint-disable-next-line no-await-in-loop
    const ok = await checkPort(p);
    if (ok) return p;
  }
  throw new Error(`No available port in range ${BASE}-${MAX}`);
}

async function main() {
  const port = await findOpenPort();
  console.log(`[webui] starting Next.js on port ${port}`);
  const child = spawn("npm", ["run", "_dev", "--", "-p", String(port)], { stdio: "inherit", env: process.env });
  child.on("exit", (code) => process.exit(code ?? 0));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});


