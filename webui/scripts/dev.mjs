#!/usr/bin/env node
import { createServer } from "http";
import { spawn } from "child_process";

const BASE = parseInt(process.env.PORT || "3010", 10);
const MAX = BASE + 100; // search up to 100 ports

const HOSTS_TO_TEST = ["::", "0.0.0.0"];

function checkPortOnHost(port, host) {
  return new Promise((resolve) => {
    const srv = createServer(() => {});
    srv.once("error", (err) => {
      if (err?.code === "EADDRNOTAVAIL" || err?.code === "EAFNOSUPPORT") {
        resolve(true);
      } else {
        resolve(false);
      }
    });
    srv.listen({ port, host, exclusive: true }, () => {
      srv.close(() => resolve(true));
    });
  });
}

async function checkPort(port) {
  for (const host of HOSTS_TO_TEST) {
    // eslint-disable-next-line no-await-in-loop
    const ok = await checkPortOnHost(port, host);
    if (!ok) return false;
  }
  return true;
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
  // Ensure Next.js uses the discovered port regardless of an existing PORT in the environment
  const child = spawn(
    "npm",
    ["run", "_dev", "--", "-p", String(port)],
    {
      stdio: "inherit",
      env: { ...process.env, PORT: String(port) },
    }
  );
  child.on("exit", (code) => process.exit(code ?? 0));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
