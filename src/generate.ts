/**
 * Generator entrypoint. Read a toolkit catalog, infer its dependencies, write a graph.
 *
 * Invokes the Python LangGraph generator (generate.py) which handles:
 *  - Dynamic schema parsing & parameter extraction
 *  - Runtime LLM inference using OpenAI GPT-4o
 *  - Dependency edge inference (producer -> consumer with supplied field label)
 *  - Graph validation & visualization generation
 */
import { existsSync } from "fs";
import { execFileSync } from "child_process";
import { resolve } from "path";

const CATALOG_PATH = process.argv.length > 2 ? process.argv[process.argv.length - 1] : "github_catalog.json";

if (!existsSync(CATALOG_PATH)) {
  console.error(`Catalog file not found: ${CATALOG_PATH}`);
  process.exit(1);
}

const pythonScript = resolve("generate.py");
try {
  execFileSync("python", [pythonScript, CATALOG_PATH], {
    stdio: "inherit",
    env: process.env,
  });
} catch (error: any) {
  console.error("Generator failed with error:", error.message);
  process.exit(1);
}
