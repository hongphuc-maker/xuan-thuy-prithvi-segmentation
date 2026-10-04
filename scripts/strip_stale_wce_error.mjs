import fs from "node:fs";

const notebookPath = process.argv[2];
if (!notebookPath) {
  throw new Error("Pass the completed WCE notebook path");
}

const notebook = JSON.parse(fs.readFileSync(notebookPath, "utf8"));
const cell = notebook.cells.find((candidate) => candidate.id === "55328ada");
if (!cell) {
  throw new Error("Expected WCE smoke-test cell 55328ada was not found");
}

const staleOutput = cell.outputs?.some(
  (output) =>
    output.output_type === "error" &&
    output.ename === "NameError" &&
    output.evalue === "name 'experiment' is not defined",
);

if (!staleOutput) {
  console.log("No stale NameError output found; notebook left unchanged");
  process.exit(0);
}

cell.outputs = [];
cell.execution_count = null;
if (cell.metadata?.colab) {
  delete cell.metadata.colab.outputId;
}

fs.writeFileSync(notebookPath, `${JSON.stringify(notebook, null, 1)}\n`, "utf8");
console.log("Removed the stale pre-run NameError output; later completed outputs are unchanged");
