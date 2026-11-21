const metrics = {
  decode: { unit: "Tokens / second", direction: "Higher is better", max: 10, digits: 2 },
  ttft: { unit: "Milliseconds", direction: "Lower is better", max: 700, digits: 1 },
  memory: { unit: "GB of sampled process footprint", direction: "Lower is better", max: 2, digits: 3 },
  ppl: { unit: "Corpus perplexity", direction: "Lower is better", max: 180, digits: 2 },
};

const chart = document.querySelector("[data-chart]");
const metric = chart.querySelector("select");
const rows = chart.querySelectorAll(".chart-row");

function showMetric() {
  const selected = metrics[metric.value];
  chart.querySelector("[data-unit]").textContent = selected.unit;
  chart.querySelector("[data-direction]").textContent = selected.direction;
  for (const row of rows) {
    const value = Number(row.dataset[metric.value]);
    row.querySelector(".bar").style.setProperty("--share", `${value / selected.max * 100}%`);
    row.querySelector("[data-value]").textContent = value.toFixed(selected.digits);
  }
}

metric.closest(".metric-select").hidden = false;
metric.addEventListener("change", showMetric);
showMetric();

const terminal = document.querySelector("[data-terminal]");
const snippets = [...terminal.querySelectorAll("[data-snippet]")];
const copy = terminal.querySelector("[data-copy]");
const copyLabel = copy.lastElementChild;
const copyStatus = terminal.querySelector("[data-copy-status]");
let active = terminal.querySelector("pre");
let resetCopy;

for (const button of snippets) {
  button.addEventListener("click", () => {
    active.hidden = true;
    active = document.getElementById(button.getAttribute("aria-controls"));
    active.hidden = false;
    for (const entry of snippets) {
      entry.setAttribute("aria-pressed", entry === button ? "true" : "false");
    }
    clearTimeout(resetCopy);
    copyLabel.textContent = "Copy";
    copyStatus.textContent = "";
  });
}

terminal.querySelector(".terminal-bar").hidden = false;
copy.addEventListener("click", async () => {
  const selected = active;
  try {
    await navigator.clipboard.writeText(selected.textContent);
    if (selected !== active) return;
    copyLabel.textContent = "Copied";
    copyStatus.textContent = "Code copied to clipboard.";
    clearTimeout(resetCopy);
    resetCopy = setTimeout(() => { copyLabel.textContent = "Copy"; }, 1800);
  } catch {
    if (selected !== active) return;
    const range = document.createRange();
    range.selectNodeContents(selected);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    selected.focus();
    copyLabel.textContent = "Selected";
    copyStatus.textContent = "Code selected. Use the browser copy command.";
  }
});
