"""Build the label-audit review page from the audit manifest.

    python scripts/build_label_audit_page.py --audit outputs/label_audit

Writes `index.html` next to the manifest, with the receipts injected as JSON. Nothing is hand-copied:
re-run `prepare_label_audit.py` and this, and the page follows the runs.

The page is published as an Artifact with the images alongside it. Verdicts are stored per receipt, so
they can be read back and turned into docs/annotation_guidelines.md.
"""

# ruff: noqa: E501 - the page template below holds a stylesheet link and CSS rules that are one line
# by nature; wrapping them to satisfy the line limit would corrupt the markup it emits.
from __future__ import annotations

import argparse
import json
from pathlib import Path

PAGE = """<title>Receipt Label Audit</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Public+Sans:wght@400;500;600&display=swap">
<style>
  :root {
    --paper: #f6f8fa;
    --panel: #ffffff;
    --panel-sunk: #eef1f5;
    --ink: #161a1f;
    --ink-soft: #545c68;
    --ink-faint: #8a93a1;
    --rule: #d9dee6;
    --rule-strong: #c2cad6;
    --accent: #2f4a7d;
    --accent-soft: #e5ebf5;
    --agree: #3f7a53;
    --convention: #a9701c;
    --convention-soft: #fbf1de;
    --real: #a33a2b;
    --real-soft: #fbe9e6;
    --shadow: 0 1px 2px rgba(22, 26, 31, .06), 0 8px 24px rgba(22, 26, 31, .05);
    --step--1: .78rem;
    --step-0: .94rem;
    --step-1: 1.13rem;
    --step-2: 1.5rem;
    --step-3: 2.1rem;
  }
  :root:not([data-theme="light"]) {
    color-scheme: light dark;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --paper: #12151a;
      --panel: #1a1f26;
      --panel-sunk: #232932;
      --ink: #e8ecf2;
      --ink-soft: #a3adbb;
      --ink-faint: #6f7987;
      --rule: #2b323c;
      --rule-strong: #3a4250;
      --accent: #8fb0e8;
      --accent-soft: #1e2a3f;
      --agree: #74b98c;
      --convention: #d8a24e;
      --convention-soft: #2e2617;
      --real: #e08472;
      --real-soft: #33201c;
      --shadow: 0 1px 2px rgba(0, 0, 0, .4), 0 8px 24px rgba(0, 0, 0, .3);
    }
  }
  :root[data-theme="dark"] {
    --paper: #12151a;
    --panel: #1a1f26;
    --panel-sunk: #232932;
    --ink: #e8ecf2;
    --ink-soft: #a3adbb;
    --ink-faint: #6f7987;
    --rule: #2b323c;
    --rule-strong: #3a4250;
    --accent: #8fb0e8;
    --accent-soft: #1e2a3f;
    --agree: #74b98c;
    --convention: #d8a24e;
    --convention-soft: #2e2617;
    --real: #e08472;
    --real-soft: #33201c;
    --shadow: 0 1px 2px rgba(0, 0, 0, .4), 0 8px 24px rgba(0, 0, 0, .3);
  }

  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding-block: 0 3rem;
    background: var(--paper);
    color: var(--ink);
    font: 400 var(--step-0)/1.55 "Public Sans", ui-sans-serif, system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 1180px; margin-inline: auto; padding-inline: 20px; }

  header { border-bottom: 1px solid var(--rule); background: var(--panel); }
  .masthead { display: flex; flex-wrap: wrap; gap: 1.25rem 2rem; align-items: flex-end;
              justify-content: space-between; padding-block: 1.5rem 1.25rem; }
  h1 { margin: 0; font: 700 var(--step-3)/1.05 "Archivo", ui-sans-serif, system-ui, sans-serif;
       letter-spacing: -.02em; text-wrap: balance; }
  .lede { margin: .45rem 0 0; max-width: 62ch; color: var(--ink-soft); }
  .tallies { display: flex; gap: 1.75rem; font-variant-numeric: tabular-nums; }
  .tally b { display: block; font: 600 var(--step-2)/1 "Archivo", sans-serif; letter-spacing: -.01em; }
  .tally span { font-size: var(--step--1); color: var(--ink-faint); text-transform: uppercase;
                letter-spacing: .08em; }
  .tally.is-convention b { color: var(--convention); }
  .tally.is-real b { color: var(--real); }

  .rail { display: flex; flex-wrap: wrap; gap: 4px; padding-block: .85rem 1rem; }
  .pip { width: 22px; height: 22px; border: 1px solid var(--rule-strong); border-radius: 3px;
         background: var(--panel-sunk); cursor: pointer; padding: 0;
         font: 500 10px/1 "IBM Plex Mono", monospace; color: var(--ink-faint); }
  .pip[data-state="clean"] { background: transparent; border-style: dashed; }
  .pip[data-state="done"] { background: var(--accent); border-color: var(--accent); color: var(--panel); }
  .pip[aria-current="true"] { outline: 2px solid var(--ink); outline-offset: 2px; }
  .pip:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

  .banner { margin-block: 1rem 0; padding: .7rem .9rem; border-radius: 4px;
            background: var(--convention-soft); color: var(--convention);
            border: 1px solid currentColor; font-size: var(--step--1); }

  main { display: grid; grid-template-columns: minmax(0, 420px) minmax(0, 1fr); gap: 2rem;
         align-items: start; padding-block: 1.75rem; }
  @media (max-width: 860px) { main { grid-template-columns: 1fr; } }

  .plate { position: sticky; top: 1rem; }
  @media (max-width: 860px) { .plate { position: static; } }
  .plate img { width: 100%; max-width: 100%; display: block; border: 1px solid var(--rule);
               border-radius: 3px; background: var(--panel); }
  .plate figcaption { display: flex; flex-wrap: wrap; gap: .5rem .75rem; align-items: baseline;
                      margin-top: .6rem; }
  .rid { font: 600 var(--step-0) "IBM Plex Mono", monospace; letter-spacing: -.01em; }
  .badge { font-size: var(--step--1); text-transform: uppercase; letter-spacing: .08em;
           color: var(--ink-faint); }
  .score { margin-left: auto; font-variant-numeric: tabular-nums; color: var(--ink-soft);
           font-size: var(--step--1); }

  .question { margin: 0 0 1.1rem; font: 600 var(--step-1)/1.35 "Archivo", sans-serif;
              text-wrap: balance; }
  .rows { display: flex; flex-direction: column; gap: .9rem; }
  .row { border: 1px solid var(--rule); border-left: 3px solid var(--real); border-radius: 4px;
         background: var(--panel); padding: .85rem .95rem; box-shadow: var(--shadow); }
  .row.is-convention { border-left-color: var(--convention); }
  .row-head { display: flex; flex-wrap: wrap; gap: .5rem .75rem; align-items: baseline; }
  .path { font: 500 var(--step-0) "IBM Plex Mono", monospace; }
  .tag { font-size: var(--step--1); padding: .1rem .45rem; border-radius: 3px;
         text-transform: uppercase; letter-spacing: .06em; }
  .tag.kind { background: var(--panel-sunk); color: var(--ink-faint); }
  .tag.conv { background: var(--convention-soft); color: var(--convention); }
  .values { display: grid; grid-template-columns: 5.5rem minmax(0, 1fr); gap: .35rem .8rem;
            margin-top: .7rem; }
  .who { font-size: var(--step--1); text-transform: uppercase; letter-spacing: .08em;
         color: var(--ink-faint); padding-top: .15rem; }
  .val { font: 400 var(--step-0)/1.45 "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums;
         overflow-wrap: anywhere; }
  .val.empty { color: var(--ink-faint); font-style: italic; }
  .verdicts { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .8rem; }
  .verdict { font: 500 var(--step--1) "Public Sans", sans-serif; padding: .32rem .7rem;
             border: 1px solid var(--rule-strong); border-radius: 999px; background: transparent;
             color: var(--ink-soft); cursor: pointer; }
  .verdict:hover { border-color: var(--ink-faint); color: var(--ink); }
  .verdict:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .verdict[aria-pressed="true"] { background: var(--ink); border-color: var(--ink); color: var(--paper); }
  .verdict[aria-pressed="true"][data-v="gold"] { background: var(--accent); border-color: var(--accent);
                                                 color: #fff; }
  .verdict[aria-pressed="true"][data-v="model"] { background: var(--real); border-color: var(--real);
                                                  color: #fff; }

  details.matched { margin-top: 1rem; border-top: 1px solid var(--rule); padding-top: .8rem; }
  details.matched summary { cursor: pointer; color: var(--ink-soft); font-size: var(--step--1); }
  details.matched ul { list-style: none; margin: .7rem 0 0; padding: 0; display: grid;
                       gap: .3rem .9rem; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); }
  details.matched li { font: 400 var(--step--1) "IBM Plex Mono", monospace; color: var(--ink-soft);
                       overflow-wrap: anywhere; }
  details.matched li b { color: var(--agree); font-weight: 500; }

  .note { margin-top: 1.1rem; }
  .note label { display: block; font-size: var(--step--1); text-transform: uppercase;
                letter-spacing: .08em; color: var(--ink-faint); margin-bottom: .35rem; }
  .note textarea { width: 100%; min-height: 4.2rem; resize: vertical; padding: .6rem .7rem;
                   border: 1px solid var(--rule-strong); border-radius: 4px; background: var(--panel);
                   color: var(--ink); font: 400 var(--step-0)/1.5 "Public Sans", sans-serif; }
  .note textarea:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }

  .pager { display: flex; gap: .6rem; align-items: center; margin-top: 1.5rem; }
  .pager button { font: 500 var(--step-0) "Public Sans", sans-serif; padding: .5rem 1rem;
                  border: 1px solid var(--rule-strong); border-radius: 4px; background: var(--panel);
                  color: var(--ink); cursor: pointer; }
  .pager button:hover:not(:disabled) { border-color: var(--ink-faint); }
  .pager button:disabled { opacity: .45; cursor: default; }
  .pager button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .pager .hint { margin-left: auto; font-size: var(--step--1); color: var(--ink-faint); }
  kbd { font: 500 var(--step--1) "IBM Plex Mono", monospace; border: 1px solid var(--rule-strong);
        border-bottom-width: 2px; border-radius: 3px; padding: 0 .3rem; }

  .clean-note { border: 1px dashed var(--rule-strong); border-radius: 4px; padding: 1.1rem;
                color: var(--ink-soft); background: transparent; }
  .clean-note b { color: var(--agree); }

  @media (prefers-reduced-motion: no-preference) {
    .row { transition: border-color .15s ease; }
  }
</style>

<header>
  <div class="wrap">
    <div class="masthead">
      <div>
        <h1>Receipt Label Audit</h1>
        <p class="lede">Thirty dev receipts where the fine-tuned model and CORD's labels disagree.
          For each difference: is the label right, or is the model? The answers become the annotation
          guidelines and the list of gold errors.</p>
      </div>
      <div class="tallies">
        <div class="tally"><b id="t-receipts">0</b><span>receipts</span></div>
        <div class="tally is-real"><b id="t-real">0</b><span>real differences</span></div>
        <div class="tally is-convention"><b id="t-conv">0</b><span>separator only</span></div>
      </div>
    </div>
    <div class="rail" id="rail" role="tablist" aria-label="Receipts"></div>
    <div class="banner" id="banner" hidden></div>
  </div>
</header>

<div class="wrap">
  <main>
    <figure class="plate" style="margin:0">
      <img id="shot" alt="" decoding="async">
      <figcaption>
        <span class="rid" id="rid"></span>
        <span class="badge" id="group"></span>
        <span class="score" id="score"></span>
      </figcaption>
    </figure>
    <section aria-live="polite">
      <h2 class="question" id="question"></h2>
      <div class="rows" id="rows"></div>
      <details class="matched" id="matched">
        <summary></summary>
        <ul id="matched-list"></ul>
      </details>
      <div class="note">
        <label for="note">Note for the guidelines</label>
        <textarea id="note" placeholder="A convention worth writing down, or why the label is wrong."></textarea>
      </div>
      <div class="pager">
        <button id="prev" type="button">Previous</button>
        <button id="next" type="button">Next receipt</button>
        <span class="hint"><kbd>&larr;</kbd> <kbd>&rarr;</kbd> to move &middot;
          <kbd>1</kbd> label <kbd>2</kbd> model <kbd>3</kbd> unclear</span>
      </div>
    </section>
  </main>
</div>

<script type="application/json" id="data">__DATA__</script>
<script>
(function () {
  const DATA = JSON.parse(document.getElementById("data").textContent);
  const receipts = DATA.receipts;
  const state = new Map();           // example_id -> {verdicts:{path:v}, note:""}
  let index = 0;
  let db = null;

  const el = (id) => document.getElementById(id);
  const disagreeing = (r) => r.rows.filter((row) => !row.agrees);
  const record = (id) => {
    if (!state.has(id)) state.set(id, { verdicts: {}, note: "" });
    return state.get(id);
  };
  const isDone = (r) => {
    const rec = record(r.example_id);
    const need = disagreeing(r);
    return need.length > 0 && need.every((row) => rec.verdicts[row.path]);
  };

  el("t-receipts").textContent = receipts.length;
  el("t-real").textContent = receipts.reduce((n, r) => n + r.n_disagreements - r.n_convention_only, 0);
  el("t-conv").textContent = receipts.reduce((n, r) => n + r.n_convention_only, 0);

  const rail = el("rail");
  receipts.forEach((r, i) => {
    const pip = document.createElement("button");
    pip.className = "pip";
    pip.type = "button";
    pip.textContent = String(i + 1);
    pip.title = r.example_id + " — F1 " + r.f1.toFixed(2);
    pip.addEventListener("click", () => { index = i; render(); });
    rail.appendChild(pip);
  });

  function paintRail() {
    [...rail.children].forEach((pip, i) => {
      const r = receipts[i];
      pip.dataset.state = disagreeing(r).length === 0 ? "clean" : (isDone(r) ? "done" : "open");
      pip.setAttribute("aria-current", i === index ? "true" : "false");
    });
  }

  function valueCell(values) {
    const span = document.createElement("span");
    if (!values.length) { span.className = "val empty"; span.textContent = "not present"; }
    else { span.className = "val"; span.textContent = values.join("   ·   "); }
    return span;
  }

  function renderRow(r, row) {
    const card = document.createElement("article");
    card.className = "row" + (row.agrees_lenient ? " is-convention" : "");

    const head = document.createElement("div");
    head.className = "row-head";
    const path = document.createElement("span");
    path.className = "path";
    path.textContent = row.path;
    const kind = document.createElement("span");
    kind.className = "tag kind";
    kind.textContent = row.kind;
    head.append(path, kind);
    if (row.agrees_lenient) {
      const conv = document.createElement("span");
      conv.className = "tag conv";
      conv.textContent = "separator only";
      head.appendChild(conv);
    }
    card.appendChild(head);

    const values = document.createElement("div");
    values.className = "values";
    const goldWho = document.createElement("span");
    goldWho.className = "who";
    goldWho.textContent = "label";
    const modelWho = document.createElement("span");
    modelWho.className = "who";
    modelWho.textContent = "model";
    values.append(goldWho, valueCell(row.gold), modelWho, valueCell(row.pred));
    card.appendChild(values);

    const verdicts = document.createElement("div");
    verdicts.className = "verdicts";
    [["gold", "Label is right"], ["model", "Model is right"], ["unclear", "Unclear"]].forEach(([v, text]) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "verdict";
      button.dataset.v = v;
      button.textContent = text;
      const rec = record(r.example_id);
      button.setAttribute("aria-pressed", rec.verdicts[row.path] === v ? "true" : "false");
      button.addEventListener("click", () => {
        const current = record(r.example_id);
        current.verdicts[row.path] = current.verdicts[row.path] === v ? undefined : v;
        if (!current.verdicts[row.path]) delete current.verdicts[row.path];
        save(r.example_id);
        render();
      });
      verdicts.appendChild(button);
    });
    card.appendChild(verdicts);
    return card;
  }

  function render() {
    const r = receipts[index];
    const rec = record(r.example_id);

    el("shot").src = r.image;
    el("shot").alt = "Receipt " + r.example_id;
    el("rid").textContent = r.example_id;
    el("group").textContent = r.group === "worst" ? "lowest scoring" : "sampled";
    el("score").textContent = "field F1 " + r.f1.toFixed(2) + " · " + (index + 1) + " of " + receipts.length;

    const diffs = disagreeing(r);
    const conv = diffs.filter((row) => row.agrees_lenient).length;
    el("question").textContent = diffs.length === 0
      ? "The label and the model agree on every field."
      : diffs.length + (diffs.length === 1 ? " field differs" : " fields differ")
        + (conv ? " — " + conv + " only in separators" : "");

    const rows = el("rows");
    rows.textContent = "";
    if (diffs.length === 0) {
      const clean = document.createElement("div");
      clean.className = "clean-note";
      clean.innerHTML = "<b>Nothing to judge here.</b> Included as a check that ordinary receipts are "
        + "labelled the way the guidelines describe.";
      rows.appendChild(clean);
    } else {
      diffs.forEach((row) => rows.appendChild(renderRow(r, row)));
    }

    const matched = r.rows.filter((row) => row.agrees);
    el("matched").hidden = matched.length === 0;
    el("matched").querySelector("summary").textContent =
      matched.length + " field" + (matched.length === 1 ? "" : "s") + " already agree";
    const list = el("matched-list");
    list.textContent = "";
    matched.forEach((row) => {
      const li = document.createElement("li");
      li.innerHTML = "<b>" + row.path + "</b> ";
      li.append(document.createTextNode(row.gold.join("  ·  ")));
      list.appendChild(li);
    });

    el("note").value = rec.note || "";
    el("prev").disabled = index === 0;
    el("next").disabled = index === receipts.length - 1;
    paintRail();
  }

  el("prev").addEventListener("click", () => { if (index > 0) { index--; render(); } });
  el("next").addEventListener("click", () => {
    if (index < receipts.length - 1) { index++; render(); }
  });
  el("note").addEventListener("change", () => {
    record(receipts[index].example_id).note = el("note").value;
    save(receipts[index].example_id);
  });

  document.addEventListener("keydown", (event) => {
    if (event.target.tagName === "TEXTAREA") return;
    if (event.key === "ArrowLeft" && index > 0) { index--; render(); }
    if (event.key === "ArrowRight" && index < receipts.length - 1) { index++; render(); }
    const pick = { "1": "gold", "2": "model", "3": "unclear" }[event.key];
    if (pick) {
      const r = receipts[index];
      const open = disagreeing(r).find((row) => !record(r.example_id).verdicts[row.path]);
      if (open) {
        record(r.example_id).verdicts[open.path] = pick;
        save(r.example_id);
        render();
      }
    }
  });

  function save(id) {
    if (!db) return;
    const rec = record(id);
    db.doc("audit/" + id).set({
      example_id: id,
      verdicts: rec.verdicts,
      note: rec.note || "",
      updatedAt: Date.now(),
    }).catch(() => {});
  }

  render();

  (async () => {
    db = (window.claude && window.claude.use) ? await window.claude.use("db") : null;
    if (!db) {
      const banner = el("banner");
      banner.hidden = false;
      banner.textContent = "Not connected to storage — judgements on this page will not be saved.";
      return;
    }
    db.collection("audit").onSnapshot((docs) => {
      let touched = false;
      docs.forEach((doc) => {
        const data = doc.data ? doc.data() : doc;
        if (!data || !data.example_id) return;
        state.set(data.example_id, { verdicts: data.verdicts || {}, note: data.note || "" });
        touched = true;
      });
      if (touched) render();
    });
  })();
})();
</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default="outputs/label_audit")
    args = parser.parse_args()

    audit = Path(args.audit)
    manifest = json.loads((audit / "manifest.json").read_text())
    slim = {
        "receipts": [
            {
                key: receipt[key]
                for key in (
                    "example_id",
                    "group",
                    "f1",
                    "image",
                    "rows",
                    "n_disagreements",
                    "n_convention_only",
                )
            }
            for receipt in manifest["receipts"]
        ]
    }
    page = audit / "index.html"
    page.write_text(PAGE.replace("__DATA__", json.dumps(slim, separators=(",", ":"))))
    print(f"{page}: {page.stat().st_size / 1024:.0f} KB, {len(slim['receipts'])} receipts")


if __name__ == "__main__":
    main()
