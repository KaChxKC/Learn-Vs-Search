"use strict";
// Thin client: it draws the arena and forwards clicks. All game logic lives on the server,
// which reuses the Python engine and agents. The server also tells us which cells won and
// where the last move landed, so this file only animates - it never judges a position.

const SPEEDS = { slow: 1000, normal: 550, fast: 140 };
const FIGHTERS = {
  Human:   { icon: "You", name: "YOU" },
  Minimax: { icon: "MM", name: "MINIMAX" },
  "Q-learn": { icon: "Q", name: "Q-LEARNER" },
  Random:  { icon: "?", name: "RANDOM" },
};
const MODES = [
  { id: "hm", title: "You vs Minimax", badge: "PLAY", desc: "Try to outsmart the search algorithm.", red: "Human", yellow: "Minimax" },
  { id: "hq", title: "You vs Q-learner", badge: "PLAY", desc: "Take on the AI that taught itself.", red: "Human", yellow: "Q-learn" },
  { id: "qm", title: "Q-learner vs Minimax", badge: "WATCH", desc: "The showdown: learning vs searching.", red: "Q-learn", yellow: "Minimax" },
  { id: "qr", title: "Q-learner vs Random", badge: "WATCH", desc: "See how far self-play gets you.", red: "Q-learn", yellow: "Random" },
];
const DIFF_HINT = {
  2: "Easy: minimax looks 2 moves ahead.",
  4: "Medium: minimax looks 4 moves ahead.",
  6: "Hard: minimax looks 6 moves ahead (slower to think).",
};

const $ = (id) => document.getElementById(id);
const cfg = { board: "5x4", mode: "hm", swap: false, depth: 4, speed: "normal" };
const score = { red: 0, yellow: 0, draw: 0 };

let state = null, gameId = null, tables = {};
let epoch = 0, renderId = 0, aiTimer = null, round = 1;
let paused = false, thinking = false, busy = false;
let counted = null, hoverCol = null, soundedLen = 0;
let soundOn = true, audio = null;

/* ---------- sound (Web Audio, no files) ---------- */

try { soundOn = localStorage.getItem("lvs-sound") !== "off"; } catch (e) { /* ignore */ }

function ensureAudio() {
  try {
    if (!audio) audio = new (window.AudioContext || window.webkitAudioContext)();
    if (audio.state === "suspended") audio.resume();
  } catch (e) { /* no audio available */ }
}

function tone(freq, dur = 0.12, type = "sine", vol = 0.15, slideTo = null, delay = 0) {
  if (!soundOn || !audio) return;
  const t = audio.currentTime + delay;
  const o = audio.createOscillator();
  const g = audio.createGain();
  o.type = type;
  o.frequency.setValueAtTime(freq, t);
  if (slideTo) o.frequency.exponentialRampToValueAtTime(slideTo, t + dur);
  g.gain.setValueAtTime(vol, t);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  o.connect(g).connect(audio.destination);
  o.start(t);
  o.stop(t + dur + 0.03);
}
const sfx = {
  click: () => tone(540, 0.07, "square", 0.05),
  drop: (delay) => tone(210, 0.13, "sine", 0.24, 70, delay),
  go: () => { tone(660, 0.1, "square", 0.07); tone(990, 0.2, "square", 0.07, null, 0.12); },
  win: () => [523, 659, 784, 1047].forEach((f, i) => tone(f, 0.2, "triangle", 0.17, null, i * 0.11)),
  lose: () => [392, 330, 262].forEach((f, i) => tone(f, 0.24, "sawtooth", 0.07, null, i * 0.17)),
  draw: () => [440, 440].forEach((f, i) => tone(f, 0.16, "square", 0.06, null, i * 0.2)),
};

/* ---------- toast (small, quiet result notice) ---------- */

let toastTimer = null;
function toast(text) {
  const t = $("toast");
  t.textContent = text;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 2200);
}

/* ---------- server ---------- */

async function api(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return r.json();
}
function showOffline(on) { $("offline").hidden = !on; }

/* ---------- who is playing ---------- */

function kinds() {
  const m = MODES.find((x) => x.id === cfg.mode);
  return cfg.swap ? { red: m.yellow, yellow: m.red } : { red: m.red, yellow: m.yellow };
}
function kindOf(p) { return p === 1 ? kinds().red : kinds().yellow; }
function humanCount() { const k = kinds(); return (k.red === "Human") + (k.yellow === "Human"); }
function colorName(p) { return p === 1 ? "RED" : "YELLOW"; }
function fighterName(p) {
  const k = kindOf(p);
  return k === "Human" && humanCount() === 2 ? colorName(p) : FIGHTERS[k].name;
}
function describe(k) {
  if (k === "Human") return humanCount() === 2 ? "A human player." : "That's you. Good luck!";
  if (k === "Random") return "Picks any legal move. A baseline.";
  if (k === "Minimax") return cfg.board === "ttt" ? "Searches the whole game. Can't be beaten."
    : `Searches ${cfg.depth} moves ahead with alpha-beta pruning.`;
  return tables[cfg.board] ? "Taught itself by playing millions of games."
    : "No trained table for this board yet - plays randomly.";
}

/* ---------- game flow ---------- */

function startGame() {
  ensureAudio();
  sfx.click();
  score.red = score.yellow = score.draw = 0;
  round = 1;
  $("menu").hidden = true;
  $("game").hidden = false;
  newGame();
}

function toMenu() {
  clearTimeout(aiTimer);
  epoch++;
  $("game").hidden = true;
  $("menu").hidden = false;
  $("toast").classList.remove("show");
}

function rematch() {
  if (state && state.terminal) round++;
  sfx.click();
  newGame();
}

async function newGame() {
  clearTimeout(aiTimer);
  epoch++;
  const ep = epoch;
  thinking = false; busy = false; counted = null; soundedLen = 0; paused = false;
  const k = kinds();
  try {
    const s = await api("/api/new", { board: cfg.board, red: k.red, yellow: k.yellow, depth: cfg.depth });
    if (ep !== epoch) return;
    showOffline(false);
    gameId = s.game_id;
    apply(s);
  } catch (e) {
    showOffline(true);
  }
}

function apply(s) {
  state = s;
  renderBoard();
  renderStatus();
  renderFighters();
  renderHud();
  updateButtons();
  tally();
  if (s.last_cell && s.history.length > soundedLen) {
    const row = s.last_cell[0];
    sfx.drop(s.gravity ? (0.3 + (row + 1) * 0.07) * 0.56 : 0);
  }
  soundedLen = s.history.length;
  scheduleAI();
}

function canHuman() {
  return !!state && !state.terminal && state.current_is_human && !busy;
}

function refreshInteractive() {
  $("board").classList.toggle("human-turn", canHuman());
  if (hoverCol !== null) setHover(hoverCol);
}

async function humanPlay(move) {
  if (!canHuman() || !state.legal.includes(move)) return;
  const ep = epoch;
  busy = true;
  clearTimeout(aiTimer);
  try {
    const s = await api("/api/move", { game_id: gameId, move });
    if (ep !== epoch) return;
    busy = false;
    if (!s.error) apply(s);
  } catch (e) {
    showOffline(true);
  } finally {
    if (ep === epoch) busy = false;
  }
}

function scheduleAI() {
  clearTimeout(aiTimer);
  if (!state || state.terminal || state.current_is_human || paused) return;
  const ep = epoch;
  aiTimer = setTimeout(() => stepAI(ep), SPEEDS[cfg.speed]);
}

async function stepAI(ep) {
  if (ep !== epoch || !state || state.terminal || state.current_is_human) return;
  thinking = true;
  renderStatus();
  renderFighters();
  try {
    const s = await api("/api/step", { game_id: gameId });
    if (ep !== epoch) return;
    thinking = false;
    if (!s.error) apply(s);
  } catch (e) {
    showOffline(true);
  } finally {
    if (ep === epoch) { thinking = false; renderStatus(); renderFighters(); updateButtons(); }
  }
}

/* ---------- fit board to the screen ---------- */

function fitBoard() {
  if (!state) return;
  const wide = innerWidth > 1040;
  const availW = wide ? innerWidth - 2 * 250 - 44 - 90 : innerWidth - 40;
  const availH = innerHeight - (wide ? 250 : 470);
  const gap = 8;
  const byW = (availW - gap * (state.cols + 1)) / state.cols;
  const headRows = state.gravity ? 0.6 : 0;
  const byH = (availH - gap * (state.rows + 1) - 8) / (state.rows + headRows);
  const cell = Math.max(44, Math.min(byW, byH, state.gravity ? 96 : 120));
  $("board-wrap").style.setProperty("--cell", Math.floor(cell) + "px");
}

/* ---------- rendering ---------- */

function renderBoard() {
  const s = state;
  const wrap = $("board-wrap");
  const board = $("board");
  const heads = $("heads");

  wrap.style.setProperty("--cols", s.cols);
  wrap.style.setProperty("--ghost", s.current === 2 ? "var(--yellow)" : "var(--red)");
  wrap.classList.toggle("ttt-mode", !s.gravity);
  fitBoard();

  board.className = "board " + (s.gravity ? "c4" : "ttt") + (canHuman() ? " human-turn" : "");
  board.innerHTML = "";
  heads.innerHTML = "";

  if (s.gravity) {
    for (let c = 0; c < s.cols; c++) {
      const h = document.createElement("div");
      h.className = "head";
      h.innerHTML = '<div class="ghost"></div>';
      h.onclick = () => humanPlay(c);
      h.onmouseenter = () => setHover(c);
      heads.appendChild(h);
    }
  }

  const last = s.last_cell ? s.last_cell.join(",") : null;
  for (let r = 0; r < s.rows; r++) {
    for (let c = 0; c < s.cols; c++) {
      const cell = document.createElement("div");
      cell.className = "cell";
      cell.dataset.c = c;
      const hole = document.createElement("div");
      hole.className = "hole";
      const v = s.grid[r][c];
      if (v) {
        const disc = document.createElement("div");
        disc.className = "disc " + (v === 1 ? "red" : "yellow");
        disc.dataset.rc = r + "," + c;
        if (last === r + "," + c) {
          disc.classList.add("last", s.gravity ? "drop" : "pop");
          disc.style.setProperty("--drop", r + 1);
        }
        hole.appendChild(disc);
      } else if (!s.gravity && s.legal.includes(r * s.cols + c)) {
        cell.classList.add("legal");
      }
      cell.appendChild(hole);
      cell.onclick = () => humanPlay(s.gravity ? c : r * s.cols + c);
      if (s.gravity) cell.onmouseenter = () => setHover(c);
      board.appendChild(cell);
    }
  }
  wrap.onmouseleave = () => setHover(null);
  if (hoverCol !== null) setHover(hoverCol);

  // After the final disc lands: light the winning line, then celebrate.
  const rid = ++renderId;
  if (s.terminal) {
    const settle = s.gravity ? 650 : 300;
    setTimeout(() => {
      if (rid !== renderId) return;
      board.classList.add("over");
      s.win_cells.forEach(([r, c]) => {
        const d = board.querySelector(`.disc[data-rc="${r},${c}"]`);
        if (d) d.classList.add("win");
      });
      celebrate(s);
    }, settle);
  }
}

function celebrate(s) {
  const w = s.winner;
  const solo = humanCount() === 1;
  if (w === null) { toast("Draw"); sfx.draw(); return; }
  const mine = solo && kindOf(w) === "Human";
  toast(solo ? (mine ? "You win" : "You lose") : colorName(w)[0] + colorName(w).slice(1).toLowerCase() + " wins");
  if (!solo || mine) sfx.win(); else sfx.lose();
}

function setHover(c) {
  hoverCol = c;
  document.querySelectorAll(".head.on, .cell.colhl").forEach((e) => e.classList.remove("on", "colhl"));
  if (c === null || !state || !state.gravity || !canHuman() || !state.legal.includes(c)) return;
  const h = $("heads").children[c];
  if (h) h.classList.add("on");
  document.querySelectorAll(`.cell[data-c="${c}"]`).forEach((e) => e.classList.add("colhl"));
}

function renderStatus() {
  const s = state;
  if (!s) return;
  const el = $("status");
  if (s.terminal) {
    const w = s.winner;
    el.className = "status " + (w === 1 ? "red" : w === 2 ? "yellow" : "draw");
    const n = w ? fighterName(w) : "";
    el.textContent = w === null ? "DRAW" : n === "YOU" ? "YOU WIN!" : n + " WINS!";
  } else {
    const p = s.current;
    el.className = "status " + (p === 1 ? "red" : "yellow");
    const n = fighterName(p);
    el.textContent = s.current_is_human
      ? (n === "YOU" ? "YOUR TURN" : n + "'S TURN")
      : thinking ? n + " IS THINKING..." : paused ? n + " - PAUSED" : n + "'S TURN";
  }
  $("note").textContent = s.note || "";
}

function renderFighters() {
  const s = state;
  [[1, "red"], [2, "yellow"]].forEach(([p, name]) => {
    const k = kindOf(p);
    $("av-" + name).textContent = FIGHTERS[k].icon;
    $("nm-" + name).textContent = fighterName(p);
    $("ds-" + name).textContent = describe(k);
    const card = $("f-" + name);
    card.classList.toggle("active", !s.terminal && s.current === p);
    card.classList.toggle("winner", s.terminal && s.winner === p);
    $("th-" + name).classList.toggle("on", thinking && !s.terminal && s.current === p);
  });
  $("sc-red").textContent = score.red;
  $("sc-yellow").textContent = score.yellow;
}

function renderHud() {
  $("round").textContent = round;
  $("moves").textContent = state.history.length;
  $("draws").textContent = score.draw;
}

function tally() {
  if (!state.terminal || counted === gameId) return;
  counted = gameId;
  if (state.winner === 1) score.red++;
  else if (state.winner === 2) score.yellow++;
  else score.draw++;
  $("sc-red").textContent = score.red;
  $("sc-yellow").textContent = score.yellow;
  $("draws").textContent = score.draw;
}

function updateButtons() {
  const done = state && state.terminal;
  $("pause").textContent = paused ? "Resume" : "Pause";
  $("pause").disabled = done;
  $("step").disabled = !(paused && state && !done && !state.current_is_human && !thinking);
  $("restart").textContent = done ? "Rematch" : "Restart";
  $("restart").classList.toggle("glow", !!done);
  document.querySelectorAll("#seg-speed button").forEach((b) => b.classList.toggle("on", b.dataset.speed === cfg.speed));
}

/* ---------- results / graphs ---------- */

let returnTo = "menu", pausedBefore = false;
const BOARD_LABEL = { ttt: "tic-tac-toe", "5x4": "Connect-4 5x4", "5x5": "Connect-4 5x5" };

function describeFigure(name) {
  const base = name.replace(/\.png$/i, "");
  let m;
  if ((m = base.match(/^(.+)_learning_curve$/)))
    return { title: "Learning curve: " + (BOARD_LABEL[m[1]] || m[1]),
             text: "How the Q-learner's results against minimax change as it trains." };
  if ((m = base.match(/^(.+)_net_score$/)))
    return { title: "Strength over training: " + (BOARD_LABEL[m[1]] || m[1]),
             text: "Win % minus loss % against each minimax depth. Zero means an even match." };
  if ((m = base.match(/^(.+)_outcomes$/)))
    return { title: "Win / draw / loss mix: " + (BOARD_LABEL[m[1]] || m[1]),
             text: "How the share of wins, draws and losses shifts as training goes on." };
  if ((m = base.match(/^(.+)_depth_scaling$/)))
    return { title: "Against deeper search: " + (BOARD_LABEL[m[1]] || m[1]),
             text: "The finished learner versus minimax looking further and further ahead." };
  if ((m = base.match(/^(.+)_coverage$/)))
    return { title: "Coverage: " + (BOARD_LABEL[m[1]] || m[1]),
             text: "How big the table gets, and whether seeing more positions makes it stronger." };
  if ((m = base.match(/^(.+)_seat_curves$/)))
    return { title: "First-move advantage: " + (BOARD_LABEL[m[1]] || m[1]),
             text: "Win rate over training when the learner moves first (solid) vs second (dashed)." };
  if (base === "board_comparison")
    return { title: "Board comparison",
             text: "The trained learner's final result on each board against minimax." };
  if ((m = base.match(/^(.+)_per_seat$/)))
    return { title: "First vs second mover: " + (BOARD_LABEL[m[1]] || m[1]),
             text: "Win rate when the learner moves first compared with second." };
  if ((m = base.match(/^alpha_beta_(.+)$/)))
    return { title: "Alpha-beta pruning: " + (BOARD_LABEL[m[1]] || m[1]),
             text: "Positions searched by plain minimax vs alpha-beta at each depth (log scale)." };
  if ((m = base.match(/^sweep_(.+?)_(alpha|epsilon|draw_reward)$/)))
    return { title: "Hyperparameter sweep: " + m[2].replace("_", " ") + " (" + (BOARD_LABEL[m[1]] || m[1]) + ")",
             text: "How this setting changes how quickly the learner converges." };
  return { title: base.replace(/_/g, " "), text: "" };
}

async function openResults() {
  const fromGame = !$("game").hidden;
  returnTo = fromGame ? "game" : "menu";
  if (fromGame) { pausedBefore = paused; paused = true; clearTimeout(aiTimer); }
  $("menu").hidden = true;
  $("game").hidden = true;
  $("results").hidden = false;
  window.scrollTo(0, 0);

  const box = $("figs");
  box.innerHTML = "";
  let figures = [];
  try { figures = (await (await fetch("/api/figures")).json()).figures || []; }
  catch (e) { showOffline(true); }

  if (!figures.length) {
    box.innerHTML = '<div class="figs-empty"><b>No graphs yet.</b><br>Generate them after a training run, for example:<br>' +
      '<code>python plot.py --csv logs/5x4.csv</code><br><code>python plot.py --alpha-beta 5x4</code></div>';
    return;
  }
  figures.forEach((f) => {
    const d = describeFigure(f.name);
    const card = document.createElement("div");
    card.className = "fig";
    card.innerHTML = `<img src="${f.url}" alt=""><div class="cap"><b></b><span></span></div>`;
    card.querySelector("b").textContent = d.title;
    card.querySelector("span").textContent = d.text;
    card.onclick = () => { $("lightbox-img").src = f.url; $("lightbox").hidden = false; };
    box.appendChild(card);
  });
}

function closeResults() {
  $("lightbox").hidden = true;
  $("results").hidden = true;
  if (returnTo === "game") {
    $("game").hidden = false;
    paused = pausedBefore;
    updateButtons();
    renderStatus();
    scheduleAI();
  } else {
    $("menu").hidden = false;
  }
}

/* ---------- menu + controls ---------- */

function syncMenu() {
  document.querySelectorAll("#modes .mode").forEach((b) => b.classList.toggle("on", b.dataset.id === cfg.mode));
  document.querySelectorAll("#seg-board button").forEach((b) => b.classList.toggle("on", b.dataset.board === cfg.board));
  document.querySelectorAll("#seg-diff button").forEach((b) => b.classList.toggle("on", parseInt(b.dataset.depth, 10) === cfg.depth));
  $("swap").innerHTML = `Swap who goes first: <b>${cfg.swap ? "ON" : "OFF"}</b>`;
  $("diff-hint").textContent = cfg.board === "ttt"
    ? "On tic-tac-toe minimax always searches to the end, so difficulty doesn't apply."
    : DIFF_HINT[cfg.depth];
  $("seg-diff").style.opacity = cfg.board === "ttt" ? 0.4 : 1;
}

function toggleSound() {
  soundOn = !soundOn;
  try { localStorage.setItem("lvs-sound", soundOn ? "on" : "off"); } catch (e) { /* ignore */ }
  $("sound").textContent = "Sound: " + (soundOn ? "on" : "off");
  ensureAudio();
  sfx.click();
}

function toggleFullscreen() {
  try {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen();
  } catch (e) { /* ignore */ }
}

function togglePause() {
  if (!state || state.terminal) return;
  paused = !paused;
  updateButtons();
  renderStatus();
  scheduleAI();
}

function buildStatic() {
  const modes = $("modes");
  MODES.forEach((m) => {
    const b = document.createElement("button");
    b.className = "mode";
    b.dataset.id = m.id;
    b.innerHTML = `<span class="badge ${m.badge === "WATCH" ? "watch" : ""}">${m.badge}</span>
      <div class="t">${m.title}</div><div class="d">${m.desc}</div>`;
    b.onclick = () => { cfg.mode = m.id; ensureAudio(); sfx.click(); syncMenu(); };
    modes.appendChild(b);
  });

  document.querySelectorAll("#seg-board button").forEach((b) => {
    b.onclick = () => { cfg.board = b.dataset.board; ensureAudio(); sfx.click(); syncMenu(); };
  });
  document.querySelectorAll("#seg-diff button").forEach((b) => {
    b.onclick = () => { cfg.depth = parseInt(b.dataset.depth, 10); ensureAudio(); sfx.click(); syncMenu(); };
  });
  document.querySelectorAll("#seg-speed button").forEach((b) => {
    b.onclick = () => { cfg.speed = b.dataset.speed; updateButtons(); scheduleAI(); };
  });
  $("swap").onclick = () => { cfg.swap = !cfg.swap; ensureAudio(); sfx.click(); syncMenu(); };
  $("start").onclick = startGame;
  $("menu-btn").onclick = () => { sfx.click(); toMenu(); };
  $("restart").onclick = rematch;
  $("pause").onclick = togglePause;
  $("step").onclick = () => stepAI(epoch);
  $("sound").onclick = toggleSound;
  $("fs").onclick = toggleFullscreen;
  $("open-results").onclick = () => { sfx.click(); openResults(); };
  $("graphs-btn").onclick = openResults;
  $("results-back").onclick = closeResults;
  $("lightbox").onclick = () => { $("lightbox").hidden = true; };
  $("sound").textContent = "Sound: " + (soundOn ? "on" : "off");

  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const inGame = !$("game").hidden;
    const k = e.key.toLowerCase();
    if (!$("lightbox").hidden) { if (k === "escape") $("lightbox").hidden = true; return; }
    if (!$("results").hidden) { if (k === "escape") closeResults(); return; }
    if (k === "f") { toggleFullscreen(); return; }
    if (!inGame) { if (k === "enter") startGame(); return; }
    if (k === "m" || k === "escape") toMenu();
    else if (k === "r") rematch();
    else if (k === "p") togglePause();
    else if (k === "s") toggleSound();
    else if (/^[1-9]$/.test(k) && state) humanPlay(parseInt(k, 10) - 1);
  });

  window.addEventListener("resize", () => { fitBoard(); });
}

window.addEventListener("DOMContentLoaded", async () => {
  buildStatic();
  syncMenu();
  if (location.protocol === "file:") { showOffline(true); return; }   // opened the file directly
  try {
    tables = (await (await fetch("/api/info")).json()).tables || {};
  } catch (e) { showOffline(true); }
});
