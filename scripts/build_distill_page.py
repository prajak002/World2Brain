"""Build docs/distill3d.html: an interactive three.js walk through the distillation pipeline, layer by layer.

World frame -> V-JEPA teacher latent -> input layer into the VIS neurons -> connectome recurrence (6,935 neurons at their real
FlyWire positions, 91,663 signed synapses) -> readout -> what the state keeps in the dark. Reads results/brain3d.json
(positions, regions, a sample of edges with signs, and one replayed test episode per environment: activity, eye frames,
teacher and reconstructed latents, actions, heading). Numbers quoted in the chapters come from results/summary.md.

  PYTHONPATH=. python scripts/build_distill_page.py            # -> docs/distill3d.html
"""
import json, os

d = json.load(open("results/brain3d.json"))
data = json.dumps(d, separators=(",", ":"))

html = r'''<title>World2Brain Distillation</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Source+Sans+3:wght@400;600;700&display=swap">
<style>
/* Layout: a sticky 3D stage (the whole pipeline laid out left to right) and a scroll column of chapters that fly the camera to one layer at a time. Dark-first, the project's existing identity: light neurons on a deep ground, region colours shared with the paper figures. */
:root{--bg:#0d0f14;--bg2:#171a22;--bg3:#1f232d;--fg:#eceae2;--fg2:#a09e95;--line:#2c3039;--fly:#4fc3c7;--ai:#e6a054;--dark:#ff6b63;--good:#4cc27a;
 --vis:#56b4e9;--cx:#f0a940;--mb:#d55181;--out:#9085e9;--display:"Fraunces",Georgia,serif;--body:"Source Sans 3","Helvetica Neue",Arial,sans-serif;color-scheme:dark}
@media (prefers-color-scheme: light){:root:not([data-theme="dark"]){--bg:#f6f4ee;--bg2:#e9e6dd;--bg3:#dedbd0;--fg:#1b1a18;--fg2:#5d5b54;--line:#d3d0c5;--fly:#0f6f73;--ai:#b8681a;--dark:#c2423b;--good:#1b8a4c;--vis:#1f78b4;--cx:#c77400;--mb:#b4306a;--out:#5a4bd1;color-scheme:light}}
:root[data-theme="light"]{--bg:#f6f4ee;--bg2:#e9e6dd;--bg3:#dedbd0;--fg:#1b1a18;--fg2:#5d5b54;--line:#d3d0c5;--fly:#0f6f73;--ai:#b8681a;--dark:#c2423b;--good:#1b8a4c;--vis:#1f78b4;--cx:#c77400;--mb:#b4306a;--out:#5a4bd1;color-scheme:light}
body{background:var(--bg);color:var(--fg);font-family:var(--body);font-size:18px;line-height:1.5;padding-block:0 60px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.top{max-width:1280px;margin:0 auto;padding-top:36px}
.kicker{font-family:var(--display);color:var(--fly);font-size:15px;letter-spacing:.04em;text-transform:uppercase;font-weight:600}
h1{font-family:var(--display);font-size:clamp(30px,4.6vw,52px);font-weight:700;line-height:1.08;margin:6px 0 0;text-wrap:balance}
.lede{color:var(--fg2);font-size:20px;max-width:72ch;margin:12px 0 0}
.stats{display:flex;flex-wrap:wrap;gap:10px 26px;margin-top:16px;font-size:15px;color:var(--fg2)}.stats b{color:var(--fg);font-variant-numeric:tabular-nums;font-size:22px;font-family:var(--display);font-weight:600;margin-right:6px}
.layout{max-width:1280px;margin:26px auto 0;display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,.8fr);gap:28px;align-items:start}
.stage{position:sticky;top:12px;height:calc(100vh - 24px);min-height:460px;background:var(--bg2);border-radius:16px;overflow:hidden;display:flex;flex-direction:column}
canvas.gl{width:100%;flex:1;min-height:0;display:block;touch-action:none;cursor:grab}
.layers{display:flex;gap:4px;padding:8px 10px 0;flex-wrap:wrap}
.layers button{font:inherit;font-size:13px;font-weight:600;background:transparent;color:var(--fg2);border:1px solid var(--line);border-radius:999px;padding:4px 11px;cursor:pointer}
.layers button[aria-pressed="true"]{color:var(--fg);border-color:var(--fly);background:var(--bg3)}
.layers button:focus-visible,.hud button:focus-visible{outline:2px solid var(--fly)}
.panel{display:grid;grid-template-columns:auto 1fr auto;gap:12px;align-items:center;padding:8px 14px;border-top:1px solid var(--line);font-size:13px}
.eye{image-rendering:pixelated;width:128px;height:32px;border:1px solid var(--line);border-radius:4px;background:#000}
.lat{height:44px;width:100%}
.gauge{width:120px}.gauge .bar{height:10px;border-radius:5px;background:var(--line);overflow:hidden}.gauge i{display:block;height:100%;background:var(--fly);width:0;transition:width .25s}
.hud{padding:10px 14px;display:grid;grid-template-columns:1fr auto;gap:8px 14px;align-items:center;border-top:1px solid var(--line);font-size:14px}
.hud .row{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
.hud button{font:inherit;font-weight:600;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:5px 11px;cursor:pointer}.hud button[aria-pressed="true"]{border-color:var(--fly);color:var(--fly)}
input[type=range]{width:100%;accent-color:var(--fly)}
.steplbl{font-variant-numeric:tabular-nums;font-weight:700}.steplbl.dark{color:var(--dark)}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:13px;color:var(--fg2)}.legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;background:var(--c);vertical-align:-1px}
.dim{color:var(--fg2);font-size:15px}
.chapters{min-width:0}
.ch{min-height:76vh;padding:28px 0 48px;border-top:1px solid var(--line)}.ch:first-child{border-top:0}
.ch .k{font-family:var(--display);font-size:15px;color:var(--fly);font-weight:600;letter-spacing:.03em;text-transform:uppercase}
.ch h2{font-family:var(--display);font-size:clamp(24px,3vw,34px);font-weight:600;line-height:1.12;margin:6px 0 10px;text-wrap:balance}
.ch p{margin:10px 0;max-width:60ch}
.ai{color:var(--ai);font-weight:700}.fly{color:var(--fly);font-weight:700}.darkw{color:var(--dark);font-weight:700}
.note{background:var(--bg2);border-left:4px solid var(--fly);padding:10px 14px;border-radius:0 10px 10px 0;font-size:16px;margin:12px 0}
table.t{border-collapse:collapse;font-size:15px;margin:10px 0;font-variant-numeric:tabular-nums}.t th,.t td{text-align:left;padding:5px 12px 5px 0;border-bottom:1px solid var(--line)}.t th{color:var(--fg2);font-weight:600}.t td.n{text-align:right}.t th.n{text-align:right}
.t tr.hi td{color:var(--fly);font-weight:700}
.wrap{overflow-x:auto}
.cta{display:flex;flex-wrap:wrap;gap:10px;margin-top:14px}.cta a{font-weight:700;color:var(--fly);text-decoration:none;border:1px solid var(--fly);border-radius:999px;padding:7px 14px}.cta a:hover{background:var(--bg3)}
.share{border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-top:10px}.share textarea{width:100%;min-height:150px;font:inherit;font-size:15px;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:8px;resize:vertical}
.share button{font:inherit;font-weight:600;background:var(--fly);color:#06202a;border:0;border-radius:8px;padding:7px 14px;cursor:pointer;margin-top:8px}
@media(max-width:860px){.layout{grid-template-columns:1fr;gap:16px}.stage{position:sticky;top:8px;height:56vh;min-height:380px}.ch{min-height:auto;padding:20px 0 28px}.panel{grid-template-columns:auto 1fr}.gauge{display:none}}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="top">
<div class="kicker">World2Brain · a video world model distilled into a fly's wiring diagram</div>
<h1>Watch a video AI get poured into 6,935 real neurons, one layer at a time</h1>
<p class="lede">Every dot on the left is a real fruit-fly neuron, placed where its cell body sits in the FlyWire connectome, and every line is a real synapse. The only thing we trained was how strongly each synapse fires. Scroll the right column to follow one walk through the pipeline: what the world looks like, what the V-JEPA teacher sees, how it enters the fly's eye, what 91,663 synapses do with it, and what the brain still knows after the lights go out.</p>
<div class="stats"><span><b>6,935</b>neurons</span><span><b>91,663</b>signed synapses</span><span><b>1</b>training signal: predict the teacher's next latent</span><span><b>8</b>dark steps per test</span></div>
</div>

<div class="layout">
<div class="stage" id="stage">
<div class="layers" id="layers" role="tablist" aria-label="pipeline layer">
<button data-l="0" aria-pressed="false">1 · world</button><button data-l="1" aria-pressed="false">2 · teacher</button><button data-l="2" aria-pressed="false">3 · input</button><button data-l="3" aria-pressed="true">4 · connectome</button><button data-l="4" aria-pressed="false">5 · readout</button><button data-l="5" aria-pressed="false">6 · in the dark</button><button data-l="6" aria-pressed="false">all</button>
</div>
<canvas class="gl" id="gl" aria-label="3D view of the distillation pipeline"></canvas>
<div class="panel">
<div><div class="dim" style="margin-bottom:4px">what the brain sees (32×8 px)</div><canvas class="eye" id="eye" width="32" height="8"></canvas></div>
<div><div class="dim" style="margin-bottom:2px"><span class="ai">teacher latent</span> vs <span class="fly">brain's guess</span> (first 48 of 256 numbers)</div><canvas class="lat" id="lat" width="600" height="44"></canvas></div>
<div class="gauge"><div class="dim" style="margin-bottom:4px">match <b id="matchv">–</b></div><div class="bar"><i id="matchbar"></i></div></div>
</div>
<div class="hud">
<div class="row"><span class="steplbl" id="steplbl">step 1 / 24</span><span id="actlbl" class="dim"></span></div>
<div class="row"><button id="play" aria-pressed="true">pause</button><button id="world" aria-pressed="false">NYC twin</button><button id="edges" aria-pressed="true">synapses</button><button id="shuf" aria-pressed="false">shuffle wiring</button></div>
<input type="range" id="step" min="0" max="23" value="0" aria-label="time step" style="grid-column:1/-1">
<div class="legend" style="grid-column:1/-1"><span style="--c:var(--vis)">VIS visual projection 2,330</span><span style="--c:var(--cx)">CX central complex 2,875</span><span style="--c:var(--mb)">MB mushroom body 427</span><span style="--c:var(--out)">OUT descending 1,303</span><span style="--c:var(--ai)">excitatory synapse</span><span style="--c:var(--out)">inhibitory synapse</span></div>
</div>
</div>

<div class="chapters" id="chapters">
<section class="ch" data-l="0"><div class="k">layer 1 · the world</div><h2>A little walker in a big room</h2>
<p>An agent walks through a 360° arena with coloured landmarks, or through a raycast twin of Midtown Manhattan built from real building footprints. It has five moves: stay, step forward, turn left 22.5°, turn right 22.5°, step back. Every episode is 24 steps.</p>
<p>The strip at the bottom of the stage is the <b>only</b> thing the brain is given: a 32×8 pixel fly-eye band of the panorama. The teacher sees the full 256 px frame.</p>
<p class="dim">Use the <b>NYC twin</b> button to swap the episode. Drag the stage to rotate, scroll on it to zoom.</p></section>

<section class="ch" data-l="1"><div class="k">layer 2 · the teacher</div><h2><span class="ai">V-JEPA 2.1</span> turns each frame into 3,840 numbers</h2>
<p>The teacher is a frozen video foundation model (ViT-B). For every step it encodes the clip that ends at that step, so a target never contains frames the student has not yet seen. We keep the mean token and a 2×2 grid of pooled tokens, 3,840 numbers, and project them to 256 principal components. That 256-vector is the <span class="ai">orange bar chart</span>: the thing the fly brain must learn to predict.</p>
<div class="note">Nothing about heading, position or actions is ever supervised. The brain is only asked: what will the teacher say next?</div></section>

<section class="ch" data-l="2"><div class="k">layer 3 · into the eye</div><h2>Light enters through 2,330 visual neurons only</h2>
<p>The fly-eye band is pushed through one input layer into the <span style="color:var(--vis);font-weight:700">VIS</span> population: TuBu, MeTu and LC visual projection neurons, the cells that in a real fly carry visual information into the navigation circuit. The action the agent just took enters the <span style="color:var(--cx);font-weight:700">central complex</span> and the <span style="color:var(--out);font-weight:700">descending neurons</span>. No other neuron receives anything from outside.</p>
<p>In the paper this input layer is a linear map. The current GPU run also tests a small cross-attention encoder, where each VIS neuron owns a query that looks over patches of the image, which cuts the input parameters by 94 percent and leaves the recurrence as the dominant part of the model.</p></section>

<section class="ch" data-l="3"><div class="k">layer 4 · the connectome</div><h2>91,663 synapses, copied wire for wire from the fly</h2>
<p>This is the whole "thinking" part. Each neuron is a leaky unit, and the only recurrent connections are the real FlyWire synapses between these 6,935 cells. Every synapse keeps its biological sign (Dale's law, from the predicted neurotransmitter). Training can make a synapse stronger or weaker. It can never flip one, add one or remove one.</p>
<p>Press <b>shuffle wiring</b> to see what a control looks like: same neurons, same number of edges, same signs, but the endpoints rewired. The paper compares the real graph against degree-preserving, block-preserving and random rewirings, and against GRU and transformer cores with the same parameter budget.</p>
<div class="note">Colour is region. Brightness is activity right now. Watch the activity ripple as the step slider moves.</div></section>

<section class="ch" data-l="4"><div class="k">layer 5 · the readout</div><h2>One linear readout guesses the teacher</h2>
<p>A single linear layer reads all 6,935 activities and produces the <span class="fly">teal bars</span>: the brain's guess of the teacher's 256-number latent. The gauge at the bottom is the cosine match between guess and truth.</p>
<p>Here is the first honest result. The fly wiring is <b>not</b> the best predictor. A parameter-matched GRU is about 25 percent better at this, and a transformer predictor is in the same run. Prediction quality is not what the wiring buys.</p></section>

<section class="ch" data-l="5"><div class="k">layer 6 · lights out</div><h2>Steps 9 to 16: the eye goes <span class="darkw">black</span></h2>
<p>During training, random stretches of each episode are blacked out. At test, steps 9 to 16 are dark. The brain receives only its own actions and must keep guessing what the teacher would see.</p>
<p>Now the second result, and the reason this project exists. If you fit a simple linear decoder to the brain's activity during those dark steps, you can read out where it thinks it is facing and where it is, relative to where the dark began:</p>
<div class="wrap"><table class="t"><tr><th>recurrent core, same budget</th><th class="n">heading error in the dark</th><th class="n">position R²</th></tr>
<tr class="hi"><td>fly connectome</td><td class="n">9.1°</td><td class="n">0.94</td></tr>
<tr><td>same graph, block-shuffled</td><td class="n">10.8°</td><td class="n">0.93</td></tr>
<tr><td>random sparse graph</td><td class="n">12.9°</td><td class="n">0.88</td></tr>
<tr><td>dense, every weight free</td><td class="n">21.3°</td><td class="n">0.68</td></tr>
<tr><td>GRU, parameter-matched</td><td class="n">30.6°</td><td class="n">0.41</td></tr>
<tr><td>linear readout of actions alone</td><td class="n">40.1°</td><td class="n">0.20</td></tr></table></div>
<p class="dim">Median over 8 dark steps, 5 seeds, arena; from results/summary.md. The transformer core and the continuous-turn version are training now and will be added to this table.</p>
<p>The state path-integrates. Real flies do exactly this to find home in the dark, and nobody told this brain to. Sparse signed wiring of this size forms the code; the real synapse list is a small, consistent bonus of 2 to 6 degrees over scrambled wiring.</p></section>

<section class="ch" data-l="6"><div class="k">the whole pipeline</div><h2>What this is, and what it is not</h2>
<p>It is a world model whose recurrent memory is a real brain's wiring diagram, and the first evidence that such a memory keeps a heading and a home vector that a generic recurrent core trained on the same objective does not.</p>
<p>It is not yet a navigator. A linear readout of the state gets the agent home in 37 percent of city episodes against a 92 percent dead-reckoning ceiling, goal-directed navigation from the state does not beat random actions, and on real handheld video no student beats copying the last frame. Those are reported as negatives, and closing them, with real scenes rebuilt as 3D Gaussian splats and transformer baselines in the same table, is the current work.</p>
<div class="cta"><a href="https://github.com/prajak002/World2Brain">Code, data and checkpoints on GitHub</a></div>
<div class="share"><div class="dim" style="margin-bottom:6px">post text (edit, then copy)</div>
<textarea id="post">We took the fruit fly's real wiring diagram (6,935 neurons, 91,663 synapses from FlyWire) and used it as the only recurrent memory of a video world model distilled from V-JEPA 2.1.

The wiring is a worse predictor than a GRU. But when the lights go out, its state keeps heading to 9° and position to R² 0.94 over 8 blind steps. The GRU: 31° and 0.41. Nobody supervised heading. The fly's circuit path-integrates on its own.

Interactive walkthrough, layer by layer, in the link. Code and checkpoints are open.</textarea>
<button id="copy">copy post text</button> <span id="copied" class="dim"></span></div></section>
</div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const D = __DATA__;
const N = D.pos.length, REG = D.region;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const col = () => ({VIS: new THREE.Color(css('--vis')), CX: new THREE.Color(css('--cx')), MB: new THREE.Color(css('--mb')), OUT: new THREE.Color(css('--out')), ai: new THREE.Color(css('--ai')), fly: new THREE.Color(css('--fly')), inh: new THREE.Color(css('--out')), dark: new THREE.Color(css('--dark'))});
let C = col();
const dark = () => { const t = document.documentElement.getAttribute('data-theme'); return t ? t === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches; };

// ---------- scene ----------
const cv = document.getElementById('gl'); const R = new THREE.WebGLRenderer({canvas: cv, antialias: true, alpha: true}); R.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene(); const cam = new THREE.PerspectiveCamera(42, 1, 0.1, 200);
const S = 1 / 95;                                           // FlyWire µm -> scene units; the brain spans about 7 x 3 x 2.5 units
const LX = {world: -11.5, teacher: -7.2, brain: 0, readout: 6.8, dial: 10.6};
const centroid = {}; for (const r of ['VIS', 'CX', 'MB', 'OUT']) { const v = new THREE.Vector3(), n = []; D.pos.forEach((p, i) => { if (REG[i] === r) { v.x += p[0]; v.y += p[1]; v.z += p[2]; n.push(i); } }); centroid[r] = v.multiplyScalar(S / n.length); }

// neurons
const pos = new Float32Array(N * 3), colBuf = new Float32Array(N * 3), base = new Float32Array(N * 3);
D.pos.forEach((p, i) => { pos[3 * i] = p[0] * S; pos[3 * i + 1] = p[1] * S; pos[3 * i + 2] = p[2] * S; });
const pg = new THREE.BufferGeometry(); pg.setAttribute('position', new THREE.BufferAttribute(pos, 3)); pg.setAttribute('color', new THREE.BufferAttribute(colBuf, 3));
const sprite = (() => { const c = document.createElement('canvas'); c.width = c.height = 64; const g = c.getContext('2d'); const gr = g.createRadialGradient(32, 32, 2, 32, 32, 30); gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.5, 'rgba(255,255,255,.55)'); gr.addColorStop(1, 'rgba(255,255,255,0)'); g.fillStyle = gr; g.fillRect(0, 0, 64, 64); return new THREE.CanvasTexture(c); })();
const pm = new THREE.PointsMaterial({size: 0.085, vertexColors: true, map: sprite, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true});
const points = new THREE.Points(pg, pm); scene.add(points);

// synapses (a 6,000-edge sample of the 91,663), real and a visual shuffle
const E = D.edges.length; const ePos = new Float32Array(E * 6), eCol = new Float32Array(E * 6);
const eg = new THREE.BufferGeometry(); eg.setAttribute('position', new THREE.BufferAttribute(ePos, 3)); eg.setAttribute('color', new THREE.BufferAttribute(eCol, 3));
const lines = new THREE.LineSegments(eg, new THREE.LineBasicMaterial({vertexColors: true, transparent: true, opacity: 0.22, depthWrite: false, blending: THREE.AdditiveBlending})); scene.add(lines);
let shuffled = false; const rnd = (() => { let s = 7; return () => (s = (s * 16807) % 2147483647) / 2147483647; })(); const shufTarget = D.edges.map(() => Math.floor(rnd() * N));
function layEdges() { for (let k = 0; k < E; k++) { const a = D.edges[k][0], b = shuffled ? shufTarget[k] : D.edges[k][1]; ePos.set([pos[3 * a], pos[3 * a + 1], pos[3 * a + 2], pos[3 * b], pos[3 * b + 1], pos[3 * b + 2]], 6 * k); const c = D.esign[k] > 0 ? C.ai : C.inh; eCol.set([c.r, c.g, c.b, c.r, c.g, c.b], 6 * k); } eg.attributes.position.needsUpdate = true; eg.attributes.color.needsUpdate = true; }

// world frame: the fly-eye band as a texture, scaled up, with a frame
const eyeTex = (() => { const c = document.createElement('canvas'); c.width = 32; c.height = 8; const t = new THREE.CanvasTexture(c); t.magFilter = THREE.NearestFilter; t.minFilter = THREE.NearestFilter; return {c, t}; })();
const worldPlane = new THREE.Mesh(new THREE.PlaneGeometry(4.4, 1.1), new THREE.MeshBasicMaterial({map: eyeTex.t})); worldPlane.position.set(LX.world, 0.9, 0); scene.add(worldPlane);
const frame = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.PlaneGeometry(4.6, 1.3)), new THREE.LineBasicMaterial({color: 0x888888, transparent: true, opacity: .6})); frame.position.copy(worldPlane.position); scene.add(frame);
const darkPlane = new THREE.Mesh(new THREE.PlaneGeometry(4.4, 1.1), new THREE.MeshBasicMaterial({color: 0x000000, transparent: true, opacity: 0})); darkPlane.position.set(LX.world, 0.9, 0.01); scene.add(darkPlane);

// latent bars: teacher (orange) and reconstruction (teal), 48 of 256
const K = 48; function bars(x, c) { const g = new THREE.Group(); for (let i = 0; i < K; i++) { const m = new THREE.Mesh(new THREE.BoxGeometry(0.07, 1, 0.07), new THREE.MeshBasicMaterial({color: c, transparent: true, opacity: .9})); m.position.set(x + (i - K / 2) * 0.085, 0, 0); g.add(m); } scene.add(g); return g; }
const barsZ = bars(LX.teacher, C.ai), barsZh = bars(LX.readout, C.fly);
function setBars(g, v) { g.children.forEach((m, i) => { const h = Math.max(0.03, Math.min(2.2, Math.abs(v[i]) * 0.9)); m.scale.y = h; m.position.y = (v[i] >= 0 ? h : -h) / 2; }); }

// flow lines: eye -> VIS centroid, readout <- brain
function flow(a, b, c) { const g = new THREE.BufferGeometry().setFromPoints([a, b]); const l = new THREE.Line(g, new THREE.LineDashedMaterial({color: c, dashSize: .25, gapSize: .18, transparent: true, opacity: .7})); l.computeLineDistances(); scene.add(l); return l; }
const f1 = flow(new THREE.Vector3(LX.world + 2.3, 0.9, 0), new THREE.Vector3(LX.teacher - 2.2, 0, 0), C.ai.getHex());
const f2 = flow(new THREE.Vector3(LX.world + 2.3, 0.6, 0), centroid.VIS.clone(), C.VIS.getHex());
const f3 = flow(new THREE.Vector3(centroid.OUT.x + 1.2, centroid.OUT.y, 0), new THREE.Vector3(LX.readout - 2.2, 0, 0), C.fly.getHex());
const f4 = flow(new THREE.Vector3(LX.readout + 2.2, 0, 0), new THREE.Vector3(LX.dial - 1.2, 0, 0), C.fly.getHex());

// heading dial: true heading (white) and the dark-window drift the paper measures (teal median 9°, grey GRU median 31°), drawn as an illustration
const dial = new THREE.Group(); dial.position.set(LX.dial, 0, 0); scene.add(dial);
dial.add(new THREE.Mesh(new THREE.RingGeometry(1.0, 1.06, 64), new THREE.MeshBasicMaterial({color: 0x888888, transparent: true, opacity: .7, side: THREE.DoubleSide})));
function arrow(c, len, w) { const m = new THREE.Mesh(new THREE.ConeGeometry(w, len, 12), new THREE.MeshBasicMaterial({color: c})); const g = new THREE.Group(); m.position.y = len / 2; g.add(m); dial.add(g); return g; }
const aTrue = arrow(0xffffff, 0.95, 0.09), aFly = arrow(C.fly.getHex(), 0.8, 0.07), aGru = arrow(0x9a9a9a, 0.65, 0.06);

// labels as sprites
function label(text, x, y, sub) { const c = document.createElement('canvas'); c.width = 512; c.height = 128; const g = c.getContext('2d'); g.font = '600 46px Fraunces, Georgia, serif'; g.fillStyle = dark() ? '#eceae2' : '#1b1a18'; g.textAlign = 'center'; g.fillText(text, 256, 56); if (sub) { g.font = '400 30px "Source Sans 3", Arial, sans-serif'; g.fillStyle = dark() ? '#a09e95' : '#5d5b54'; g.fillText(sub, 256, 104); } const t = new THREE.CanvasTexture(c); const s = new THREE.Sprite(new THREE.SpriteMaterial({map: t, transparent: true, depthWrite: false})); s.scale.set(3.2, 0.8, 1); s.position.set(x, y, 0); scene.add(s); return s; }
const labels = [];
function makeLabels() { labels.forEach(l => scene.remove(l)); labels.length = 0;
  labels.push(label('1 · world', LX.world, 2.1, 'fly-eye band, 32×8'), label('2 · teacher', LX.teacher, 2.1, 'V-JEPA latent, 256-d'), label('3 · input', (LX.teacher + centroid.VIS.x) / 2 - 0.8, -2.2, 'into VIS neurons only'),
    label('4 · connectome', 0, 2.5, '6,935 neurons · 91,663 synapses'), label('5 · readout', LX.readout, 2.1, "the brain's guess"), label('6 · heading', LX.dial, 2.1, 'what the dark state keeps')); }
makeLabels();

// ---------- camera control: orbit around a focus that the chapters move ----------
let yaw = -0.35, pitch = 0.18, dist = 9, target = new THREE.Vector3(0, 0, 0), want = {yaw, pitch, dist, target: target.clone()};
const FOCUS = [
  {t: [LX.world, 0.6, 0], d: 6.5, yaw: -0.15, pitch: 0.1}, {t: [LX.teacher, 0.3, 0], d: 6.5, yaw: -0.2, pitch: 0.15}, {t: [(LX.teacher + centroid.VIS.x) / 2, 0, 0], d: 8, yaw: -0.4, pitch: 0.25},
  {t: [0, 0.2, 0], d: 7.5, yaw: -0.5, pitch: 0.22}, {t: [LX.readout, 0.2, 0], d: 6.5, yaw: 0.2, pitch: 0.15}, {t: [LX.dial, 0.2, 0], d: 6, yaw: 0.3, pitch: 0.1}, {t: [0, 0.3, 0], d: 19, yaw: -0.25, pitch: 0.2}];
let layer = 3; function setLayer(l) { layer = l; const f = FOCUS[l]; want.target.set(...f.t); want.dist = f.d; want.yaw = f.yaw; want.pitch = f.pitch;
  document.querySelectorAll('#layers button').forEach(b => b.setAttribute('aria-pressed', String(+b.dataset.l === l))); }
let drag = null; cv.addEventListener('pointerdown', e => { drag = {x: e.clientX, y: e.clientY}; cv.setPointerCapture(e.pointerId); });
cv.addEventListener('pointermove', e => { if (!drag) return; want.yaw += (e.clientX - drag.x) * 0.006; want.pitch = Math.max(-1.2, Math.min(1.2, want.pitch + (e.clientY - drag.y) * 0.006)); drag = {x: e.clientX, y: e.clientY}; });
cv.addEventListener('pointerup', () => drag = null); cv.addEventListener('wheel', e => { e.preventDefault(); want.dist = Math.max(2.5, Math.min(30, want.dist * (1 + e.deltaY * 0.0012))); }, {passive: false});
function resize() { const w = cv.clientWidth, h = cv.clientHeight; if (cv.width !== w * R.getPixelRatio() || cv.height !== h * R.getPixelRatio()) { R.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix(); } }

// ---------- episode replay ----------
let world = 'arena', step = 0, playing = true, last = 0; const T = 24, BLIND = [8, 16];
const eye2d = document.getElementById('eye').getContext('2d'), lat2d = document.getElementById('lat').getContext('2d');
const ACT = ['stay', 'forward', 'turn left', 'turn right', 'back'];
function paintEye(obs, blind) { const img = eye2d.createImageData(32, 8); for (let y = 0; y < 8; y++) for (let x = 0; x < 32; x++) { const i = 4 * (y * 32 + x), p = obs[y][x]; img.data[i] = blind ? 0 : p[0]; img.data[i + 1] = blind ? 0 : p[1]; img.data[i + 2] = blind ? 0 : p[2]; img.data[i + 3] = 255; } eye2d.putImageData(img, 0, 0); eyeTex.c.getContext('2d').putImageData(img, 0, 0); eyeTex.t.needsUpdate = true; }
function paintLat(z, zh) { const g = lat2d, W = 600, H = 44; g.clearRect(0, 0, W, H); g.fillStyle = css('--line'); g.fillRect(0, H / 2, W, 1); const bw = W / K;
  for (let i = 0; i < K; i++) { const a = -z[i] * 7, b = -zh[i] * 7; g.fillStyle = css('--ai'); g.fillRect(i * bw + 1, H / 2 + Math.min(0, a), bw / 2 - 1, Math.abs(a)); g.fillStyle = css('--fly'); g.fillRect(i * bw + bw / 2, H / 2 + Math.min(0, b), bw / 2 - 1, Math.abs(b)); } }
function show(s) { step = s; const ep = D[world], blind = s >= BLIND[0] && s < BLIND[1], A = ep.act[s];
  for (let i = 0; i < N; i++) { const c = C[REG[i]], a = Math.min(1, Math.abs(A[i]) * 1.6), k = 0.18 + 0.82 * a; colBuf[3 * i] = c.r * k; colBuf[3 * i + 1] = c.g * k; colBuf[3 * i + 2] = c.b * k; }
  pg.attributes.color.needsUpdate = true;
  paintEye(ep.obs[s], blind); paintLat(ep.z[s], ep.zh[s]); setBars(barsZ, ep.z[s]); setBars(barsZh, ep.zh[s]);
  darkPlane.material.opacity = blind ? 0.92 : 0; f2.material.opacity = blind ? 0.08 : 0.7;
  const m = ep.match[s]; document.getElementById('matchv').textContent = m.toFixed(2); document.getElementById('matchbar').style.width = Math.max(0, m) * 100 + '%';
  const lbl = document.getElementById('steplbl'); lbl.textContent = `step ${s + 1} / ${T}` + (blind ? ' · DARK' : ''); lbl.classList.toggle('dark', blind);
  document.getElementById('actlbl').textContent = s > 0 ? `action before this step: ${ACT[ep.actions[s - 1]]}` : 'episode start';
  document.getElementById('step').value = s;
  // heading dial: true heading relative to step 1, and the paper's median dark-window errors drawn as drift that grows over the dark steps
  const h0 = ep.heading[0], h = ep.heading[s] - h0; aTrue.rotation.z = -h; const k = blind ? (s - BLIND[0] + 1) / 8 : (s >= BLIND[1] ? 0.3 : 0);
  aFly.rotation.z = -h + k * 9.1 * Math.PI / 180; aGru.rotation.z = -h + k * 30.6 * Math.PI / 180; }
document.getElementById('step').addEventListener('input', e => { playing = false; document.getElementById('play').textContent = 'play'; document.getElementById('play').setAttribute('aria-pressed', 'false'); show(+e.target.value); });
document.getElementById('play').addEventListener('click', e => { playing = !playing; e.currentTarget.textContent = playing ? 'pause' : 'play'; e.currentTarget.setAttribute('aria-pressed', String(playing)); });
document.getElementById('world').addEventListener('click', e => { world = world === 'arena' ? 'nyc' : 'arena'; e.currentTarget.setAttribute('aria-pressed', String(world === 'nyc')); e.currentTarget.textContent = world === 'nyc' ? 'arena' : 'NYC twin'; show(step); });
document.getElementById('edges').addEventListener('click', e => { lines.visible = !lines.visible; e.currentTarget.setAttribute('aria-pressed', String(lines.visible)); });
document.getElementById('shuf').addEventListener('click', e => { shuffled = !shuffled; e.currentTarget.setAttribute('aria-pressed', String(shuffled)); layEdges(); });
document.querySelectorAll('#layers button').forEach(b => b.addEventListener('click', () => setLayer(+b.dataset.l)));
const io = new IntersectionObserver(es => es.forEach(x => { if (x.isIntersecting) setLayer(+x.target.dataset.l); }), {rootMargin: '-45% 0px -45% 0px'});
document.querySelectorAll('.ch').forEach(s => io.observe(s));
document.getElementById('copy').addEventListener('click', async () => { const t = document.getElementById('post'); try { await navigator.clipboard.writeText(t.value); document.getElementById('copied').textContent = 'copied'; } catch (e) { t.select(); document.getElementById('copied').textContent = 'select and copy'; } });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { C = col(); layEdges(); makeLabels(); show(step); });

layEdges(); show(0); setLayer(3);
function tick(t) { requestAnimationFrame(tick); resize();
  if (playing && t - last > 650) { last = t; show((step + 1) % T); }
  yaw += (want.yaw - yaw) * 0.06; pitch += (want.pitch - pitch) * 0.06; dist += (want.dist - dist) * 0.06; target.lerp(want.target, 0.06);
  cam.position.set(target.x + dist * Math.sin(yaw) * Math.cos(pitch), target.y + dist * Math.sin(pitch), target.z + dist * Math.cos(yaw) * Math.cos(pitch)); cam.lookAt(target);
  points.rotation.y = 0; const dim = layer === 6 ? 1 : 0.25; [worldPlane, frame].forEach(m => m.material.opacity = (layer === 0 || layer === 6) ? 1 : dim + 0.2);
  R.render(scene, cam); }
requestAnimationFrame(tick);
</script>
'''

os.makedirs("docs", exist_ok=True)
open("docs/distill3d.html", "w", encoding="utf-8").write(html.replace("__DATA__", data))
print("wrote docs/distill3d.html", os.path.getsize("docs/distill3d.html") / 1e6, "MB")
