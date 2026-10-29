data = open("brain3d.json").read()
html = r'''<title>Fly Brain in 3D</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Source+Sans+3:wght@400;600;700&display=swap">
<style>
/* Layout: a sticky 3D brain on the left, a long column of annotated chapters on the right that drive the scene as you scroll. Dark-first: the brain is drawn as light on a deep ground. */
:root{--bg:#0d0f14;--bg2:#171a22;--fg:#eceae2;--fg2:#a09e95;--line:#2c3039;--fly:#4fc3c7;--ai:#e6a054;--dark:#ff6b63;--good:#4cc27a;
 --vis:#56b4e9;--cx:#f0a940;--mb:#d55181;--out:#9085e9;--display:"Fraunces",Georgia,serif;--body:"Source Sans 3","Helvetica Neue",Arial,sans-serif;color-scheme:dark}
@media (prefers-color-scheme: light){:root:not([data-theme="dark"]){--bg:#f6f4ee;--bg2:#e9e6dd;--fg:#1b1a18;--fg2:#5d5b54;--line:#d3d0c5;--fly:#0f6f73;--ai:#b8681a;--dark:#c2423b;--good:#1b8a4c;--vis:#1f78b4;--cx:#c77400;--mb:#b4306a;--out:#5a4bd1;color-scheme:light}}
:root[data-theme="light"]{--bg:#f6f4ee;--bg2:#e9e6dd;--fg:#1b1a18;--fg2:#5d5b54;--line:#d3d0c5;--fly:#0f6f73;--ai:#b8681a;--dark:#c2423b;--good:#1b8a4c;--vis:#1f78b4;--cx:#c77400;--mb:#b4306a;--out:#5a4bd1;color-scheme:light}
body{background:var(--bg);color:var(--fg);font-family:var(--body);font-size:18px;line-height:1.5;padding-block:0 60px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.top{max-width:1280px;margin:0 auto;padding-top:36px}
h1{font-family:var(--display);font-size:clamp(30px,4.5vw,50px);font-weight:700;line-height:1.1;margin:0;text-wrap:balance}
.lede{color:var(--fg2);font-size:20px;max-width:70ch;margin:12px 0 0}
.layout{max-width:1280px;margin:28px auto 0;display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,.85fr);gap:28px;align-items:start}
.stage{position:sticky;top:12px;height:calc(100vh - 24px);min-height:420px;background:var(--bg2);border-radius:16px;overflow:hidden;display:flex;flex-direction:column}
canvas.gl{width:100%;flex:1;min-height:0;display:block;touch-action:none;cursor:grab}
.hud{padding:10px 14px;display:grid;grid-template-columns:1fr auto;gap:8px 14px;align-items:center;border-top:1px solid var(--line);font-size:14px}
.hud .row{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
.hud button{font:inherit;font-weight:600;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:5px 11px;cursor:pointer}.hud button[aria-pressed="true"]{border-color:var(--fly);color:var(--fly)}
.hud button:focus-visible{outline:2px solid var(--fly)}
input[type=range]{width:100%;accent-color:var(--fly)}
.steplbl{font-variant-numeric:tabular-nums;font-weight:700}.steplbl.dark{color:var(--dark)}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:13px;color:var(--fg2)}.legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;background:var(--c);vertical-align:-1px}
.panel{display:grid;grid-template-columns:auto 1fr auto;gap:12px;align-items:center;padding:8px 14px;border-top:1px solid var(--line);font-size:13px}
.eye{image-rendering:pixelated;width:128px;height:32px;border:1px solid var(--line);border-radius:4px;background:#000}
.lat{height:44px;width:100%}
.gauge{width:120px}.gauge .bar{height:10px;border-radius:5px;background:var(--line);overflow:hidden}.gauge i{display:block;height:100%;background:var(--fly);width:0;transition:width .25s}
.chapters{min-width:0}
.ch{min-height:72vh;padding:28px 0 48px;border-top:1px solid var(--line)}.ch:first-child{border-top:0}
.ch .k{font-family:var(--display);font-size:15px;color:var(--fly);font-weight:600;letter-spacing:.03em;text-transform:uppercase}
.ch h2{font-family:var(--display);font-size:clamp(24px,3vw,34px);font-weight:600;line-height:1.12;margin:6px 0 10px;text-wrap:balance}
.ch p{margin:10px 0;max-width:60ch}
.ai{color:var(--ai);font-weight:700}.fly{color:var(--fly);font-weight:700}.darkw{color:var(--dark);font-weight:700}
.note{background:var(--bg2);border-left:4px solid var(--fly);padding:10px 14px;border-radius:0 10px 10px 0;font-size:16px;margin:12px 0}
.notes{border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-top:10px}
.notes textarea{width:100%;min-height:72px;font:inherit;font-size:16px;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:8px;resize:vertical}
.notes button{font:inherit;font-weight:600;background:var(--fly);color:#06202a;border:0;border-radius:8px;padding:7px 14px;cursor:pointer;margin-right:8px}
.notes ul{list-style:none;padding:0;margin:12px 0 0;display:grid;gap:8px}.notes li{background:var(--bg2);border-radius:8px;padding:8px 12px;font-size:15px;display:grid;grid-template-columns:1fr auto;gap:8px}
.notes li a{color:var(--fly);cursor:pointer;text-decoration:underline}.notes li b{color:var(--fg2);font-weight:600;font-size:13px}
.dim{color:var(--fg2);font-size:15px}
@media(max-width:860px){.layout{grid-template-columns:1fr;gap:16px}.stage{position:sticky;top:8px;height:52vh;min-height:360px}.ch{min-height:auto;padding:20px 0 28px}}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="top">
<h1>Inside a fly brain that was taught by a video AI</h1>
<p class="lede">Every dot is one of the 6,935 real neurons, placed where its cell body sits in the fruit-fly brain (FlyWire). Colour shows how active the neuron is as the trained brain replays one walk. Drag to rotate, scroll the right column to step through the story, and leave your own notes on any step.</p>
</div>

<div class="layout">
<div class="stage" id="stage">
<canvas class="gl" id="gl"></canvas>
<div class="panel">
<div><div class="dim" style="margin-bottom:4px">brain's input (32×8 px)</div><canvas class="eye" id="eye" width="32" height="8"></canvas></div>
<div><div class="dim" style="margin-bottom:2px"><span class="ai">teacher latent</span> vs <span class="fly">brain's reconstruction</span> (first 48 of 256 numbers)</div><canvas class="lat" id="lat" width="600" height="44"></canvas></div>
<div class="gauge"><div class="dim" style="margin-bottom:4px">match <b id="matchv">–</b></div><div class="bar"><i id="matchbar"></i></div></div>
</div>
<div class="hud">
<div class="row"><span class="steplbl" id="steplbl">step 1 / 24</span><span id="actlbl" class="dim"></span></div>
<div class="row"><button id="play" aria-pressed="false">▶ play</button><button id="wArena" aria-pressed="true">test arena</button><button id="wNyc" aria-pressed="false">Times Square</button><button id="edgesBtn" aria-pressed="true">synapses</button></div>
<input type="range" id="step" min="0" max="23" value="0" aria-label="time step" style="grid-column:1/-1">
<div class="legend" style="grid-column:1/-1"><span style="--c:var(--vis)">Eyes (VIS) 2,330</span><span style="--c:var(--cx)">Compass &amp; map (CX) 2,875</span><span style="--c:var(--mb)">Memory (MB) 427</span><span style="--c:var(--out)">Motor (OUT) 1,303</span><span class="dim">· brightness = activity</span></div>
</div>
</div>

<div class="chapters" id="chapters">
<section class="ch" data-focus="all" data-step="0"><div class="k">Chapter 1 · The brain</div><h2>6,935 real neurons, in their real places.</h2>
<p>Each dot is one neuron from the FlyWire map of an adult fruit-fly brain, drawn at the position of its cell body. We use four groups: the <b style="color:var(--vis)">visual projection neurons</b> that carry what the eyes see, the <b style="color:var(--cx)">central complex</b> (the fly's compass and map), the <b style="color:var(--mb)">mushroom-body output</b> (memory), and the <b style="color:var(--out)">descending neurons</b> that drive movement.</p>
<p>The thin lines are a sample of the 91,663 synapses between them: blue lines excite, red lines inhibit. In our network a connection may exist <em>only</em> where one of these synapses exists.</p>
<div class="note">Drag the brain to rotate it. Left and right hemispheres are both present; the middle is the central complex.</div></section>

<section class="ch" data-focus="VIS" data-step="2"><div class="k">Chapter 2 · What goes in</div><h2>Only the eye neurons receive the picture.</h2>
<p>The brain sees a 32×8 pixel strip of the horizon (shown under the 3D view). That strip is fed into the visual neurons only. Movement commands ("turn left", "forward") enter the compass and motor groups. Nothing else is told anything.</p>
<p>From this the brain must produce the <span class="ai">teacher's</span> description of the moment: the 256-number latent that V-JEPA 2.1, a large video model, computes from the full-resolution video. The bar strip compares the teacher's numbers (orange) with the brain's reconstruction (teal). The match gauge is their cosine similarity.</p></section>

<section class="ch" data-focus="all" data-step="5"><div class="k">Chapter 3 · Distillation</div><h2>Training means copying the teacher's numbers.</h2>
<p>This is distillation: a large model's output becomes a small model's target. We never ask the fly brain to classify or to label anything. We only ask it to <em>agree with V-JEPA 2.1</em> about what the scene is, step after step.</p>
<p>Only the strengths of the existing synapses are learned (plus the input and readout weights). The wiring diagram itself is frozen. So whatever this brain learns, it learns inside the fly's own circuit.</p>
<div class="note">Press play. Watch the activity move through the groups while the match stays around 85–90 % with the eyes open.</div></section>

<section class="ch" data-focus="all" data-step="8"><div class="k">Chapter 4 · Lights off</div><h2>Steps 9 to 16: the picture is removed.</h2>
<p>From here the eye neurons get nothing. The brain only knows the move it just made. To keep matching the teacher it has to <span class="darkw">imagine</span> how the view changes as it turns and steps.</p>
<p>Notice two things in the 3D view: the visual group goes quiet, and the central complex keeps working. The match drops but does not collapse.</p></section>

<section class="ch" data-focus="CX" data-step="12"><div class="k">Chapter 5 · A compass appears</div><h2>The central complex keeps track of direction in the dark.</h2>
<p>We never told the network which way it faces. Yet from its central-complex activity a simple linear readout recovers the heading to within about 10° in the test arena, even during the blind steps. The fully-connected network and a standard GRU lose heading in the dark (36° to 45° error).</p>
<p>This is the fly's real job for these neurons, re-appearing from the task alone. One caveat: it appears across the population, not specifically in the 51 "EPG" compass cells.</p></section>

<section class="ch" data-focus="all" data-step="16"><div class="k">Chapter 6 · Lights on</div><h2>The picture returns and the match recovers.</h2>
<p>As soon as the eyes are back the match climbs. In the synthetic arena the brain recovers almost fully. In Times Square (switch worlds above) the whole walk is harder: narrow 110° view, real clutter, far less training data, so the match is lower throughout and can fall to zero late in the walk.</p></section>

<section class="ch" data-focus="all" data-step="20"><div class="k">Chapter 7 · Why the wiring matters</div><h2>Same neurons, scrambled wires, worse brain.</h2>
<p>We trained identical networks with the real synapses, with the synapses scrambled (each neuron keeps its number of connections), with random synapses, and fully connected. On the error in the dark, the real wiring is best in both worlds: 0.63 vs 0.68 scrambled and 0.65 random in the arena; 0.89 vs 0.91 and 0.92 in Times Square. A standard GRU beats it in the arena (0.54) but not in Times Square (0.91).</p>
<p>So: the real wiring is a genuinely useful prior for learning a world model, but not yet better than every unconstrained network.</p></section>

<section class="ch" data-focus="all" data-step="23"><div class="k">Chapter 8 · Your notes</div><h2>Annotate what you see.</h2>
<p>Add a note at the current step and world. Notes are saved in this browser only. Use "copy all" to paste them into a document or message.</p>
<div class="notes" id="notes">
<textarea id="noteText" placeholder="e.g. step 12, arena: compass stays bright while VIS is dark"></textarea>
<div style="margin-top:8px"><button id="noteAdd">Add note at this step</button><button id="noteCopy" style="background:var(--bg2);color:var(--fg);border:1px solid var(--line)">Copy all</button><span class="dim" id="noteMsg"></span></div>
<ul id="noteList"></ul>
</div></section>
</div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const D = __DATA__;
(function(){
const N = D.pos.length, css = getComputedStyle(document.documentElement), col = n => new THREE.Color(css.getPropertyValue(n).trim());
const RC = {VIS:'--vis', CX:'--cx', MB:'--mb', OUT:'--out'};
let world = 'arena', step = 0, playing = false, focus = 'all', showEdges = true;

// ---------- three.js scene ----------
const canvas = document.getElementById('gl');
const renderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:true}); renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene(); const camera = new THREE.PerspectiveCamera(40, 1, 1, 5000); camera.position.set(0, 30, 720); camera.lookAt(0, 0, 0);
const root = new THREE.Group(); scene.add(root);
const pos = new Float32Array(N*3); for (let i=0;i<N;i++){ pos[3*i]=D.pos[i][0]; pos[3*i+1]=-D.pos[i][1]; pos[3*i+2]=D.pos[i][2]; }
const base = new Float32Array(N*3); for (let i=0;i<N;i++){ const c = col(RC[D.region[i]]); base[3*i]=c.r; base[3*i+1]=c.g; base[3*i+2]=c.b; }
const colors = new Float32Array(N*3), sizes = new Float32Array(N);
const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos,3)); geo.setAttribute('color', new THREE.BufferAttribute(colors,3)); geo.setAttribute('psize', new THREE.BufferAttribute(sizes,1));
const mat = new THREE.ShaderMaterial({transparent:true, depthWrite:false, vertexColors:true,
 uniforms:{scale:{value:1}}, vertexShader:`attribute float psize; varying vec3 vc; varying float va; void main(){ vc=color; va=psize; vec4 mv=modelViewMatrix*vec4(position,1.0); gl_PointSize = psize*300.0/-mv.z; gl_Position=projectionMatrix*mv; }`,
 fragmentShader:`varying vec3 vc; varying float va; void main(){ float d=length(gl_PointCoord-0.5); if(d>0.5) discard; float a=smoothstep(0.5,0.25,d); gl_FragColor=vec4(vc, a*min(1.0, va/2.0)); }`});
const points = new THREE.Points(geo, mat); root.add(points);
// edges
const E = Math.min(D.edges.length, 3500), ep = new Float32Array(E*6), ec = new Float32Array(E*6); const cE = new THREE.Color('#5a9cff'), cI = new THREE.Color('#ff7a8a');
for (let k=0;k<E;k++){ const [a,b]=D.edges[k]; ep.set([pos[3*a],pos[3*a+1],pos[3*a+2],pos[3*b],pos[3*b+1],pos[3*b+2]],6*k); const c = D.esign[k]>0?cE:cI; ec.set([c.r,c.g,c.b,c.r,c.g,c.b],6*k); }
const eg = new THREE.BufferGeometry(); eg.setAttribute('position', new THREE.BufferAttribute(ep,3)); eg.setAttribute('color', new THREE.BufferAttribute(ec,3));
const edges = new THREE.LineSegments(eg, new THREE.LineBasicMaterial({vertexColors:true, transparent:true, opacity:0.045})); root.add(edges);

// orbit by drag, slow auto-rotate
let rotY = 0.3, rotX = 0.25, dragging=false, lx=0, ly=0, auto=true;
canvas.addEventListener('pointerdown', e=>{dragging=true; auto=false; lx=e.clientX; ly=e.clientY; canvas.setPointerCapture(e.pointerId); canvas.style.cursor='grabbing';});
canvas.addEventListener('pointermove', e=>{ if(!dragging) return; rotY += (e.clientX-lx)*0.006; rotX += (e.clientY-ly)*0.006; rotX=Math.max(-1.3,Math.min(1.3,rotX)); lx=e.clientX; ly=e.clientY; });
canvas.addEventListener('pointerup', e=>{dragging=false; canvas.style.cursor='grab';});
canvas.addEventListener('wheel', e=>{ e.preventDefault(); camera.position.z = Math.max(300, Math.min(1500, camera.position.z + e.deltaY*0.8)); }, {passive:false});
function resize(){ const w=canvas.clientWidth, h=canvas.clientHeight; if(!w||!h) return; renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix(); }
new ResizeObserver(resize).observe(canvas); resize();

// ---------- state -> colors ----------
function applyStep(){
  const A = D[world].act[step], blind = step>=8 && step<16;
  for (let i=0;i<N;i++){
    const a = Math.abs(A[i])/127, inF = (focus==='all' || D.region[i]===focus);
    const dim = inF ? 1 : 0.18;
    const g = 0.25 + 0.95*a;                              // brightness = activity
    colors[3*i] = base[3*i]*g*dim; colors[3*i+1] = base[3*i+1]*g*dim; colors[3*i+2] = base[3*i+2]*g*dim;
    sizes[i] = (2.2 + 4.5*a) * (inF ? 1 : 0.7);
  }
  geo.attributes.color.needsUpdate = true; geo.attributes.psize.needsUpdate = true;
  edges.visible = showEdges;
  // HUD
  const sl = document.getElementById('steplbl'); sl.textContent = `step ${step+1} / 24` + (blind ? '  ·  LIGHTS OFF' : '  ·  eyes open'); sl.classList.toggle('dark', blind);
  const names = ['stay','step forward','turn left','turn right','step back']; document.getElementById('actlbl').textContent = step>0 ? 'last move: ' + names[D[world].actions[step-1]] : 'start';
  document.getElementById('step').value = step;
  const m = D[world].match[step]; document.getElementById('matchv').textContent = m + '%'; document.getElementById('matchbar').style.width = m + '%';
  // eye
  const ec = document.getElementById('eye').getContext('2d'), img = ec.createImageData(32,8), O = D[world].obs[step];
  for (let y=0;y<8;y++) for (let x=0;x<32;x++){ const p=(y*32+x)*4, v = blind ? [0,0,0] : O[y][x]; img.data[p]=v[0]; img.data[p+1]=v[1]; img.data[p+2]=v[2]; img.data[p+3]=255; }
  ec.putImageData(img,0,0);
  // latent bars
  const lc = document.getElementById('lat'), c2 = lc.getContext('2d'); c2.clearRect(0,0,600,44); const z=D[world].z[step], zh=D[world].zh[step]; const mid=22, sc=6;
  c2.strokeStyle = css.getPropertyValue('--line').trim(); c2.beginPath(); c2.moveTo(0,mid); c2.lineTo(600,mid); c2.stroke();
  for (let i=0;i<48;i++){ const x=i*12.5; c2.fillStyle=css.getPropertyValue('--ai').trim(); c2.fillRect(x, Math.min(mid, mid-z[i]*sc), 5, Math.abs(z[i]*sc)); c2.fillStyle=css.getPropertyValue('--fly').trim(); c2.fillRect(x+6, Math.min(mid, mid-zh[i]*sc), 5, Math.abs(zh[i]*sc)); }
}
function setStep(s){ step = Math.max(0, Math.min(23, s)); applyStep(); }
document.getElementById('step').addEventListener('input', e => { playing=false; document.getElementById('play').setAttribute('aria-pressed','false'); document.getElementById('play').textContent='▶ play'; setStep(+e.target.value); });
document.getElementById('play').addEventListener('click', e => { playing=!playing; e.currentTarget.setAttribute('aria-pressed', playing); e.currentTarget.textContent = playing ? '❚❚ pause' : '▶ play'; });
function setWorld(w){ world=w; document.getElementById('wArena').setAttribute('aria-pressed', w==='arena'); document.getElementById('wNyc').setAttribute('aria-pressed', w==='nyc'); applyStep(); }
document.getElementById('wArena').addEventListener('click', ()=>setWorld('arena')); document.getElementById('wNyc').addEventListener('click', ()=>setWorld('nyc'));
document.getElementById('edgesBtn').addEventListener('click', e=>{ showEdges=!showEdges; e.currentTarget.setAttribute('aria-pressed', showEdges); applyStep(); });
let last = 0; function loop(t){ requestAnimationFrame(loop); if (playing && t-last > 700){ last=t; setStep((step+1)%24); } if (auto && !dragging) rotY += 0.0015; root.rotation.set(rotX, rotY, 0); renderer.render(scene, camera); }
requestAnimationFrame(loop);

// ---------- scroll-driven chapters ----------
const chapters = [...document.querySelectorAll('.ch')];
const io = new IntersectionObserver(es => { es.forEach(en => { if (en.isIntersecting){ focus = en.target.dataset.focus; if(!playing) setStep(+en.target.dataset.step); else applyStep(); } }); }, {rootMargin:'-40% 0px -45% 0px'});
chapters.forEach(c => io.observe(c));

// ---------- notes (this browser only) ----------
const KEY='wm2cx-brain3d-notes'; let notes=[]; try{ notes = JSON.parse(localStorage.getItem(KEY)||'[]'); }catch(e){ notes=[]; }
function save(){ try{ localStorage.setItem(KEY, JSON.stringify(notes)); }catch(e){} }
function renderNotes(){ const ul=document.getElementById('noteList'); ul.innerHTML=''; notes.forEach((n,i)=>{ const li=document.createElement('li'); li.innerHTML=`<div><b>${n.world==='arena'?'test arena':'Times Square'} · step ${n.step+1}</b><br>${n.text.replace(/</g,'&lt;')}</div><div><a data-go="${i}">go</a> · <a data-del="${i}">delete</a></div>`; ul.appendChild(li); });
  ul.querySelectorAll('a[data-go]').forEach(a=>a.addEventListener('click',()=>{ const n=notes[+a.dataset.go]; setWorld(n.world); setStep(n.step); document.getElementById('stage').scrollIntoView({behavior:'smooth'}); }));
  ul.querySelectorAll('a[data-del]').forEach(a=>a.addEventListener('click',()=>{ notes.splice(+a.dataset.del,1); save(); renderNotes(); })); }
document.getElementById('noteAdd').addEventListener('click', ()=>{ const t=document.getElementById('noteText').value.trim(); if(!t) return; notes.push({world, step, text:t, at:new Date().toISOString()}); save(); renderNotes(); document.getElementById('noteText').value=''; document.getElementById('noteMsg').textContent='saved'; setTimeout(()=>document.getElementById('noteMsg').textContent='',1500); });
document.getElementById('noteCopy').addEventListener('click', ()=>{ const txt = notes.map(n=>`[${n.world==='arena'?'arena':'Times Square'} step ${n.step+1}] ${n.text}`).join('\n'); const msg=document.getElementById('noteMsg');
  if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(txt).then(()=>msg.textContent='copied').catch(()=>{ msg.textContent='select and copy:'; const ta=document.getElementById('noteText'); ta.value=txt; ta.select(); }); else { msg.textContent='select and copy:'; const ta=document.getElementById('noteText'); ta.value=txt; ta.select(); } });
renderNotes(); applyStep();
})();
</script>
'''
open("fly_brain_3d.html","w").write(html.replace("__DATA__", data)); import os; print(os.path.getsize("fly_brain_3d.html")//1024, "KB")
