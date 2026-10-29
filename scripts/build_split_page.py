import json, base64, io, os
from PIL import Image
R="/Users/prajaksen/worldmodel2connectom"
D=json.load(open(f"{R}/results/brain3d.json"))
imgs=[]
for t in range(24):
    im=Image.open(f"{R}/results/hq/nyc_ep4_t{t:02d}.png").convert("RGB").resize((640,640),Image.LANCZOS); b=io.BytesIO(); im.save(b,"JPEG",quality=82); imgs.append("data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode())
slim={"region":D["region"],"pos":D["pos"],"edges":D["edges"][:3500],"esign":D["esign"][:3500],"nyc":{k:D["nyc"][k] for k in ("act","actions","heading","match","z","zh")},"frames":imgs}
html=r'''<title>World and Brain Side by Side</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Source+Sans+3:wght@400;600;700&display=swap">
<style>
:root{--bg:#0d0f14;--bg2:#171a22;--fg:#eceae2;--fg2:#a09e95;--line:#2c3039;--fly:#4fc3c7;--ai:#e6a054;--dark:#ffb347;--vis:#56b4e9;--cx:#f0a940;--mb:#d55181;--out:#9085e9;--display:"Fraunces",Georgia,serif;--body:"Source Sans 3","Helvetica Neue",Arial,sans-serif;color-scheme:dark}
@media (prefers-color-scheme: light){:root:not([data-theme="dark"]){--bg:#f6f4ee;--bg2:#e9e6dd;--fg:#1b1a18;--fg2:#5d5b54;--line:#d3d0c5;--fly:#0f6f73;--ai:#b8681a;--dark:#b8681a;--vis:#1f78b4;--cx:#c77400;--mb:#b4306a;--out:#5a4bd1;color-scheme:light}}
:root[data-theme="light"]{--bg:#f6f4ee;--bg2:#e9e6dd;--fg:#1b1a18;--fg2:#5d5b54;--line:#d3d0c5;--fly:#0f6f73;--ai:#b8681a;--dark:#b8681a;--vis:#1f78b4;--cx:#c77400;--mb:#b4306a;--out:#5a4bd1;color-scheme:light}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--body);font-size:17px;line-height:1.5;padding:0 max(16px,env(safe-area-inset-right,0px)) 48px max(16px,env(safe-area-inset-left,0px))}
.wrap{max-width:1400px;margin:0 auto}
h1{font-family:var(--display);font-size:clamp(26px,3.6vw,42px);font-weight:700;line-height:1.1;margin:28px 0 6px;text-wrap:balance}
.lede{color:var(--fg2);font-size:18px;max-width:80ch;margin:0 0 18px}
.panes{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.pane{background:var(--bg2);border-radius:14px;overflow:hidden;display:flex;flex-direction:column;min-height:0}
.pane h2{font-family:var(--display);font-size:17px;font-weight:600;margin:0;padding:10px 14px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;gap:8px}
.pane h2 small{color:var(--fg2);font-family:var(--body);font-weight:400;font-size:13px}
.world{position:relative;aspect-ratio:1;background:#000}
.world img{width:100%;height:100%;display:block;object-fit:cover;transition:filter .35s}
.world.blind img{filter:brightness(.18) saturate(.4)}
.world .ov{position:absolute;inset:0;display:none;align-items:center;justify-content:center;text-align:center;color:var(--dark);font-family:var(--display);font-size:clamp(22px,3vw,36px);font-weight:700;text-shadow:0 2px 12px #000}
.world.blind .ov{display:flex}
.world .tag{position:absolute;left:10px;bottom:10px;background:#000a;color:#fff;font-size:13px;padding:4px 9px;border-radius:6px}
canvas.gl{width:100%;aspect-ratio:1;display:block;touch-action:none;cursor:grab}
.legend{display:flex;flex-wrap:wrap;gap:10px 14px;font-size:13px;color:var(--fg2);padding:8px 14px;border-top:1px solid var(--line)}.legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;background:var(--c);vertical-align:-1px}
.ctrl{position:sticky;bottom:0;background:var(--bg);border-top:1px solid var(--line);padding:12px 0 8px;margin-top:14px;z-index:5}
.ctrl .row{display:grid;grid-template-columns:auto 1fr auto;gap:14px;align-items:center}
button{font:inherit;font-weight:600;background:var(--bg2);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:6px 14px;cursor:pointer}button[aria-pressed="true"]{border-color:var(--fly);color:var(--fly)}button:focus-visible{outline:2px solid var(--fly)}
input[type=range]{width:100%;accent-color:var(--fly);margin:0}
.steplbl{font-variant-numeric:tabular-nums;font-weight:700;white-space:nowrap}.steplbl.dark{color:var(--dark)}
.ticks{display:flex;justify-content:space-between;font-size:12px;color:var(--fg2);margin:2px 0 0;padding:0 2px}
.cap{font-size:17px;margin:10px 0 0;min-height:3.2em;max-width:100ch}
.below{display:grid;grid-template-columns:1fr 220px;gap:14px;margin-top:14px;align-items:end}
.lat{width:100%;height:70px;display:block}
.dim{color:var(--fg2);font-size:14px}.ai{color:var(--ai);font-weight:700}.fly{color:var(--fly);font-weight:700}
.gauge .bar{height:12px;border-radius:6px;background:var(--line);overflow:hidden}.gauge i{display:block;height:100%;background:var(--fly);width:0;transition:width .25s}
.story{margin-top:32px;max-width:80ch}.story h3{font-family:var(--display);font-size:24px;margin:22px 0 6px}.story p{margin:8px 0}
.story table{border-collapse:collapse;font-size:15px;margin:8px 0}.story td,.story th{border-bottom:1px solid var(--line);padding:5px 10px;text-align:left}
.story th{color:var(--fg2);font-weight:600}
@media(max-width:800px){.panes{grid-template-columns:1fr}.below{grid-template-columns:1fr}.ctrl .row{grid-template-columns:1fr auto}.ctrl .row input{grid-column:1/-1}}
@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
<div class="wrap">
<h1>What the world model sees, and what the fly brain does, at the same moment</h1>
<p class="lede">Left: the real-New-York view that the teacher (V-JEPA 2.1) watches. Right: the 6,935 real fly neurons of the trained student, lit by their activity. One slider drives both. Steps 9 to 16 are blind: the view is removed and the brain must imagine.</p>
<div class="panes">
 <div class="pane"><h2>World: Times Square digital twin <small>what the teacher sees · step <span id="sA">1</span></small></h2>
  <div class="world" id="world"><img id="frame" alt="street view from the agent's camera"><div class="ov">EYES CLOSED<br><span style="font-size:.6em;font-family:var(--body);font-weight:600">the brain imagines this view</span></div><div class="tag" id="tag"></div></div>
  <div class="legend"><span class="dim">Valdrada NYC twin · 768 px WebGL render · forward 110° camera · 4 m steps, 22.5° turns</span></div></div>
 <div class="pane"><h2>Brain: FlyWire connectome student <small>brightness = activity · step <span id="sB">1</span></small></h2>
  <canvas class="gl" id="gl"></canvas>
  <div class="legend"><span style="--c:var(--vis)">Eyes (VIS) 2,330</span><span style="--c:var(--cx)">Compass &amp; map (CX) 2,875</span><span style="--c:var(--mb)">Memory (MB) 427</span><span style="--c:var(--out)">Motor (OUT) 1,303</span><span class="dim">drag to rotate · wheel to zoom</span></div></div>
</div>
<div class="ctrl">
 <div class="row"><button id="play" aria-pressed="false">▶ play</button><input type="range" id="step" min="0" max="23" value="0" aria-label="time step"><span class="steplbl" id="steplbl">step 1 / 24 · eyes open</span></div>
 <div class="ticks"><span>1</span><span>eyes open</span><span style="color:var(--dark)">9 ← blind → 16</span><span>eyes open</span><span>24</span></div>
 <p class="cap" id="cap"></p>
</div>
<div class="below">
 <div><div class="dim"><span class="ai">teacher's summary</span> (V-JEPA 2.1 latent) vs <span class="fly">brain's reconstruction</span>, first 48 of 256 numbers</div><canvas class="lat" id="lat" width="1000" height="70"></canvas></div>
 <div class="gauge"><div class="dim">match (cosine) <b id="matchv">–</b></div><div class="bar"><i id="matchbar"></i></div><div class="dim" id="act" style="margin-top:6px"></div></div>
</div>
<div class="story">
<h3>How to read it</h3>
<p>The teacher is a large video model. For every frame it writes a 256-number description of the scene. The student is a recurrent network whose only recurrent connections are the 91,663 real synapses between these neurons, with their real excitatory or inhibitory signs. It receives a 64×64 glimpse into the visual neurons and the movement command into the compass and motor groups, and is trained to reproduce the teacher's numbers. Nothing else is given.</p>
<p>When the picture is removed (steps 9 to 16) the visual group goes quiet and the central complex keeps working. The match falls but the brain keeps predicting the teacher from its own motion. When the picture returns the match recovers.</p>
<h3>What this episode shows</h3>
<table><tr><th>phase</th><th>steps</th><th>match (this episode)</th></tr><tr><td>eyes open</td><td>1 to 8</td><td id="m1"></td></tr><tr><td>blind</td><td>9 to 16</td><td id="m2"></td></tr><tr><td>eyes open again</td><td>17 to 24</td><td id="m3"></td></tr></table>
<p class="dim">Times Square is hard: one seed, 120 training episodes, a narrow view and real clutter, so the match is lower than in the synthetic arena and can drop to zero late in a walk. Across the test set the real wiring still predicts best (blind error 0.885 vs 0.906 scrambled, 0.917 random, 0.907 GRU).</p>
</div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const D = __DATA__;
(function(){
const N=D.pos.length, css=getComputedStyle(document.documentElement), col=n=>new THREE.Color(css.getPropertyValue(n).trim()); const RC={VIS:'--vis',CX:'--cx',MB:'--mb',OUT:'--out'};
let step=0, playing=false;
const canvas=document.getElementById('gl'); const renderer=new THREE.WebGLRenderer({canvas,antialias:true,alpha:true}); renderer.setPixelRatio(Math.min(devicePixelRatio,2));
const scene=new THREE.Scene(); const camera=new THREE.PerspectiveCamera(40,1,1,5000); camera.position.set(0,30,760); camera.lookAt(0,0,0); const root=new THREE.Group(); scene.add(root);
const pos=new Float32Array(N*3); for(let i=0;i<N;i++){pos[3*i]=D.pos[i][0];pos[3*i+1]=-D.pos[i][1];pos[3*i+2]=D.pos[i][2];}
const base=new Float32Array(N*3); for(let i=0;i<N;i++){const c=col(RC[D.region[i]]);base[3*i]=c.r;base[3*i+1]=c.g;base[3*i+2]=c.b;}
const colors=new Float32Array(N*3), sizes=new Float32Array(N);
const geo=new THREE.BufferGeometry(); geo.setAttribute('position',new THREE.BufferAttribute(pos,3)); geo.setAttribute('color',new THREE.BufferAttribute(colors,3)); geo.setAttribute('psize',new THREE.BufferAttribute(sizes,1));
const mat=new THREE.ShaderMaterial({transparent:true,depthWrite:false,vertexColors:true,vertexShader:`attribute float psize; varying vec3 vc; varying float va; void main(){vc=color;va=psize;vec4 mv=modelViewMatrix*vec4(position,1.0);gl_PointSize=psize*460.0/-mv.z;gl_Position=projectionMatrix*mv;}`,fragmentShader:`varying vec3 vc; varying float va; void main(){float d=length(gl_PointCoord-0.5); if(d>0.5) discard; float a=smoothstep(0.5,0.25,d); gl_FragColor=vec4(vc,a*min(1.0,va/2.0));}`});
root.add(new THREE.Points(geo,mat));
const E=D.edges.length, ep=new Float32Array(E*6), ec=new Float32Array(E*6); const cE=new THREE.Color('#5a9cff'), cI=new THREE.Color('#ff7a8a');
for(let k=0;k<E;k++){const [a,b]=D.edges[k]; ep.set([pos[3*a],pos[3*a+1],pos[3*a+2],pos[3*b],pos[3*b+1],pos[3*b+2]],6*k); const c=D.esign[k]>0?cE:cI; ec.set([c.r,c.g,c.b,c.r,c.g,c.b],6*k);}
const eg=new THREE.BufferGeometry(); eg.setAttribute('position',new THREE.BufferAttribute(ep,3)); eg.setAttribute('color',new THREE.BufferAttribute(ec,3)); root.add(new THREE.LineSegments(eg,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:0.04})));
let rotY=0.3, rotX=0.25, dragging=false, lx=0, ly=0, auto=true;
canvas.addEventListener('pointerdown',e=>{dragging=true;auto=false;lx=e.clientX;ly=e.clientY;canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{if(!dragging)return; rotY+=(e.clientX-lx)*0.006; rotX=Math.max(-1.3,Math.min(1.3,rotX+(e.clientY-ly)*0.006)); lx=e.clientX; ly=e.clientY;});
canvas.addEventListener('pointerup',()=>{dragging=false;});
canvas.addEventListener('wheel',e=>{e.preventDefault(); camera.position.z=Math.max(300,Math.min(1500,camera.position.z+e.deltaY*0.8));},{passive:false});
function resize(){const w=canvas.clientWidth,h=canvas.clientHeight; if(!w||!h)return; renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix();} new ResizeObserver(resize).observe(canvas); resize();
const names=['stay','step forward','turn left','turn right','step back'];
function caption(t){ if(t===0) return 'Start. The teacher looks at this view and writes its 256-number summary. The brain gets a 64×64 glimpse and must produce the same numbers.';
 if(t<8) return 'Eyes open. The visual neurons (blue) carry the glimpse in; the central complex (orange) integrates the turns. The bars below show the brain tracking the teacher.';
 if(t===8) return 'The picture is switched off for 8 steps. The brain only knows the move it just made.';
 if(t<16) return 'Blind. The visual group goes quiet; the compass and memory groups keep working. The match decays slowly rather than collapsing.';
 if(t===16) return 'The picture is back. Fresh input reaches the visual neurons and the match climbs again.';
 return 'Eyes open. Same neurons, same synapses as a real fly; only the synapse strengths were learned, by copying the teacher.'; }
function apply(){ const A=D.nyc.act[step], blind=step>=8&&step<16;
 for(let i=0;i<N;i++){const a=Math.abs(A[i])/127, g=0.25+0.95*a; colors[3*i]=base[3*i]*g; colors[3*i+1]=base[3*i+1]*g; colors[3*i+2]=base[3*i+2]*g; sizes[i]=2.4+5.0*a;}
 geo.attributes.color.needsUpdate=true; geo.attributes.psize.needsUpdate=true;
 document.getElementById('frame').src=D.frames[step]; document.getElementById('world').classList.toggle('blind',blind);
 document.getElementById('tag').textContent=`heading ${Math.round(D.nyc.heading[step])}° · next move: ${names[D.nyc.actions[step]]}`;
 document.getElementById('sA').textContent=step+1; document.getElementById('sB').textContent=step+1;
 const sl=document.getElementById('steplbl'); sl.textContent=`step ${step+1} / 24 · ${blind?'EYES CLOSED':'eyes open'}`; sl.classList.toggle('dark',blind);
 document.getElementById('step').value=step; document.getElementById('cap').textContent=caption(step);
 const m=D.nyc.match[step]; document.getElementById('matchv').textContent=m+'%'; document.getElementById('matchbar').style.width=m+'%';
 document.getElementById('act').textContent=step>0?'last move: '+names[D.nyc.actions[step-1]]:'start of walk';
 const lc=document.getElementById('lat'), c2=lc.getContext('2d'); c2.clearRect(0,0,1000,70); const z=D.nyc.z[step], zh=D.nyc.zh[step], mid=35, sc=9;
 c2.strokeStyle=css.getPropertyValue('--line').trim(); c2.beginPath(); c2.moveTo(0,mid); c2.lineTo(1000,mid); c2.stroke();
 for(let i=0;i<48;i++){const x=i*20.8; c2.fillStyle=css.getPropertyValue('--ai').trim(); c2.fillRect(x,Math.min(mid,mid-z[i]*sc),8,Math.abs(z[i]*sc)); c2.fillStyle=css.getPropertyValue('--fly').trim(); c2.fillRect(x+9,Math.min(mid,mid-zh[i]*sc),8,Math.abs(zh[i]*sc));}
}
function setStep(s){step=Math.max(0,Math.min(23,s)); apply();}
document.getElementById('step').addEventListener('input',e=>{playing=false; const b=document.getElementById('play'); b.setAttribute('aria-pressed','false'); b.textContent='▶ play'; setStep(+e.target.value);});
document.getElementById('play').addEventListener('click',e=>{playing=!playing; e.currentTarget.setAttribute('aria-pressed',playing); e.currentTarget.textContent=playing?'❚❚ pause':'▶ play';});
document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')setStep(step+1); if(e.key==='ArrowLeft')setStep(step-1); if(e.key===' '&&e.target===document.body){e.preventDefault();document.getElementById('play').click();}});
let last=0; function loop(t){requestAnimationFrame(loop); if(playing&&t-last>1100){last=t; setStep((step+1)%24);} if(auto&&!dragging) rotY+=0.0015; root.rotation.set(rotX,rotY,0); renderer.render(scene,camera);} requestAnimationFrame(loop);
const mean=a=>Math.round(a.reduce((x,y)=>x+y,0)/a.length); const M=D.nyc.match; document.getElementById('m1').textContent=mean(M.slice(0,8))+'% average'; document.getElementById('m2').textContent=mean(M.slice(8,16))+'% average'; document.getElementById('m3').textContent=mean(M.slice(16))+'% average';
D.frames.forEach(s=>{const i=new Image(); i.src=s;}); apply();
})();
</script>
'''
open(f"{R}/docs/world_brain_split.html","w").write(html.replace("__DATA__",json.dumps(slim,separators=(",",":"))))
print(os.path.getsize(f"{R}/docs/world_brain_split.html")//1024,"KB")
