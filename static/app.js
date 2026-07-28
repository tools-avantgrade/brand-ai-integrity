/* Brand AI Integrity — frontend
   Estratto da index.html per consentire una CSP con script-src 'self'
   (nessuno script inline). Audit 25 luglio 2026, scheda SEC-04. */

const state={brandName:'',sector:'',email:'',userAnswers:{},questions:[],socialOptions:[],results:null};
const $=id=>document.getElementById(id);
const PRIVACY_VERSION='2026-07';

function showStep(n){[$('step1'),$('step2'),$('step3')].forEach((s,i)=>s.classList.toggle('hidden',i!==n-1));$('emailGate').classList.add('hidden');document.querySelector('.steps').style.display=n===3?'none':'flex';document.querySelector('.header').style.display=n===3?'none':'';document.querySelector('.footer').style.display=n===3?'none':'';['dot1','dot2','dot3'].forEach((id,i)=>{$(id).className='step-dot '+(i<n-1?'done':i===n-1?'active':'pending')});['line1','line2'].forEach((id,i)=>{$(id).className='step-line'+(i<n-1?' done':'')});window.scrollTo({top:0,behavior:'smooth'})}
function showGate(){[$('step1'),$('step2'),$('step3')].forEach(s=>s.classList.add('hidden'));$('emailGate').classList.remove('hidden');document.querySelector('.steps').style.display='none';document.querySelector('.header').style.display='none';document.querySelector('.footer').style.display='none';['dot1','dot2','dot3'].forEach((id,i)=>{$(id).className='step-dot '+(i<2?'done':'active')});['line1','line2'].forEach((id,i)=>{$(id).className='step-line done'});window.scrollTo({top:0,behavior:'smooth'})}

$('brandName').addEventListener('input',chk1);
$('sectorName').addEventListener('input',chk1);
function chk1(){state.brandName=$('brandName').value.trim();state.sector=$('sectorName').value.trim();$('btnStep1').disabled=!state.brandName||!state.sector}

$('btnStep1').addEventListener('click',async()=>{try{const r=await fetch('/api/questions');const d=await r.json();state.questions=d.questions;state.socialOptions=d.social_options}catch(e){alert('Errore caricamento domande.');return}buildQ();$('brandTitle').textContent=state.brandName;showStep(2)});

function buildQ(){const c=$('questionsContainer');c.innerHTML='';state.userAnswers={};state.questions.forEach((q,i)=>{const lb=q.label.replace('{BRAND_NAME}',state.brandName);const d=document.createElement('div');d.className='form-group';
if(q.prefill_from==='sector'){state.userAnswers[i]=state.sector;d.innerHTML=`<label>${lb}</label><input type="text" value="${state.sector}" disabled style="opacity:.6">`;c.appendChild(d);c.appendChild(Object.assign(document.createElement('div'),{className:'q-divider'}));return}
if(q.type==='checkbox'){state.userAnswers[i]='';d.innerHTML=`<label>${lb}</label><div class="pills" id="pills_${i}"></div>`;c.appendChild(d);const pd=$(`pills_${i}`);(q.options||state.socialOptions).forEach(opt=>{const p=document.createElement('div');p.className='pill';p.textContent=opt;p.addEventListener('click',()=>{p.classList.toggle('active');const a=[];pd.querySelectorAll('.pill.active').forEach(x=>a.push(x.textContent));state.userAnswers[i]=a.join(', ');chk2()});pd.appendChild(p)});c.appendChild(Object.assign(document.createElement('div'),{className:'q-divider'}));return}
const placeholder=q.id==='website'?`es. nomebrand.com`:`Scrivi qui la risposta corretta...`;
d.innerHTML=`<label>${lb}</label><textarea id="qa_${i}" placeholder="${placeholder}"></textarea>`;c.appendChild(d);c.appendChild(Object.assign(document.createElement('div'),{className:'q-divider'}));const ta=document.getElementById(`qa_${i}`);if(ta)ta.addEventListener('input',()=>{state.userAnswers[i]=ta.value.trim();chk2()})})}

function chk2(){let v=true;state.questions.forEach((q,i)=>{const val=state.userAnswers[i];if(!val||(typeof val==='string'&&!val.trim()))v=false});$('btnAnalyze').disabled=!v}

$('btnBack1').addEventListener('click',()=>showStep(1));
$('btnAnalyze').addEventListener('click',()=>runAnalysis());

const QT={sector:'il settore',target:'il pubblico target',locations:'le sedi',social:'i canali social',website:'il sito web'};

async function runAnalysis(){$('loadingOverlay').classList.add('visible');$('loadingPhase').textContent='Preparazione analisi...';$('loadingSub').textContent='Connessione alle AI';$('orbRing').style.strokeDashoffset='439.8';$('orbPct').innerHTML='0<span class="orb-pct-sym">%</span>';
const body={brand_name:state.brandName,sector:state.sector,user_answers:{}};for(const[k,v]of Object.entries(state.userAnswers))body.user_answers[String(k)]=String(v);
try{const resp=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
/* Gestione del limite di richieste lato server */
if(resp.status===429){$('loadingOverlay').classList.remove('visible');const d=await resp.json().catch(()=>({}));alert(d.error||'Troppe analisi dallo stesso indirizzo. Riprova più tardi.');return}
if(!resp.ok){$('loadingOverlay').classList.remove('visible');alert('Errore del server. Riprova.');return}
const reader=resp.body.getReader();const dec=new TextDecoder();let buf='',et='',ed='';
while(true){const{done,value}=await reader.read();if(done)break;buf+=dec.decode(value,{stream:true});const lines=buf.split('\n');buf=lines.pop();
for(const l of lines){if(l.startsWith('event: ')){et=l.slice(7).trim();ed=''}else if(l.startsWith('data: ')){ed=l.slice(6);if(et&&ed){try{const d=JSON.parse(ed);if(et==='progress')hProg(d);else if(et==='complete')hComp(d)}catch(e){}}et='';ed=''}}}}catch(e){$('loadingOverlay').classList.remove('visible');alert('Errore. Riprova.')}}

function hProg(d){const pct=Math.min(100,Math.round((d.step/d.total)*100));const circ=439.8;$('orbRing').style.strokeDashoffset=circ-(circ*pct/100);$('orbPct').innerHTML=pct+'<span class="orb-pct-sym">%</span>';const qi=(d.qn||1)-1;const q=state.questions[qi];const topic=q?QT[q.id]||'il brand':'il brand';let phase='Elaborazione...';if(d.phase==='gemini')phase='Gemini verifica '+topic+' di '+state.brandName;else if(d.phase==='chatgpt')phase='ChatGPT verifica '+topic+' di '+state.brandName;else if(d.phase==='eval')phase='Valutazione: '+topic+' di '+state.brandName;else if(d.phase==='comment')phase='Analisi qualitativa in corso...';$('loadingPhase').textContent=phase;$('loadingSub').textContent=d.msg||''}

function hComp(d){state.results=d;$('loadingOverlay').classList.remove('visible');showGate()}

function animC(el,target,dur=1500){const st=performance.now();function tick(now){const t=Math.min(1,(now-st)/dur);const ease=1-Math.pow(1-t,3);el.textContent=Math.round(target*ease);if(t<1)requestAnimationFrame(tick)}requestAnimationFrame(tick)}
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML}
function md(s){let h=esc(s);h=h.replace(/\[([^\]]+)\]\(([^)]+)\)/g,'<a href="$2" target="_blank" rel="noopener" style="color:var(--orange);text-decoration:underline">$1</a>');h=h.replace(/\*\*([^*]+)\*\*/g,'<strong style="color:var(--text)">$1</strong>');h=h.replace(/(?:^|\n)[*\-]\s+/g,function(m){return m.replace(/[*\-]\s+/,'<br>• ')});h=h.replace(/\*([^*]+)\*/g,'<em>$1</em>');h=h.replace(/\n/g,'<br>');return h}
function cfs(s){return s>=80?'#4CAF50':s>=60?'#FF9800':'#F44336'}

function renderResults(){const r=state.results,s=r.summary,sc=s.integrity_score,gs=s.ai_scores.gemini||0,os=s.ai_scores.openai||0;
const hc=cfs(sc);$('scoreHero').style.background=`linear-gradient(135deg,${hc},${hc}dd)`;animC($('scoreValue'),sc);$('scoreJudgment').textContent=sc>=80?'ECCELLENTE':sc>=60?'BUONO':'SCARSO';
let msg,mb;if(sc>=80){msg='Ottimo! Le AI rappresentano il tuo brand in modo chiaro e affidabile! 😎';mb='rgba(76,175,80,.12)'}else if(sc>=60){msg='Buono, ma puoi fare di meglio! Alcune imprecisioni sono migliorabili.';mb='rgba(255,152,0,.12)'}else{msg='Le AI non rappresentano correttamente il tuo brand. Parliamone! 😭';mb='rgba(244,67,54,.12)'}
const me=$('scoreMessage');me.style.background=mb;me.style.border=`1px solid ${hc}40`;me.style.color=hc;me.textContent=msg;
setTimeout(()=>animC($('geminiScore'),gs),300);setTimeout(()=>animC($('chatgptScore'),os),500);
const dc=$('detailsContainer');dc.innerHTML='';Object.keys(r.eval_results).sort((a,b)=>parseInt(a)-parseInt(b)).forEach(idx=>{const res=r.eval_results[idx];const qi=parseInt(idx);if(qi>=state.questions.length)return;const q=state.questions[qi];const qt=q.label.replace('{BRAND_NAME}',state.brandName);const avg=res.average_score||0;const st=res.status||'incorrect';
const card=document.createElement('div');card.className='detail-card';const bc=st==='correct'?'var(--green)':st==='partial'?'var(--yellow)':'var(--red)';const bt=st==='correct'?'✅ Corretta':st==='partial'?'⚠️ Parziale':'❌ Da migliorare';
let bh=`<div class="answer-block ground-truth"><strong>✅ La tua risposta:</strong><br>${esc(String(state.userAnswers[qi]||''))}</div>`;
const aa=r.ai_answers[String(idx)]||{};
for(const[an,al,cls]of[['gemini','⚫ Gemini','gemini-block'],['openai','🟢 ChatGPT','chatgpt-block']]){if(aa[an]){const ar=res[an]||{};const asc=ar.score||0;const scl=asc>=.75?(an==='gemini'?'var(--gemini)':'var(--chatgpt)'):asc>=.5?'var(--yellow)':'var(--red)';const reasonHtml=ar.reason?`<div style="margin-top:12px;padding:10px 14px;background:rgba(255,255,255,.05);border-radius:8px;border-top:1px solid rgba(255,255,255,.08)"><span style="display:block;font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:${scl};margin-bottom:6px">💬 Motivazione del punteggio</span><span style="font-size:.88rem;color:var(--text);line-height:1.65">${md(ar.reason)}</span></div>`:'';bh+=`<div class="answer-block ${cls}" style="border-left-color:${scl}"><strong>${al}</strong> <span style="color:${scl};font-weight:700">(${asc.toFixed(2)})</span><br>${md(aa[an])}${reasonHtml}</div>`}}
/* Nessun onclick inline: l'apertura è gestita da event delegation (vedi in fondo) */
card.innerHTML=`<div class="detail-header"><span class="q-text">D${qi+1}: ${esc(qt.substring(0,55))}...</span><span class="status-badge" style="background:${bc}20;color:${bc}">${bt}</span><span class="chevron">▼</span></div><div class="detail-body"><div class="detail-content">${bh}</div></div>`;dc.appendChild(card)});
}

/* Apertura/chiusura delle schede di dettaglio */
$('detailsContainer').addEventListener('click',e=>{const h=e.target.closest('.detail-header');if(h&&h.parentElement)h.parentElement.classList.toggle('open')});

function gp(){return{brand_name:state.brandName,sector:state.sector,summary:state.results.summary,eval_results:state.results.eval_results,user_answers:Object.fromEntries(Object.entries(state.userAnswers).map(([k,v])=>[String(k),String(v)])),ai_answers:state.results.ai_answers,qualitative_comment:state.results.qualitative_comment||''}}

/* FIX SEC-04: il pulsante resta disabilitato finché tutti i campi sono
   compilati E la casella del consenso è spuntata dall'utente. */
function chkGate(){
  const em=$('gateEmail').value.trim();
  const ok=$('gateNome').value.trim()&&$('gateCognome').value.trim()&&$('gateAzienda').value.trim()
    &&/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(em)&&$('gatePrivacy').checked;
  $('btnGate').disabled=!ok;
}
['gateNome','gateCognome','gateAzienda','gateEmail'].forEach(id=>$(id).addEventListener('input',chkGate));
$('gatePrivacy').addEventListener('change',chkGate);
chkGate();

$('btnGate').addEventListener('click',async()=>{
  const nm=$('gateNome').value.trim();const cn=$('gateCognome').value.trim();
  const az=$('gateAzienda').value.trim();const em=$('gateEmail').value.trim();
  if(!$('gatePrivacy').checked){$('gateStatus').innerHTML='<span style="color:var(--yellow)">Accetta il consenso per procedere</span>';return}
  state.email=em;state.nome=nm;state.cognome=cn;state.azienda=az;
  $('btnGate').disabled=true;
  $('gateStatus').innerHTML='<span style="color:var(--text2)">Invio report in corso...</span>';
  /* FIX SEC-04: prova del consenso inviata al server (data, ora, testo, versione) */
  const consentText=document.querySelector('label[for="gatePrivacy"]').innerText.trim();
  fetch('/api/send-email',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({email:em,nome:nm,cognome:cn,azienda:az,
      privacy_consent:true,
      consent_ts:new Date().toISOString(),
      consent_text:consentText,
      privacy_version:PRIVACY_VERSION,
      ...gp()})})
    .then(r=>r.json())
    .then(d=>{if(!d.success)console.error('Email error:',d.message)})
    .catch(e=>console.error('Email fetch error:',e));
  renderResults();showStep(3);
});

$('btnDownloadPdf').addEventListener('click',async()=>{$('btnDownloadPdf').disabled=true;$('pdfStatus').textContent='Generazione PDF in corso...';try{const r=await fetch('/api/download-pdf',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email:state.email,privacy_consent:true,privacy_version:PRIVACY_VERSION,...gp()})});const b=await r.blob();const u=URL.createObjectURL(b);const a=document.createElement('a');a.href=u;a.download=`Brand_AI_Integrity_${state.brandName}.pdf`;document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(u);$('pdfStatus').textContent=''}catch(e){$('pdfStatus').textContent='Errore nella generazione del PDF'}$('btnDownloadPdf').disabled=false});

showStep(1);
