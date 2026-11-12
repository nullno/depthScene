/**
 * 浏览器侧：采集 + 呈现
 *   1. getUserMedia 打开摄像头
 *   2. 每帧把画面画到 canvas（60fps，不受推理影响）
 *   3. 每 ~150ms 抓一帧 JPEG 丢给 POST /frame，拿回结构化结果
 *   4. 用最近一次结果在画面上叠加：障碍框、识别框、扇区条、深度缩略图
 */

const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

const LEVEL_COLOR = { safe: '#35e0c8', warn: '#ffb648', danger: '#ff5a6e' };
const DET_COLOR = '#8fa4ff';

const stage = $('stage');
const ctx = stage.getContext('2d');
const video = $('cam');
const radar = $('radar');
const rctx = radar.getContext('2d');

const sendCanvas = document.createElement('canvas');
const sendCtx = sendCanvas.getContext('2d');

let state = null;
let inflight = false;
let lastSend = 0;
let frameCount = 0;
let fps = 0;
let lastFpsAt = performance.now();

/** 水平翻转：同时作用于「显示」和「上传」，两边坐标才一致；默认开启 */
let mirrored = localStorage.getItem('ds-mirror') !== '0';
let inferRate = 0;
let lastRespAt = 0;

function applyMirror() {
  $('btnMirror').classList.toggle('on', mirrored);
  localStorage.setItem('ds-mirror', mirrored ? '1' : '0');
}

function drawSource(c, src, w, h) {
  c.save();
  if (mirrored) {
    c.translate(w, 0);
    c.scale(-1, 1);
  }
  c.drawImage(src, 0, 0, w, h);
  c.restore();
}

/* ---------------- 摄像头 ---------------- */

async function openCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('浏览器不支持 getUserMedia（需要 https 或 localhost）');
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
  });
  video.srcObject = stream;
  await video.play().catch(() => {});
  await new Promise((resolve) => {
    if (video.videoWidth) return resolve();
    video.addEventListener('loadedmetadata', resolve, { once: true });
  });

  stage.width = video.videoWidth || 640;
  stage.height = video.videoHeight || 480;

  // 上传宽 640：检测模型输入就是 640，再小就要把细节找回来，框会明显变糙
  const scale = Math.min(1, 640 / stage.width);
  sendCanvas.width = Math.round(stage.width * scale);
  sendCanvas.height = Math.round(stage.height * scale);

  $('stageHint').classList.add('hide');
  $('badgeMode').textContent = '推理中…';
}

function toast(text, ms = 6000) {
  const el = document.createElement('div');
  el.className = 'toast';
  el.textContent = text;
  $('toastWrap').appendChild(el);
  setTimeout(() => {
    el.style.transition = 'opacity .3s';
    el.style.opacity = '0';
    setTimeout(() => el.remove(), 320);
  }, ms);
}

/* ---------------- 发送帧 ---------------- */

function sendFrame() {
  if (inflight || !video.videoWidth || document.hidden) return;
  inflight = true;
  lastSend = performance.now();

  drawSource(sendCtx, video, sendCanvas.width, sendCanvas.height);
  sendCanvas.toBlob(
    async (blob) => {
      if (!blob) {
        inflight = false;
        return;
      }
      try {
        const res = await fetch('/frame', {
          method: 'POST',
          headers: { 'Content-Type': 'image/jpeg' },
          body: blob,
        });
        const data = await res.json();
        if (data && data.sectors) applyState(data);
      } catch (err) {
        $('badgeMode').textContent = '服务未连接';
      } finally {
        inflight = false;
      }
    },
    'image/jpeg',
    0.65
  );
}

function applyState(next) {
  state = next;

  const now = performance.now();
  if (lastRespAt) {
    const inst = 1000 / Math.max(1, now - lastRespAt);
    inferRate = inferRate ? inferRate * 0.6 + inst * 0.4 : inst;
  }
  lastRespAt = now;

  const size = next.depthInput ? ` · ${next.depthInput}` : '';
  $('badgeMode').textContent = `SERVER · ${(next.device || 'cpu').toUpperCase()}${size}`;
  $('hudDepth').textContent = next.depthMs ? `${Math.round(next.depthMs)}ms` : '--';
  $('hudDetect').textContent = next.detectMs ? `${Math.round(next.detectMs)}ms` : '--';
  $('hudRate').textContent = inferRate ? `${inferRate.toFixed(1)}/s` : '--';
  $('depthMeta').textContent = `近 ${(next.near ?? 0.4).toFixed(1)} · 远 ${(next.far ?? 6).toFixed(1)}m`;

  if (next.depth) $('depthImg').src = next.depth;

  drawRadar(next);
  updateClearance(next);
  updateAdvice(next);
  updateDetections(next);
}

/* ---------------- 画面绘制 ---------------- */

function drawChip(text, x, y, color, placed, W, fontSize, chipH) {
  const padX = Math.round(fontSize * 0.42);
  const prevDash = ctx.getLineDash();
  ctx.setLineDash([]);
  ctx.font = `600 ${fontSize}px ui-monospace, Consolas, monospace`;
  const chipW = ctx.measureText(text).width + padX * 2;

  let lx = clamp(x, 2, Math.max(2, W - chipW - 2));
  let ly = y - chipH - 3;
  if (ly < 2) ly = y + 3;

  for (const p of placed) {
    if (lx < p.x + p.w && lx + chipW > p.x && ly < p.y + p.h && ly + chipH > p.y) {
      ly = p.y + p.h + 2;
    }
  }
  placed.push({ x: lx, y: ly, w: chipW, h: chipH });

  ctx.fillStyle = 'rgba(4, 10, 16, 0.86)';
  ctx.fillRect(lx, ly, chipW, chipH);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.strokeRect(lx + 0.5, ly + 0.5, chipW - 1, chipH - 1);
  ctx.fillStyle = color;
  ctx.fillText(text, lx + padX, ly + chipH / 2 + 0.5);
  ctx.setLineDash(prevDash);
}

function drawStage() {
  const W = stage.width;
  const H = stage.height;

  drawSource(ctx, video, W, H);

  if (state) {
    const fontSize = Math.max(12, Math.round(H * 0.032));
    const chipH = fontSize + 9;
    const placed = [];
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';

    // 障碍连通域：虚线 + 距离
    for (const t of state.threats || []) {
      const x = t.x * W;
      const y = t.y * H;
      const near = typeof t.distance === 'number' && t.distance <= 1.0;
      const color = near ? '#ff5a6e' : '#ffb648';

      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(x, y, t.w * W, t.h * H);

      if (typeof t.distance === 'number') {
        drawChip(`${t.distance.toFixed(1)}m`, x, y, color, placed, W, fontSize, chipH);
      }
      ctx.setLineDash([]);
    }

    // 识别目标：实线 + 类别 + 距离
    for (const d of state.detections || []) {
      const x = d.x * W;
      const y = d.y * H;
      ctx.strokeStyle = DET_COLOR;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, d.w * W, d.h * H);

      const name = d.labelZh || d.label;
      const dist = typeof d.distance === 'number' ? ` ${d.distance.toFixed(1)}m` : '';
      drawChip(`${name}${dist}`, x, y, DET_COLOR, placed, W, fontSize, chipH);
    }

    const sectors = state.sectors || [];
    if (sectors.length) {
      const sw = W / sectors.length;
      for (let i = 0; i < sectors.length; i++) {
        ctx.fillStyle = LEVEL_COLOR[sectors[i].level] || LEVEL_COLOR.safe;
        ctx.globalAlpha = 0.2 + (1 - sectors[i].clearance) * 0.55;
        ctx.fillRect(i * sw + 1, H - 8, sw - 2, 6);
        ctx.globalAlpha = 1;
      }
    }

    ctx.strokeStyle = 'rgba(255,255,255,0.4)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(W / 2 - 9, H / 2);
    ctx.lineTo(W / 2 + 9, H / 2);
    ctx.moveTo(W / 2, H / 2 - 9);
    ctx.lineTo(W / 2, H / 2 + 9);
    ctx.stroke();
  }

  frameCount++;
  const now = performance.now();
  if (now - lastFpsAt >= 1000) {
    fps = (frameCount * 1000) / (now - lastFpsAt);
    frameCount = 0;
    lastFpsAt = now;
    $('hudFps').textContent = fps.toFixed(0);
  }

  requestAnimationFrame(drawStage);
}

/* ---------------- 侧栏 ---------------- */

function sizeCanvas(canvas) {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(120, Math.round(rect.width));
  const h = Math.max(80, Math.round(rect.height));
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  return { w: canvas.width, h: canvas.height };
}

function drawRadar(s) {
  const sectors = s.sectors || [];
  if (!sectors.length) return;

  const { w: W, h: H } = sizeCanvas(radar);
  const c = rctx;
  c.clearRect(0, 0, W, H);

  const cx = W / 2;
  const cy = H - 4;
  const R = Math.min(W / 2 - 10, H - 22);
  const far = s.far || 6;
  const scale = (d) => clamp((d / far) * R, 0, R);

  c.beginPath();
  c.arc(cx, cy, R, Math.PI, 2 * Math.PI);
  c.fillStyle = 'rgba(255,255,255,0.022)';
  c.fill();

  c.strokeStyle = 'rgba(255,255,255,0.07)';
  c.lineWidth = 1;
  for (let m = 1; m <= far; m++) {
    const r = scale(m);
    if (r < 6) continue;
    c.beginPath();
    c.arc(cx, cy, r, Math.PI, 2 * Math.PI);
    c.stroke();
  }
  for (let k = 0; k <= 6; k++) {
    const a = Math.PI + (k / 6) * Math.PI;
    c.beginPath();
    c.moveTo(cx, cy);
    c.lineTo(cx + Math.cos(a) * R, cy + Math.sin(a) * R);
    c.stroke();
  }

  const n = sectors.length;
  for (let i = 0; i < n; i++) {
    const sec = sectors[i];
    const a0 = Math.PI + (i / n) * Math.PI;
    const a1 = Math.PI + ((i + 1) / n) * Math.PI;
    const r = Math.max(2, scale(sec.distance));
    const color = LEVEL_COLOR[sec.level] || LEVEL_COLOR.safe;

    c.beginPath();
    c.moveTo(cx, cy);
    c.arc(cx, cy, r, a0, a1);
    c.closePath();
    c.globalAlpha = 0.14 + (1 - sec.clearance) * 0.42;
    c.fillStyle = color;
    c.fill();
    c.globalAlpha = 1;

    if (i % 2 === 0) {
      c.beginPath();
      c.arc(cx, cy, r, a0, a1);
      c.strokeStyle = color;
      c.lineWidth = 2;
      c.stroke();
    }
  }

  const rec = s.recommendation || {};
  const idx = typeof rec.targetIndex === 'number' ? rec.targetIndex : Math.floor(n / 2);
  const ang = Math.PI + ((idx + 0.5) / n) * Math.PI;
  const len = R * 0.72;
  const ex = cx + Math.cos(ang) * len;
  const ey = cy + Math.sin(ang) * len;

  c.strokeStyle = '#ffffff';
  c.lineWidth = 2.2;
  c.beginPath();
  c.moveTo(cx, cy);
  c.lineTo(ex, ey);
  c.stroke();

  c.save();
  c.translate(ex, ey);
  c.rotate(ang);
  c.beginPath();
  c.moveTo(9, 0);
  c.lineTo(-4, 5);
  c.lineTo(-4, -5);
  c.closePath();
  c.fillStyle = '#ffffff';
  c.fill();
  c.restore();

  c.beginPath();
  c.arc(cx, cy, 4.5, 0, Math.PI * 2);
  c.fillStyle = '#35e0c8';
  c.fill();

  $('radarMeta').textContent = `${n} 扇区 · 最远 ${far.toFixed(0)}m`;
}

let clearEls = [];

function buildClear3() {
  const box = $('clear3');
  box.innerHTML = '';
  clearEls = ['LEFT', 'CENTER', 'RIGHT'].map((label) => {
    const el = document.createElement('div');
    el.className = 'c3';
    el.innerHTML = `<b>--</b><small>${label}</small><i><span></span></i>`;
    box.appendChild(el);
    return { box: el, value: el.querySelector('b'), bar: el.querySelector('span') };
  });
}

function updateClearance(s) {
  const rec = s.recommendation;
  if (!rec) return;
  if (!clearEls.length) buildClear3();

  const values = [rec.clearance.left, rec.clearance.center, rec.clearance.right];
  const far = s.far || 6;

  clearEls.forEach((el, i) => {
    const v = values[i];
    el.value.textContent = Number.isFinite(v) ? v.toFixed(1) : '--';
    const level = v <= 1.0 ? 'danger' : v <= 2.6 ? 'warn' : '';
    el.box.className = `c3 ${level}`;
    el.bar.style.width = `${Math.round(clamp(v / far, 0, 1) * 100)}%`;
  });

  $('sectorMeta').textContent = `最近 ${s.nearest ? s.nearest.distance.toFixed(2) : '--'}m`;
}

function updateAdvice(s) {
  const rec = s.recommendation;
  if (!rec) return;
  $('advice').dataset.action = rec.action;
  $('adviceText').textContent = rec.text;
  $('adviceSub').textContent =
    `${rec.code} · 左 ${rec.clearance.left.toFixed(1)}m / 中 ${rec.clearance.center.toFixed(1)}m / 右 ${rec.clearance.right.toFixed(1)}m`;
}

function updateDetections(s) {
  const dets = s.detections || [];
  const classes = s.classes ? `${s.classes} 类` : '';
  const models = (s.models || []).map((m) => m.name).join(' + ');
  $('detMeta').textContent = String(dets.length);
  $('detNote').textContent = models
    ? `${models} · 可识别 ${classes}`
    : '类别随导出的词表变化';

  $('detList').innerHTML = dets.length
    ? dets.map((d) => {
        const name = d.labelZh ? `${d.labelZh}<small>${d.label}</small>` : d.label;
        const dist = typeof d.distance === 'number' ? `${d.distance.toFixed(1)}m · ` : '';
        return `<div class="det-row"><b>${name}</b><span>${dist}${Math.round(d.score * 100)}%</span></div>`;
      }).join('')
    : '<p class="card-note">暂无目标</p>';
}

/* ---------------- 启动 ---------------- */

async function boot() {
  try {
    await openCamera();
    $('stageHint').classList.add('hide');
    requestAnimationFrame(drawStage);
    setInterval(() => {
      if (performance.now() - lastSend > 150) sendFrame();
    }, 60);
  } catch (err) {
    const msg = String(err.message || err);
    $('stageHint').textContent = /NotAllowed|Permission/i.test(msg)
      ? '摄像头权限被拒绝，请在地址栏允许后刷新'
      : `摄像头不可用：${msg}`;
    $('badgeMode').textContent = '无摄像头';
    toast($('stageHint').textContent, 9000);
  }
}

$('btnMirror').addEventListener('click', () => {
  mirrored = !mirrored;
  applyMirror();
});

window.addEventListener('resize', () => {
  if (state) drawRadar(state);
});

buildClear3();
applyMirror();
boot();
