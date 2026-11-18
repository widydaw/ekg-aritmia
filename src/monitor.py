"""Monitor EKG bergerak (sweep) seperti monitor pasien, sebagai komponen HTML di Streamlit.

Skala kertas standar: 25 mm/s dan 10 mm/mV. Sinyal digambar dari kiri ke kanan, lalu
menimpa sapuan sebelumnya. Label kelas tiap detak muncul saat detak itu tergambar.
"""
from __future__ import annotations

import json

import numpy as np
import streamlit.components.v1 as components

CLASS_COLORS = {"N": "#2b7a76", "S": "#c17a07", "V": "#8d2e86", "F": "#6b7280", "Q": "#6b7280"}

_HTML = r"""
<div id="mon">
  <div class="bar">
    <button id="play" title="Jeda / putar">❚❚ Jeda</button>
    <div class="seg" id="spd"><button data-s="0.5">0,5×</button><button data-s="1" class="on">1×</button><button data-s="2">2×</button></div>
    <label class="chk"><input type="checkbox" id="beep"> Bunyi bip</label>
    <span class="sp"></span>
    <span class="ro"><small>Waktu</small><b id="tm">00:00</b></span>
    <span class="ro"><small>HR</small><b id="hr" class="hr">–</b><small>bpm</small></span>
    <span class="ro"><small>Detak</small><b id="bt">–</b></span>
  </div>
  <canvas id="cv"></canvas>
  <div class="foot">25 mm/s · 10 mm/mV · __TITLE__</div>
</div>
<style>
  html,body{margin:0;background:transparent;font-family:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;color:#1b2130}
  #mon{border:1px solid #dcc9cd;border-radius:14px;overflow:hidden;background:#fffdfc}
  .bar{display:flex;flex-wrap:wrap;align-items:center;gap:10px 16px;padding:10px 14px;border-bottom:1px solid #ecdfe1;background:#fbf7f6}
  button{font:inherit;font-size:13px;border:1px solid #dcc9cd;background:#fff;border-radius:8px;padding:5px 11px;cursor:pointer;color:#1b2130}
  button:hover{border-color:#b3243f;color:#b3243f}
  .seg{display:inline-flex;border:1px solid #dcc9cd;border-radius:8px;padding:2px;background:#fff}
  .seg button{border:0;padding:3px 9px}
  .seg button.on{background:#b3243f;color:#fff;border-radius:6px}
  .chk{font-size:13px;display:flex;gap:6px;align-items:center;color:#4b5366}
  .sp{flex:1}
  .ro{display:flex;align-items:baseline;gap:6px;font-variant-numeric:tabular-nums}
  .ro small{font-family:ui-monospace,Consolas,monospace;font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:#868c9c}
  .ro b{font-family:ui-monospace,Consolas,monospace;font-size:20px;font-weight:600;min-width:2ch}
  .hr{color:#b3243f}
  .chip{font-family:ui-monospace,Consolas,monospace;font-size:13px;font-weight:700;border:1.5px solid currentColor;border-radius:999px;padding:0 9px}
  canvas{display:block;width:100%;height:__CH__px}
  .foot{font-family:ui-monospace,Consolas,monospace;font-size:11px;color:#868c9c;padding:6px 14px;border-top:1px solid #ecdfe1}
</style>
<script>
(() => {
  const D = __DATA__;
  const fs = D.fs, y = D.y, pk = D.peaks, lb = D.labels, col = D.colors, dur = y.length / fs;
  const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let t = 0, speed = 1, playing = !reduce, last = performance.now(), audio = null, lastBeep = -1;
  const sorted = y.slice().sort((a, b) => a - b), base = sorted[Math.floor(sorted.length / 2)];
  const NAMES = {N: 'Normal', S: 'Supraventrikular prematur', V: 'Ventrikular prematur'};

  function size() {
    const dpr = Math.min(2, devicePixelRatio || 1), w = cv.clientWidth, h = cv.clientHeight;
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return {w, h};
  }
  let S = size();
  addEventListener('resize', () => { S = size(); });

  function beep(cls) {
    if (!audio) return;
    const o = audio.createOscillator(), g = audio.createGain();
    o.frequency.value = cls === 'V' ? 660 : cls === 'S' ? 880 : 1000;
    g.gain.setValueAtTime(0.0001, audio.currentTime);
    g.gain.exponentialRampToValueAtTime(0.15, audio.currentTime + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + 0.09);
    o.connect(g).connect(audio.destination); o.start(); o.stop(audio.currentTime + 0.1);
  }

  function draw() {
    const {w, h} = S;
    const win = 8;                                  // detik per sapuan
    const pxmm = w / (win * 25);                    // 25 mm/s
    const mid = h * 0.55, Y = v => mid - (v - base) * 10 * pxmm;   // 10 mm/mV
    ctx.fillStyle = '#fffdfc'; ctx.fillRect(0, 0, w, h);
    ctx.lineWidth = 1;
    for (let i = 0; i * pxmm <= w; i++) { ctx.strokeStyle = i % 5 ? 'rgba(214,84,108,.11)' : 'rgba(214,84,108,.32)'; const x = Math.round(i * pxmm) + .5; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke(); }
    for (let i = -80; i <= 80; i++) { const yy = mid + i * pxmm; if (yy < 0 || yy > h) continue; ctx.strokeStyle = i % 5 ? 'rgba(214,84,108,.11)' : 'rgba(214,84,108,.32)'; ctx.beginPath(); ctx.moveTo(0, Math.round(yy) + .5); ctx.lineTo(w, Math.round(yy) + .5); ctx.stroke(); }

    const cyc = Math.floor(t / win), head = t - cyc * win, gap = 0.2;
    const seg = (a, b, t0) => {
      ctx.beginPath(); let first = true;
      const i0 = Math.max(0, Math.floor((t0 + a) * fs)), i1 = Math.min(y.length - 1, Math.floor((t0 + b) * fs));
      const step = Math.max(1, Math.floor((i1 - i0) / (w * 2)));
      for (let i = i0; i <= i1; i += step) { const x = (i / fs - t0) / win * w, yy = Y(y[i]); first ? ctx.moveTo(x, yy) : ctx.lineTo(x, yy); first = false; }
      ctx.stroke();
      for (let k = 0; k < pk.length; k++) {                    // label kelas tiap detak
        const ts = pk[k] / fs; if (ts < t0 + a || ts > t0 + b) continue;
        const x = (ts - t0) / win * w, c = lb ? (col[lb[k]] || '#1b2130') : '#b3243f';
        ctx.fillStyle = c; ctx.beginPath(); ctx.arc(x, Y(y[pk[k]]), 3.5, 0, 7); ctx.fill();
        if (lb) { ctx.font = '700 14px ui-monospace,Consolas,monospace'; ctx.textAlign = 'center'; ctx.fillText(lb[k], x, 18); }
        ctx.strokeStyle = '#1b2130';
      }
    };
    ctx.strokeStyle = '#1b2130'; ctx.lineWidth = 1.8; ctx.lineJoin = 'round';
    if (cyc > 0 && head + gap < win) { ctx.globalAlpha = .55; seg(head + gap, win, (cyc - 1) * win); ctx.globalAlpha = 1; }
    ctx.lineWidth = 1.8; seg(0, head, cyc * win);
    const hi = Math.min(y.length - 1, Math.floor(t * fs));
    ctx.fillStyle = '#b3243f'; ctx.beginPath(); ctx.arc(head / win * w, Y(y[hi]), 4, 0, 7); ctx.fill();
    ctx.fillStyle = 'rgba(255,253,252,.9)'; ctx.fillRect(head / win * w + 4, 0, gap / win * w, h);

    // pembacaan
    let k = -1; while (k + 1 < pk.length && pk[k + 1] / fs <= t) k++;
    const abs = D.start + t, mm = Math.floor(abs / 60), ss = Math.floor(abs % 60);
    document.getElementById('tm').textContent = String(mm).padStart(2, '0') + ':' + String(ss).padStart(2, '0');
    if (k >= 1) document.getElementById('hr').textContent = Math.round(60 * fs / (pk[k] - pk[k - 1]));
    if (k >= 0 && lb) { const c = lb[k]; const el = document.getElementById('bt'); el.innerHTML = '<span class="chip" style="color:' + (col[c] || '#1b2130') + '" title="' + (NAMES[c] || c) + '">' + c + '</span>'; }
    else if (k >= 0) document.getElementById('bt').textContent = k + 1;
    if (k !== lastBeep) { if (k >= 0 && playing) beep(lb ? lb[k] : 'N'); lastBeep = k; }
  }

  function loop(now) {
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    if (playing) { t += dt * speed; if (t >= dur) { t = 0; lastBeep = -1; } }
    draw(); requestAnimationFrame(loop);
  }
  document.getElementById('play').onclick = e => { playing = !playing; e.target.textContent = playing ? '❚❚ Jeda' : '▶ Putar'; };
  if (!playing) document.getElementById('play').textContent = '▶ Putar';
  document.getElementById('spd').onclick = e => { const b = e.target.closest('button'); if (!b) return; speed = +b.dataset.s; document.querySelectorAll('#spd button').forEach(x => x.classList.toggle('on', x === b)); };
  document.getElementById('beep').onchange = e => { if (e.target.checked) { audio = audio || new (window.AudioContext || window.webkitAudioContext)(); audio.resume(); } else if (audio) { audio.suspend(); audio = null; } };
  requestAnimationFrame(loop);
})();
</script>
"""


def ecg_monitor(signal, fs: float, peaks=None, labels=None, start: float = 0.0, duration: float = 120.0,
                title: str = "", canvas_height: int = 280) -> None:
    """Tampilkan monitor bergerak untuk potongan `start` s.d. `start + duration` detik."""
    signal = np.asarray(signal, float)
    i0 = int(max(0, start * fs))
    i1 = int(min(len(signal), (start + duration) * fs))
    y = np.round(signal[i0:i1], 3)
    pk, lb = [], None
    if peaks is not None:
        peaks = np.asarray(peaks)
        m = (peaks >= i0) & (peaks < i1)
        pk = (peaks[m] - i0).astype(int).tolist()
        if labels is not None:
            lb = [str(v) for v in np.asarray(labels)[m]]
    data = {"fs": float(fs), "y": y.tolist(), "peaks": pk, "labels": lb, "colors": CLASS_COLORS, "start": i0 / fs}
    html = (_HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
            .replace("__CH__", str(canvas_height)).replace("__TITLE__", title.replace("<", "&lt;")))
    components.html(html, height=canvas_height + 100)
