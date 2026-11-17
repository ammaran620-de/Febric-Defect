/* Static-site API shim: lets the FastAPI dashboard run on GitHub Pages.
   - /api/* and /predict are answered in the browser (no server).
   - The detector runs locally with ONNX Runtime Web; images never leave the device.
   - Preprocessing mirrors Ultralytics (rect letterbox to a multiple of 32, pad 114, RGB, /255). */
(() => {
  const realFetch = window.fetch.bind(window);
  const BASE = new URL('.', document.baseURI).href;
  const NAMES = ['hole', 'objects', 'oil spot', 'thread error'];
  const S = 416, IOU = 0.5;  // Ultralytics predict default NMS IoU
  const st = {rows: [], thumbs: {}, demoMax: 0, nextId: 1, sess: null};
  const json = (o, s = 200) => new Response(JSON.stringify(o), {status: s, headers: {'Content-Type': 'application/json'}});

  st.ready = Promise.all(['data/history.json', 'data/summary.json', 'data/errors.json'].map(u => realFetch(BASE + u).then(r => r.json())))
    .then(([h, s, e]) => { st.rows = h; st.summary = s; st.errors = e; st.demoMax = Math.max(0, ...h.map(r => r.id)); st.nextId = st.demoMax + 1; });

  window.THUMB = id => st.thumbs[id] || (id <= st.demoMax ? BASE + 'demo/thumbs/' + id + '.jpg' : '');

  async function session() {
    if (!st.sess) {
      ort.env.wasm.wasmPaths = BASE + 'vendor/'; ort.env.wasm.numThreads = 1;
      st.sess = await ort.InferenceSession.create(BASE + 'model/best.onnx', {executionProviders: ['wasm']});
    }
    return st.sess;
  }
  const iou = (a, b) => {
    const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]), x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
    const i = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9);
  };
  async function detect(blob, conf) {
    const bmp = await createImageBitmap(blob); const w = bmp.width, h = bmp.height;
    const t0 = performance.now(); const sess = await session();
    const r = Math.min(S / w, S / h), nw = Math.round(w * r), nh = Math.round(h * r);
    const dw = (S - nw) % 32, dh = (S - nh) % 32;
    const l = Math.round(dw / 2 - 0.1), t = Math.round(dh / 2 - 0.1), rr = Math.round(dw / 2 + 0.1), bb = Math.round(dh / 2 + 0.1);
    const W = nw + l + rr, H = nh + t + bb;
    // exact OpenCV INTER_LINEAR (no antialiasing) so confidences match the Python pipeline
    const sc = document.createElement('canvas'); sc.width = w; sc.height = h;
    const sx = sc.getContext('2d', {willReadFrequently: true}); sx.drawImage(bmp, 0, 0); const sd = sx.getImageData(0, 0, w, h).data;
    const n = W * H, f = new Float32Array(3 * n).fill(114 / 255), fx = w / nw, fy = h / nh;
    for (let oy = 0; oy < nh; oy++) {
      let yy = (oy + 0.5) * fy - 0.5, y0 = Math.floor(yy), wy = yy - y0; if (y0 < 0) { y0 = 0; wy = 0; } let y1 = Math.min(y0 + 1, h - 1); if (y0 > h - 1) { y0 = h - 1; wy = 0; }
      for (let ox = 0; ox < nw; ox++) {
        let xx = (ox + 0.5) * fx - 0.5, x0 = Math.floor(xx), wx = xx - x0; if (x0 < 0) { x0 = 0; wx = 0; } let x1 = Math.min(x0 + 1, w - 1); if (x0 > w - 1) { x0 = w - 1; wx = 0; }
        const o = (oy + t) * W + ox + l;
        for (let c = 0; c < 3; c++) {
          const v = (sd[(y0 * w + x0) * 4 + c] * (1 - wx) + sd[(y0 * w + x1) * 4 + c] * wx) * (1 - wy) + (sd[(y1 * w + x0) * 4 + c] * (1 - wx) + sd[(y1 * w + x1) * 4 + c] * wx) * wy;
          f[c * n + o] = Math.round(v) / 255;
        }
      }
    }
    const out = await sess.run({[sess.inputNames[0]]: new ort.Tensor('float32', f, [1, 3, H, W])});
    const o = out[sess.outputNames[0]]; const N = o.dims[2], d = o.data, cand = [];
    for (let i = 0; i < N; i++) {
      let bs = 0, bc = 0;
      for (let c = 0; c < 4; c++) { const s = d[(4 + c) * N + i]; if (s > bs) { bs = s; bc = c; } }
      if (bs < conf) continue;
      const bx = d[i], by = d[N + i], bw = d[2 * N + i], bh = d[3 * N + i];
      const cl = (v, m) => Math.min(Math.max(v, 0), m);
      cand.push([cl((bx - bw / 2 - l) / r, w), cl((by - bh / 2 - t) / r, h), cl((bx + bw / 2 - l) / r, w), cl((by + bh / 2 - t) / r, h), bs, bc]);
    }
    cand.sort((a, b) => b[4] - a[4]); const keep = [];
    for (const b of cand) if (!keep.some(k => k[5] === b[5] && iou(b, k) > IOU)) keep.push(b);
    const ms = performance.now() - t0;
    return {bmp, w, h, ms, defects: keep.map(b => ({box: b.slice(0, 4).map(v => Math.round(v * 10) / 10), confidence: Math.round(b[4] * 1000) / 1000, class: NAMES[b[5]]}))};
  }
  function thumb(bmp, w, h, defects) {
    const s = 480 / w, c = document.createElement('canvas'); c.width = 480; c.height = Math.round(h * s);
    const x = c.getContext('2d'); x.drawImage(bmp, 0, 0, c.width, c.height); x.strokeStyle = '#c4321f'; x.lineWidth = 2;
    defects.forEach(d => x.strokeRect(d.box[0] * s, d.box[1] * s, (d.box[2] - d.box[0]) * s, (d.box[3] - d.box[1]) * s));
    return new Promise(res => c.toBlob(b => res(URL.createObjectURL(b)), 'image/jpeg', 0.85));
  }
  const strip = ({boxes, ...r}) => r;
  async function handle(path, init) {
    await st.ready; const u = new URL(path, 'http://x'); const p = u.pathname, q = u.searchParams;
    if (p === '/api/summary') return json(st.summary);
    if (p === '/api/errors') return json(st.errors);
    if (p === '/api/history') {
      let r = [...st.rows].sort((a, b) => b.id - a.id); const v = q.get('verdict'); if (v === 'PASS' || v === 'FAIL') r = r.filter(x => x.verdict === v);
      return json(r.slice(0, +(q.get('limit') || 200)).map(strip));
    }
    if (p === '/api/stats') {
      const rows = st.rows, n = rows.length, fail = rows.filter(r => r.verdict === 'FAIL').length, by = {}, src = {};
      rows.forEach(r => { r.boxes.forEach(d => by[d.class] = (by[d.class] || 0) + 1); src[r.source] = (src[r.source] || 0) + 1; });
      return json({inspections: n, failed: fail, passed: n - fail, defects: rows.reduce((a, r) => a + r.n_defects, 0),
        avg_ms: n ? Math.round(rows.reduce((a, r) => a + r.ms, 0) / n * 10) / 10 : 0, by_class: by, by_source: src, recent: rows.slice(-40).map(r => r.n_defects)});
    }
    if (p === '/api/inspect' || p === '/predict') {
      const file = init.body.get('file'); const conf = parseFloat(q.get('conf') || '0.25');
      const res = await detect(file, conf);
      if (p === '/predict') return json({width: res.w, height: res.h, defects: res.defects});
      const id = st.nextId++, ds = res.defects;
      st.rows.push({id, ts: new Date().toISOString().slice(0, 19) + '+00:00', source: q.get('source') || 'upload', filename: file.name || 'image',
        verdict: ds.length ? 'FAIL' : 'PASS', n_defects: ds.length, max_conf: Math.max(0, ...ds.map(d => d.confidence)), classes: [...new Set(ds.map(d => d.class))].sort().join(','),
        conf_used: conf, ms: Math.round(res.ms * 10) / 10, boxes: ds});
      st.thumbs[id] = await thumb(res.bmp, res.w, res.h, ds);
      return json({id, verdict: ds.length ? 'FAIL' : 'PASS', defects: ds, ms: Math.round(res.ms * 10) / 10, width: res.w, height: res.h});
    }
    return json({detail: 'Not found'}, 404);
  }
  window.fetch = (input, init) => {
    const url = typeof input === 'string' ? input : input.url;
    return (url.startsWith('/api/') || url.startsWith('/predict')) ? handle(url, init) : realFetch(input, init);
  };
  document.addEventListener('click', async e => {
    const a = e.target.closest('a[href="#csv"]'); if (!a) return; e.preventDefault(); await st.ready;
    const cols = ['id', 'ts', 'source', 'filename', 'verdict', 'n_defects', 'max_conf', 'classes', 'conf_used', 'ms'];
    const esc = v => { v = String(v ?? ''); return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; };
    const csv = [cols.join(',')].concat([...st.rows].sort((a, b) => a.id - b.id).map(r => cols.map(c => esc(r[c])).join(','))).join('\n');
    const l = document.createElement('a'); l.href = URL.createObjectURL(new Blob([csv], {type: 'text/csv'})); l.download = 'inspections.csv'; l.click();
  });
  window.__fabric = st;  // for tests
})();
