// Interactive viewer for deeperfly keypoint locations on the NeuroMechFly model.
//
// MuJoCo (compiled to WebAssembly by Google DeepMind, vendored under
// vendor/mujoco/) loads the flattened MJCF emitted by
// scripts/build_keypoint_viewer_assets.py and does the forward kinematics; we
// render the body meshes and the keypoint stick-and-ball overlay with Three.js.
// Joint sliders write directly into `data.qpos`; nothing is simulated — each
// change is a single `mj_forward` (kinematics only), so the overlay, read from
// the solved body frames, stays locked to the mesh.

import * as THREE from 'three';
import { OrbitControls } from './vendor/three/OrbitControls.js';
import loadMujoco from './vendor/mujoco/mujoco.js';

const ASSETS = './assets';
const overlayEl = document.getElementById('overlay');
const overlayMsg = document.getElementById('overlay-msg');

const fail = (msg, err) => {
  console.error(err || msg);
  overlayEl.classList.remove('hidden');
  overlayEl.innerHTML = `<div class="err"><strong>Could not load the viewer.</strong>` +
    `<br>${msg}${err ? `<br><br><code>${String(err)}</code>` : ''}</div>`;
};

main().catch((e) => fail('Unexpected error while starting up.', e));

async function main() {
  setupTheme();
  overlayMsg.textContent = 'Loading MuJoCo (WebAssembly)…';
  const mj = await loadMujoco();

  overlayMsg.textContent = 'Fetching the fly model…';
  const [xmlText, pose, keypoints, colors] = await Promise.all([
    fetch(`${ASSETS}/model/fly.xml`).then((r) => r.text()),
    fetch(`${ASSETS}/pose.json`).then((r) => r.json()),
    fetch(`${ASSETS}/keypoints.json`).then((r) => r.json()),
    fetch(`${ASSETS}/colors.json`).then((r) => r.json()),
  ]);

  // Write the MJCF and every mesh it references into MuJoCo's virtual filesystem.
  const meshFiles = [...new Set(
    [...xmlText.matchAll(/<mesh[^>]*\bfile="([^"]+)"/g)].map((m) => m[1]))];
  try { mj.FS.mkdir('/work'); } catch (_) { /* already exists */ }
  mj.FS.writeFile('/work/fly.xml', xmlText);
  overlayMsg.textContent = `Loading ${meshFiles.length} meshes…`;
  await Promise.all(meshFiles.map(async (f) => {
    const buf = new Uint8Array(await fetch(`${ASSETS}/model/${f}`).then((r) => r.arrayBuffer()));
    mj.FS.writeFile(`/work/${f}`, buf);
  }));

  overlayMsg.textContent = 'Compiling the model…';
  const model = loadModel(mj, '/work/fly.xml');
  const data = new mj.MjData(model);

  // Initialise to the neutral (resting) pose.
  const qpos = pose.neutral_qpos.slice();
  applyQpos(mj, model, data, qpos);

  buildScene(mj, model, data, qpos, pose, keypoints, colors);
  overlayEl.classList.add('hidden');
}

const GRAY = new THREE.Color(0.78, 0.78, 0.80);
const INITIAL_OPACITY = 0.82;

// The panel defaults to dark. When embedded in the MkDocs Material docs (same
// origin), mirror the site's light/dark palette and follow its toggle live;
// when opened standalone (or cross-origin), stay on the dark default.
function setupTheme() {
  const root = document.documentElement;
  const apply = (dark) => root.setAttribute('data-theme', dark ? 'dark' : 'light');
  apply(true);
  try {
    const pbody = window.parent !== window ? window.parent.document.body : null;
    if (!pbody) return;
    const sync = () => apply(pbody.getAttribute('data-md-color-scheme') === 'slate');
    sync();
    new MutationObserver(sync).observe(pbody,
      { attributes: true, attributeFilter: ['data-md-color-scheme'] });
  } catch (_) { /* cross-origin / standalone: keep the dark default */ }
}

// Camera presets mirroring deeperfly's orbit rig (see cameras.py /
// default_config.toml [cameras.*]): the camera sits at
// target + distance * [cos(el)cos(az), cos(el)sin(az), sin(el)] and looks back
// at the target, world z up. The seven side views reuse the rig's azimuths
// (right hind .. left hind through the front); H/B/T add the hind/bottom/top
// poles. Bottom and top look along z, so they carry a non-z `up` to stay
// well-defined.
const VIEW_PRESETS = {
  rh: { az: -120, el: 0 },
  rm: { az: -90, el: 0 },
  rf: { az: -45, el: 0 },
  f: { az: 0, el: 0 },
  lf: { az: 45, el: 0 },
  lm: { az: 90, el: 0 },
  lh: { az: 120, el: 0 },
  h: { az: 180, el: 0 },
  b: { az: 0, el: -90, up: [1, 0, 0] },
  t: { az: 0, el: 90, up: [1, 0, 0] },
};

function loadModel(mj, path) {
  // The high-level binding name has shifted across builds; try the known spellings.
  if (typeof mj.mj_loadXML === 'function') return mj.mj_loadXML(path);
  if (mj.MjModel && typeof mj.MjModel.from_xml_path === 'function')
    return mj.MjModel.from_xml_path(path);
  throw new Error('no XML model-loading entry point found in the MuJoCo build');
}

function applyQpos(mj, model, data, qpos) {
  const q = data.qpos; // live view into WASM heap
  for (let i = 0; i < qpos.length; i++) q[i] = qpos[i];
  mj.mj_forward(model, data);
}

function buildScene(mj, model, data, qpos, pose, keypoints, colors) {
  const stage = document.getElementById('stage');
  THREE.Object3D.DEFAULT_UP.set(0, 0, 1); // MuJoCo is z-up

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  stage.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
  camera.up.set(0, 0, 1);

  // Centre the view on the fly (a few mm across) and frame it nicely.
  const center = new THREE.Vector3(0, 0, 0.9);
  camera.position.set(5.5, -5.5, 3.5);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(center);
  controls.minDistance = 1.5;
  controls.maxDistance = 30;

  scene.add(new THREE.AmbientLight(0xffffff, 0.65));
  const key = new THREE.DirectionalLight(0xffffff, 0.9); key.position.set(4, -6, 8);
  const fill = new THREE.DirectionalLight(0xffffff, 0.4); fill.position.set(-6, 4, 2);
  scene.add(key, fill);

  // On-demand rendering: redraw only when the pose, view or appearance changes,
  // so an idle page uses no CPU (and headless capture can settle). `dirty` also
  // forces a kinematics + overlay refresh on the next frame.
  let dirty = true, frameQueued = false;
  const requestRender = () => {
    if (!frameQueued) { frameQueued = true; requestAnimationFrame(frame); }
  };
  const markDirty = () => { dirty = true; requestRender(); };

  bodyIds(model, keypoints); // resolve keypoint body names -> ids once
  const meshGroup = buildMeshes(model, colors);
  const overlay = buildOverlay(keypoints);
  scene.add(meshGroup, overlay.group);

  buildSliders(pose, keypoints, mj, model, data, qpos, markDirty);

  document.getElementById('toggle-colors').addEventListener('change', (e) => {
    for (const { mesh, flyColor } of meshGroup.userData.items)
      mesh.material.color.copy(e.target.checked ? flyColor : GRAY);
    requestRender();
  });
  document.getElementById('toggle-ontop').addEventListener('change', (e) => {
    overlay.setOnTop(e.target.checked); requestRender();
  });
  document.getElementById('toggle-wings').addEventListener('change', (e) => {
    for (const it of meshGroup.userData.items) if (it.isWing) it.mesh.visible = e.target.checked;
    requestRender();
  });
  document.getElementById('opacity').addEventListener('input', (e) => {
    const o = parseFloat(e.target.value);
    meshGroup.visible = o > 0;
    for (const { mesh } of meshGroup.userData.items) mesh.material.opacity = o;
    requestRender();
  });
  document.getElementById('dot-size').addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    if (isFinite(v)) { overlay.setDotSize(v); requestRender(); }
  });
  document.getElementById('line-width').addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    if (isFinite(v)) { overlay.setLineWidth(v); requestRender(); }
  });
  document.getElementById('node-opacity').addEventListener('input', (e) => {
    overlay.setNodeOpacity(parseFloat(e.target.value)); requestRender();
  });
  document.getElementById('edge-opacity').addEventListener('input', (e) => {
    overlay.setEdgeOpacity(parseFloat(e.target.value)); requestRender();
  });
  document.getElementById('toggle-combine-abdomen').addEventListener('change', (e) => {
    overlay.setCombined(e.target.checked); requestRender();
  });
  controls.addEventListener('change', requestRender); // orbit / zoom / pan
  wireViewPresets(camera, controls, requestRender);

  function resize() {
    const w = stage.clientWidth, h = stage.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(1, h);
    camera.updateProjectionMatrix();
    requestRender();
  }
  window.addEventListener('resize', resize);
  resize();

  const m4 = new THREE.Matrix4();
  function syncFromMujoco() {
    mj.mj_forward(model, data);
    const gx = data.geom_xpos, gm = data.geom_xmat;
    for (const { mesh, g } of meshGroup.userData.items) {
      m4.set(gm[9*g+0], gm[9*g+1], gm[9*g+2], gx[3*g+0],
             gm[9*g+3], gm[9*g+4], gm[9*g+5], gx[3*g+1],
             gm[9*g+6], gm[9*g+7], gm[9*g+8], gx[3*g+2],
             0, 0, 0, 1);
      mesh.matrix.copy(m4);
    }
    const bx = data.xpos, bm = data.xmat;
    const pos = overlay.positions; // Float32Array, 3 per point
    overlay.points.forEach((p, i) => {
      const b = 3 * p.bodyId, r = 9 * p.bodyId, o = p.offset;
      const wx = bx[b]   + bm[r]   * o[0] + bm[r+1] * o[1] + bm[r+2] * o[2];
      const wy = bx[b+1] + bm[r+3] * o[0] + bm[r+4] * o[1] + bm[r+5] * o[2];
      const wz = bx[b+2] + bm[r+6] * o[0] + bm[r+7] * o[1] + bm[r+8] * o[2];
      p.ball.position.set(wx, wy, wz);
      pos[3*i] = wx; pos[3*i+1] = wy; pos[3*i+2] = wz;
    });
    overlay.syncBones();
  }

  function frame() {
    frameQueued = false;
    if (dirty) { syncFromMujoco(); dirty = false; }
    controls.update();
    renderer.render(scene, camera);
  }
  requestRender(); // initial draw
}

// Snap the camera to a preset orbit angle when a "View" button is clicked,
// preserving the current zoom (distance to target). OrbitControls fixes its
// orbit axis from camera.up at construction, so when a preset uses a non-z up
// (top/bottom) we refresh that axis before re-solving.
function wireViewPresets(camera, controls, requestRender) {
  const dir = new THREE.Vector3();
  const yUp = new THREE.Vector3(0, 1, 0);
  const setView = (v) => {
    const az = v.az * Math.PI / 180, el = v.el * Math.PI / 180;
    dir.set(Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el));
    const dist = camera.position.distanceTo(controls.target);
    camera.up.set(...(v.up || [0, 0, 1]));
    controls._quat.setFromUnitVectors(camera.up, yUp);
    controls._quatInverse.copy(controls._quat).invert();
    camera.position.copy(controls.target).addScaledVector(dir, dist);
    controls.update(); // re-points the camera at the target
    requestRender();
  };
  for (const [id, v] of Object.entries(VIEW_PRESETS)) {
    const btn = document.querySelector(`button[data-view="${id}"]`);
    if (btn) btn.addEventListener('click', () => setView(v));
  }
}

// Build one Three.js mesh per MuJoCo geom (mesh + capsule cover this model).
// Each mesh starts flat grey; the "Colors" toggle swaps in its flygym color.
function buildMeshes(model, colors) {
  const group = new THREE.Group();
  const items = [];
  for (let g = 0; g < model.ngeom; g++) {
    const geometry = geometryForGeom(model, g, model.geom_type[g]);
    if (!geometry) continue;
    const rgb = colors.geom_rgb[g] || [0.7, 0.7, 0.7];
    // colors.json holds sRGB values (from flygym); tag them as such so three
    // doesn't treat them as linear and wash the dark colors out.
    const flyColor = new THREE.Color().setRGB(rgb[0], rgb[1], rgb[2], THREE.SRGBColorSpace);
    const material = new THREE.MeshStandardMaterial({
      color: flyColor.clone(), roughness: 0.75, metalness: 0.0,
      transparent: true, opacity: INITIAL_OPACITY, side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.matrixAutoUpdate = false;
    const isWing = model.geom(g).name.includes('wing');
    mesh.visible = !isWing; // wings hidden by default
    group.add(mesh);
    items.push({ mesh, g, flyColor, isWing });
  }
  group.userData.items = items;
  return group;
}

function geometryForGeom(model, g, type) {
  // mjGEOM_PLANE=0, HFIELD=1, SPHERE=2, CAPSULE=3, ELLIPSOID=4, CYLINDER=5,
  // BOX=6, MESH=7 (stable MuJoCo ordering).
  const s = (k) => model.geom_size[g * 3 + k];
  if (model.geom_dataid[g] >= 0) return meshGeometry(model, model.geom_dataid[g]);
  switch (type) {
    case 2: return new THREE.SphereGeometry(s(0), 16, 12);
    case 3: { // capsule: size = [radius, half-length] along local z
      const geo = new THREE.CapsuleGeometry(s(0), 2 * s(1), 6, 12);
      geo.rotateX(Math.PI / 2); // Three capsule axis is +Y; MuJoCo's is +Z
      return geo;
    }
    case 4: { const geo = new THREE.SphereGeometry(1, 16, 12); geo.scale(s(0), s(1), s(2)); return geo; }
    case 5: { const geo = new THREE.CylinderGeometry(s(0), s(0), 2 * s(1), 16); geo.rotateX(Math.PI / 2); return geo; }
    case 6: return new THREE.BoxGeometry(2 * s(0), 2 * s(1), 2 * s(2));
    default: return null; // plane / hfield: skip
  }
}

function meshGeometry(model, dataid) {
  const va = model.mesh_vertadr[dataid], vn = model.mesh_vertnum[dataid];
  const fa = model.mesh_faceadr[dataid], fn = model.mesh_facenum[dataid];
  const allVerts = model.mesh_vert, allFaces = model.mesh_face; // cache: getters re-marshal
  const verts = new Float32Array(vn * 3);
  for (let i = 0; i < vn * 3; i++) verts[i] = allVerts[va * 3 + i];
  const index = new Uint32Array(fn * 3);
  for (let i = 0; i < fn * 3; i++) index[i] = allFaces[fa * 3 + i];
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  geo.setIndex(new THREE.BufferAttribute(index, 1));
  geo.computeVertexNormals();
  return geo;
}

// Stick-and-ball overlay drawn on top of the mesh (depthTest off) so it is always
// visible. Nodes are unit spheres scaled by `dotSize`; bones are unit cylinders
// scaled to span their endpoints with radius `lineWidth` (both in model mm, so a
// real adjustable thickness on every platform — unlike WebGL line width).
const DEFAULT_DOT = 0.0275, DEFAULT_LINE = 0.01;

function buildOverlay(keypoints) {
  const group = new THREE.Group();
  let dotSize = DEFAULT_DOT, lineWidth = DEFAULT_LINE, combined = false;

  // depthTest is off by default (overlay drawn "on top"); transparent:true puts
  // the overlay in the same render pass as the mesh, so when depthTest is turned
  // back on the higher renderOrder + the mesh's depth let the mesh occlude it.
  // Node (sphere) and edge (cylinder) materials are tracked apart so their
  // opacities can be set independently.
  const nodeMats = [], edgeMats = [];
  const sphereGeom = new THREE.SphereGeometry(1, 16, 12);
  const cylGeom = new THREE.CylinderGeometry(1, 1, 1, 8); // unit; axis +Y
  const newNode = (color) => {
    const mat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(color), depthTest: false, transparent: true });
    nodeMats.push(mat);
    const ball = new THREE.Mesh(sphereGeom, mat);
    ball.renderOrder = 2; ball.scale.setScalar(dotSize);
    group.add(ball);
    return ball;
  };
  const newEdge = (color) => {
    const mat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(color), depthTest: false, transparent: true });
    edgeMats.push(mat);
    const cyl = new THREE.Mesh(cylGeom, mat);
    cyl.renderOrder = 1;
    group.add(cyl);
    return cyl;
  };

  const points = keypoints.points.map((p) => ({
    ball: newNode(p.color), offset: p.offset, body: p.body, name: p.name, bodyId: p.bodyId,
  }));

  const positions = new Float32Array(keypoints.points.length * 3);
  const bones = keypoints.bones.map((pair) => (
    { cyl: newEdge(keypoints.points[pair[0]].color), i: pair[0], j: pair[1] }));

  // Combined ("medial") abdomen markers: average each left/right pair into one
  // midline chain, shown in white instead of the two side chains when the
  // "Combine abdomen" toggle is on.
  const nameIdx = new Map(keypoints.points.map((p, i) => [p.name, i]));
  const medialSrc = [0, 1, 2]
    .map((n) => [nameIdx.get(`l_abdomen${n}`), nameIdx.get(`r_abdomen${n}`)])
    .filter((pair) => pair.every((i) => i != null));
  const abdIdx = new Set(medialSrc.flat()); // side points/bones hidden when combined
  const medialPos = new Float32Array(medialSrc.length * 3);
  const medialBalls = medialSrc.map(() => { const m = newNode('#ffffff'); m.visible = false; return m; });
  const medialBones = [];
  for (let k = 0; k + 1 < medialSrc.length; k++) {
    const cyl = newEdge('#ffffff'); cyl.visible = false;
    medialBones.push({ cyl, i: k, j: k + 1 });
  }

  const a = new THREE.Vector3(), b = new THREE.Vector3(), dir = new THREE.Vector3();
  const up = new THREE.Vector3(0, 1, 0);
  const placeCyl = (cyl, pa, pb) => {
    dir.subVectors(pb, pa);
    const len = dir.length();
    cyl.position.addVectors(pa, pb).multiplyScalar(0.5);
    cyl.quaternion.setFromUnitVectors(up, dir.normalize());
    cyl.scale.set(lineWidth, len, lineWidth);
  };
  const syncBones = () => {
    for (const { cyl, i, j } of bones) {
      placeCyl(cyl, a.fromArray(positions, 3 * i), b.fromArray(positions, 3 * j));
    }
    if (combined) {
      for (let k = 0; k < medialSrc.length; k++) {
        const [li, ri] = medialSrc[k];
        for (let c = 0; c < 3; c++) medialPos[3 * k + c] = (positions[3 * li + c] + positions[3 * ri + c]) / 2;
        medialBalls[k].position.fromArray(medialPos, 3 * k);
      }
      for (const { cyl, i, j } of medialBones) {
        placeCyl(cyl, a.fromArray(medialPos, 3 * i), b.fromArray(medialPos, 3 * j));
      }
    }
  };

  return {
    group, points, positions, syncBones,
    setDotSize: (s) => {
      dotSize = s;
      for (const p of points) p.ball.scale.setScalar(s);
      for (const m of medialBalls) m.scale.setScalar(s);
    },
    setLineWidth: (w) => { lineWidth = w; syncBones(); },
    setOnTop: (onTop) => { for (const m of [...nodeMats, ...edgeMats]) m.depthTest = !onTop; },
    setNodeOpacity: (o) => { for (const m of nodeMats) m.opacity = o; },
    setEdgeOpacity: (o) => { for (const m of edgeMats) m.opacity = o; },
    setCombined: (on) => {
      combined = on;
      for (const i of abdIdx) points[i].ball.visible = !on;
      for (const { cyl, i, j } of bones) if (abdIdx.has(i) && abdIdx.has(j)) cyl.visible = !on;
      for (const m of medialBalls) m.visible = on;
      for (const { cyl } of medialBones) cyl.visible = on;
      if (on) syncBones();
    },
  };
}

function bodyIds(model, keypoints) {
  // Resolve each keypoint's body name to its integer id once.
  const id = (name) => {
    const accessor = model.body(name);
    return accessor.id;
  };
  for (const p of keypoints.points) p.bodyId = id(p.body);
}

function buildSliders(pose, keypoints, mj, model, data, qpos, onChange) {
  const container = document.getElementById('groups');
  const byGroup = new Map(pose.groups.map((gr) => [gr.key, []]));
  for (const j of pose.joints) (byGroup.get(j.group) || []).push(j);

  const limbColor = Object.fromEntries(keypoints.limbs.map((l) => [l.name, l.color]));
  const groupSwatch = {
    lf_leg: limbColor.lf_leg, lm_leg: limbColor.lm_leg, lh_leg: limbColor.lh_leg,
    rf_leg: limbColor.rf_leg, rm_leg: limbColor.rm_leg, rh_leg: limbColor.rh_leg,
    l_antenna: limbColor.l_antenna, r_antenna: limbColor.r_antenna,
    abdomen: limbColor.l_abdomen, wings: '#9aa0a6', head: '#9aa0a6',
  };

  const sliders = [];
  for (const gr of pose.groups) {
    const joints = byGroup.get(gr.key) || [];
    if (!joints.length) continue;
    const details = document.createElement('details');
    if (gr.key.endsWith('_leg')) details.open = false;
    const summary = document.createElement('summary');
    const sw = document.createElement('span');
    sw.className = 'swatch';
    sw.style.background = groupSwatch[gr.key] || '#9aa0a6';
    summary.append(sw, document.createTextNode(`${gr.label} (${joints.length})`));
    details.appendChild(summary);

    for (const j of joints) {
      const wrap = document.createElement('div'); wrap.className = 'joint';
      const row = document.createElement('div'); row.className = 'row';
      const name = document.createElement('span'); name.textContent = j.label;
      const val = document.createElement('span');
      const input = document.createElement('input');
      input.type = 'range';
      input.min = j.range[0]; input.max = j.range[1]; input.step = 0.001;
      input.value = j.neutral;
      const deg = (rad) => `${(rad * 180 / Math.PI).toFixed(0)}°`;
      val.textContent = deg(j.neutral);
      input.addEventListener('input', () => {
        const v = parseFloat(input.value);
        qpos[j.qposadr] = v;
        data.qpos[j.qposadr] = v;
        val.textContent = deg(v);
        onChange();
      });
      row.append(name, val);
      wrap.append(row, input);
      details.appendChild(wrap);
      sliders.push({ input, val, j, deg });
    }
    container.appendChild(details);
  }

  // Load a full qpos vector (all DOFs, not just the sliders) and sync the panel.
  const setPose = (values) => {
    for (let i = 0; i < values.length; i++) { qpos[i] = values[i]; data.qpos[i] = values[i]; }
    for (const { input, val, j, deg } of sliders) {
      input.value = qpos[j.qposadr];
      val.textContent = deg(qpos[j.qposadr]);
    }
    onChange();
  };
  document.getElementById('reset').addEventListener('click', () => setPose(pose.neutral_qpos));
  document.getElementById('zero').addEventListener('click',
    () => setPose(new Array(pose.neutral_qpos.length).fill(0)));

  // Legend.
  const legend = document.getElementById('legend');
  const approx = keypoints.approximate.length;
  legend.innerHTML =
    `${keypoints.points.length} keypoints · ${pose.joints.length} joint DOFs.` +
    (approx ? `<br>The ${approx} abdomen markers form two lateral chains; ` +
      `"Combine abdomen" merges them at the midline.` : '');
}
