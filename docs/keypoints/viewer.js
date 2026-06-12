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

  document.getElementById('toggle-points').addEventListener('change', (e) => {
    overlay.group.visible = e.target.checked; requestRender();
  });
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
  document.getElementById('skel-opacity').addEventListener('input', (e) => {
    overlay.setOpacity(parseFloat(e.target.value)); requestRender();
  });
  controls.addEventListener('change', requestRender); // orbit / zoom / pan

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

// Build one Three.js mesh per MuJoCo geom (mesh + capsule cover this model).
// Each mesh starts flat grey; the "Colours" toggle swaps in its flygym colour.
function buildMeshes(model, colors) {
  const group = new THREE.Group();
  const items = [];
  for (let g = 0; g < model.ngeom; g++) {
    const geometry = geometryForGeom(model, g, model.geom_type[g]);
    if (!geometry) continue;
    const rgb = colors.geom_rgb[g] || [0.7, 0.7, 0.7];
    // colors.json holds sRGB values (from flygym); tag them as such so three
    // doesn't treat them as linear and wash the dark colours out.
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
  let dotSize = DEFAULT_DOT, lineWidth = DEFAULT_LINE;

  // depthTest is off by default (overlay drawn "on top"); transparent:true puts
  // the overlay in the same render pass as the mesh, so when depthTest is turned
  // back on the higher renderOrder + the mesh's depth let the mesh occlude it.
  const materials = [];
  const sphereGeom = new THREE.SphereGeometry(1, 16, 12);
  const points = keypoints.points.map((p) => {
    const mat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(p.color), depthTest: false, transparent: true });
    materials.push(mat);
    const ball = new THREE.Mesh(sphereGeom, mat);
    ball.renderOrder = 2;
    ball.scale.setScalar(dotSize);
    group.add(ball);
    return { ball, offset: p.offset, body: p.body, name: p.name, bodyId: p.bodyId };
  });

  const positions = new Float32Array(keypoints.points.length * 3);
  const cylGeom = new THREE.CylinderGeometry(1, 1, 1, 8); // unit; axis +Y
  const bones = keypoints.bones.map((pair) => {
    const mat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(keypoints.points[pair[0]].color),
      depthTest: false, transparent: true });
    materials.push(mat);
    const cyl = new THREE.Mesh(cylGeom, mat);
    cyl.renderOrder = 1;
    group.add(cyl);
    return { cyl, i: pair[0], j: pair[1] };
  });

  const a = new THREE.Vector3(), b = new THREE.Vector3(), dir = new THREE.Vector3();
  const up = new THREE.Vector3(0, 1, 0);
  const syncBones = () => {
    for (const { cyl, i, j } of bones) {
      a.fromArray(positions, 3 * i);
      b.fromArray(positions, 3 * j);
      dir.subVectors(b, a);
      const len = dir.length();
      cyl.position.addVectors(a, b).multiplyScalar(0.5);
      cyl.quaternion.setFromUnitVectors(up, dir.normalize());
      cyl.scale.set(lineWidth, len, lineWidth);
    }
  };

  return {
    group, points, positions, syncBones,
    setDotSize: (s) => { dotSize = s; for (const p of points) p.ball.scale.setScalar(s); },
    setLineWidth: (w) => { lineWidth = w; syncBones(); },
    setOnTop: (onTop) => { for (const m of materials) m.depthTest = !onTop; },
    setOpacity: (o) => { for (const m of materials) m.opacity = o; },
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

  document.getElementById('reset').addEventListener('click', () => {
    for (const { input, val, j, deg } of sliders) {
      input.value = j.neutral;
      qpos[j.qposadr] = j.neutral;
      data.qpos[j.qposadr] = j.neutral;
      val.textContent = deg(j.neutral);
    }
    onChange();
  });

  // Legend.
  const legend = document.getElementById('legend');
  const approx = keypoints.approximate.length;
  legend.innerHTML =
    `${keypoints.points.length} keypoints · ${pose.joints.length} joint DOFs.` +
    (approx ? `<br>The ${approx} abdomen markers have no exact NeuroMechFly ` +
      `counterpart and are placed for illustration.` : '');
}
