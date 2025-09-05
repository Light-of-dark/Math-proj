// Simplified fluid simulation using Stable Fluids (Jos Stam's method)
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const N = 128; // higher grid size for smoother simulation
const iter = 4;
const dt = 0.1;
const diff = 0.00001;
const visc = 0.00001;

let size = (N+2)*(N+2);

let u = new Float32Array(size);
let v = new Float32Array(size);
let u_prev = new Float32Array(size);
let v_prev = new Float32Array(size);
let dens = new Float32Array(size);
let dens_prev = new Float32Array(size);

function IX(x, y) { return x + (N+2)*y; }

function add_source(x, s) {
  for (let i = 0; i < size; i++) x[i] += dt * s[i];
}

function set_bnd(b, x) {
  for (let i = 1; i <= N; i++) {
    x[IX(0, i)]   = b === 1 ? -x[IX(1, i)] : x[IX(1, i)];
    x[IX(N+1, i)] = b === 1 ? -x[IX(N, i)] : x[IX(N, i)];
    x[IX(i, 0)]   = b === 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
    x[IX(i, N+1)] = b === 2 ? -x[IX(i, N)] : x[IX(i, N)];
  }
  x[IX(0, 0)]       = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
  x[IX(0, N+1)]     = 0.5 * (x[IX(1, N+1)] + x[IX(0, N)]);
  x[IX(N+1, 0)]     = 0.5 * (x[IX(N, 0)] + x[IX(N+1, 1)]);
  x[IX(N+1, N+1)]   = 0.5 * (x[IX(N, N+1)] + x[IX(N+1, N)]);
}

function lin_solve(b, x, x0, a, c) {
  for (let k = 0; k < iter; k++) {
    for (let i = 1; i <= N; i++) {
      for (let j = 1; j <= N; j++) {
        x[IX(i, j)] = (x0[IX(i, j)] + a*(x[IX(i-1, j)] + x[IX(i+1, j)] + x[IX(i, j-1)] + x[IX(i, j+1)])) / c;
      }
    }
    set_bnd(b, x);
  }
}

function diffuse(b, x, x0, diff) {
  let a = dt * diff * N * N;
  lin_solve(b, x, x0, a, 1+4*a);
}

function advect(b, d, d0, u, v) {
  let dt0 = dt*N;
  for (let i = 1; i <= N; i++) {
    for (let j = 1; j <= N; j++) {
      let x = i - dt0 * u[IX(i, j)];
      let y = j - dt0 * v[IX(i, j)];
      if (x < 0.5) x = 0.5; if (x > N+0.5) x = N+0.5;
      if (y < 0.5) y = 0.5; if (y > N+0.5) y = N+0.5;
      let i0 = Math.floor(x); let i1 = i0+1;
      let j0 = Math.floor(y); let j1 = j0+1;
      let s1 = x-i0; let s0 = 1-s1;
      let t1 = y-j0; let t0 = 1-t1;
      d[IX(i, j)] =
        s0*(t0*d0[IX(i0, j0)] + t1*d0[IX(i0, j1)]) +
        s1*(t0*d0[IX(i1, j0)] + t1*d0[IX(i1, j1)]);
    }
  }
  set_bnd(b, d);
}

function project(u, v, p, div) {
  for (let i = 1; i <= N; i++) {
    for (let j = 1; j <= N; j++) {
      div[IX(i, j)] = -0.5*(u[IX(i+1, j)]-u[IX(i-1, j)]+v[IX(i, j+1)]-v[IX(i, j-1)])/N;
      p[IX(i, j)] = 0;
    }
  }
  set_bnd(0, div); set_bnd(0, p);
  lin_solve(0, p, div, 1, 4);
  for (let i = 1; i <= N; i++) {
    for (let j = 1; j <= N; j++) {
      u[IX(i, j)] -= 0.5*N*(p[IX(i+1, j)]-p[IX(i-1, j)]);
      v[IX(i, j)] -= 0.5*N*(p[IX(i, j+1)]-p[IX(i, j-1)]);
    }
  }
  set_bnd(1, u); set_bnd(2, v);
}

function dens_step(x, x0, u, v, diff) {
  add_source(x, x0);
  [x0, x] = [x, x0]; diffuse(0, x, x0, diff);
  [x0, x] = [x, x0]; advect(0, x, x0, u, v);
}

function vel_step(u, v, u0, v0, visc) {
  add_source(u, u0); add_source(v, v0);
  [u0, u] = [u, u0]; diffuse(1, u, u0, visc);
  [v0, v] = [v, v0]; diffuse(2, v, v0, visc);
  project(u, v, u0, v0);
  [u0, u] = [u, u0]; [v0, v] = [v, v0];
  advect(1, u, u0, u0, v0); advect(2, v, v0, u0, v0);
  project(u, v, u0, v0);
}

function render_dens() {
  const imageData = ctx.createImageData(canvas.width, canvas.height);
  const data = imageData.data;
  const cellW = canvas.width / N;
  const cellH = canvas.height / N;

  for (let i = 1; i <= N; i++) {
    for (let j = 1; j <= N; j++) {
      let d = dens[IX(i, j)];
      let intensity = Math.min(255, d * 8);
      let r = Math.min(255, intensity * 0.5);
      let g = Math.min(255, intensity * 0.8);
      let b = 255;
      let alpha = Math.min(255, intensity);

      let xStart = Math.floor((i-1) * cellW);
      let yStart = Math.floor((j-1) * cellH);
      let xEnd = Math.floor(i * cellW);
      let yEnd = Math.floor(j * cellH);

      for (let x = xStart; x < xEnd; x++) {
        for (let y = yStart; y < yEnd; y++) {
          let idx = (x + y*canvas.width) * 4;
          data[idx] = r;
          data[idx+1] = g;
          data[idx+2] = b;
          data[idx+3] = alpha;
        }
      }
    }
  }

  ctx.putImageData(imageData, 0, 0);
}

function step() {
  vel_step(u, v, u_prev, v_prev, visc);
  dens_step(dens, dens_prev, u, v, diff);
  render_dens();
  u_prev.fill(0); v_prev.fill(0); dens_prev.fill(0);
  requestAnimationFrame(step);
}

canvas.addEventListener('mousemove', e => {
  const i = Math.floor((e.offsetX / canvas.width) * N) + 1;
  const j = Math.floor((e.offsetY / canvas.height) * N) + 1;
  dens_prev[IX(i, j)] = 1000;
  u_prev[IX(i, j)] = (Math.random() - 0.5) * 600;
  v_prev[IX(i, j)] = (Math.random() - 0.5) * 600;
});
document.getElementById('clearBtn').addEventListener('click', () => {
  dens.fill(0);
  dens_prev.fill(0);
  u.fill(0);
  v.fill(0);
  u_prev.fill(0);
  v_prev.fill(0);
});

step();
