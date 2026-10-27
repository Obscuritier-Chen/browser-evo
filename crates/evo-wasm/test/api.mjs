import { createRequire } from "node:module";
import assert from "node:assert/strict";

const require = createRequire(import.meta.url);
const wasm = require("../pkg/evo_wasm.js");

const fill = new Float32Array(16);
wasm.fill_perturbation(9n, fill);
for (let index = 0; index < fill.length; index++) {
    assert.equal(fill[index], wasm.normal(9n, BigInt(index)));
}

const ranks = new Float32Array(3);
wasm.centered_ranks(new Float32Array([3, 1, 2]), ranks);
assert.deepEqual([...ranks], [0.5, -0.5, 0]);

const tie = new Float32Array(3);
wasm.centered_ranks(new Float32Array([1, 1, 1]), tie);
assert.deepEqual([...tie], [-0.5, 0, 0.5]);

const grad = new Float32Array(4);
wasm.es_gradient(new BigInt64Array([5n]), new Float32Array([1]), 2, 0.5, grad);
const eps = new Float32Array(4);
wasm.fill_perturbation(5n, eps);
for (let index = 0; index < 4; index++) {
    assert.ok(Math.abs(grad[index] - eps[index]) < 1e-6);
}

const theta = new Float32Array([1, -1]);
wasm.sgd_step(theta, new Float32Array([0.5, -0.5]), 0.1);
assert.ok(Math.abs(theta[0] - 0.95) < 1e-6);
assert.ok(Math.abs(theta[1] + 0.95) < 1e-6);

const adamTheta = new Float32Array([1]);
const adamM = new Float32Array([0]);
const adamV = new Float32Array([0]);
wasm.adam_step(adamTheta, new Float32Array([2]), adamM, adamV, 1, 0.1);
assert.ok(Math.abs(adamTheta[0] - 0.9) < 1e-4);

console.log("wasm api checks passed");
