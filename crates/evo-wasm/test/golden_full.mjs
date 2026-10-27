import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import assert from "node:assert/strict";

const require = createRequire(import.meta.url);
const wasm = require("../pkg/evo_wasm.js");
const golden = (name) => new URL(`../../evo-core/tests/golden/${name}`, import.meta.url);

const toBits = (value) => {
    const view = new DataView(new ArrayBuffer(4));
    view.setFloat32(0, value);
    return view.getUint32(0);
};

let textChecked = 0;
for (const line of readFileSync(golden("normal_bits.txt"), "utf8").trim().split("\n")) {
    const [seedHex, indexHex, bitsHex] = line.trim().split(/\s+/);
    const actual = toBits(wasm.normal(BigInt("0x" + seedHex), BigInt("0x" + indexHex)));
    assert.equal(actual, parseInt(bitsHex, 16), `seed=${seedHex} index=${indexHex}`);
    textChecked++;
}

const bin = readFileSync(golden("normal_1e6.bin"));
const count = bin.length / 4;
const millionChecked = (() => {
    const out = new Float32Array(count);
    wasm.fill_perturbation(42n, out);
    for (let index = 0; index < count; index++) {
        assert.equal(toBits(out[index]), bin.readUInt32LE(index * 4), `index=${index}`);
    }
    return count;
})();

const dim = 16;
const seeds = new BigInt64Array([1n, 2n, 3n, 4n, 5n, 6n, 7n, 8n]);
const weights = new Float32Array([0.5, -0.25, 0.125, 0.0, -0.5, 0.75, -0.125, 0.25]);
const population = 16;
const sigma = 0.5;
const lr = 0.0625;
const theta = new Float32Array(dim);
for (let i = 0; i < dim; i++) theta[i] = (i - 4) * 0.25;
const m = new Float32Array(dim);
const v = new Float32Array(dim);
const grad = new Float32Array(dim);
const expected = readFileSync(golden("es_update.txt"), "utf8").trim().split("\n");
let esChecked = 0;
for (let step = 1; step <= 3; step++) {
    wasm.es_gradient(seeds, weights, population, sigma, grad);
    wasm.adam_step(theta, grad, m, v, step, lr);
    for (let index = 0; index < dim; index++) {
        const [, , bitsHex] = expected[esChecked].split(/\s+/);
        assert.equal(toBits(theta[index]), parseInt(bitsHex, 16), `step=${step} index=${index}`);
        esChecked++;
    }
}

const ranks = new Float32Array(3);
wasm.centered_ranks(new Float32Array([3, 1, 2]), ranks);
assert.deepEqual([...ranks], [0.5, -0.5, 0]);

console.log(`wasm golden: text=${textChecked} million=${millionChecked} es=${esChecked} all bit-exact`);
