use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;

#[pyfunction]
fn normal(seed: u64, index: u64) -> f32 {
    evo_core::normal::normal(seed, index)
}

#[pyfunction]
fn fill_perturbation(seed: u64, out: &Bound<'_, PyArray1<f32>>) {
    evo_core::perturb::fill_perturbation(seed, out.readwrite().as_slice_mut().unwrap());
}

#[pyfunction]
fn centered_ranks(fitness: PyReadonlyArray1<'_, f32>, out: &Bound<'_, PyArray1<f32>>) {
    let fitness = fitness.as_slice().unwrap();
    let mut order = vec![0usize; fitness.len()];
    evo_core::rank::centered_ranks(fitness, &mut order, out.readwrite().as_slice_mut().unwrap());
}

#[pyfunction]
fn es_gradient(
    seeds: PyReadonlyArray1<'_, u64>,
    weights: PyReadonlyArray1<'_, f32>,
    population: usize,
    sigma: f32,
    grad: &Bound<'_, PyArray1<f32>>,
) {
    let pairs: Vec<(u64, f32)> = seeds
        .as_slice()
        .unwrap()
        .iter()
        .copied()
        .zip(weights.as_slice().unwrap().iter().copied())
        .collect();
    evo_core::optim::es_gradient(&pairs, population, sigma, grad.readwrite().as_slice_mut().unwrap());
}

#[pyfunction]
fn sgd_step(theta: &Bound<'_, PyArray1<f32>>, grad: PyReadonlyArray1<'_, f32>, lr: f32) {
    evo_core::optim::sgd_step(theta.readwrite().as_slice_mut().unwrap(), grad.as_slice().unwrap(), lr);
}

#[pyfunction]
fn adam_step(
    theta: &Bound<'_, PyArray1<f32>>,
    grad: PyReadonlyArray1<'_, f32>,
    m: &Bound<'_, PyArray1<f32>>,
    v: &Bound<'_, PyArray1<f32>>,
    step: u32,
    lr: f32,
) {
    let config = evo_core::optim::AdamConfig { lr, ..Default::default() };
    evo_core::optim::adam_step(
        theta.readwrite().as_slice_mut().unwrap(),
        grad.as_slice().unwrap(),
        m.readwrite().as_slice_mut().unwrap(),
        v.readwrite().as_slice_mut().unwrap(),
        step,
        &config,
    );
}

#[pymodule]
fn evo_py(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(normal, module)?)?;
    module.add_function(wrap_pyfunction!(fill_perturbation, module)?)?;
    module.add_function(wrap_pyfunction!(centered_ranks, module)?)?;
    module.add_function(wrap_pyfunction!(es_gradient, module)?)?;
    module.add_function(wrap_pyfunction!(sgd_step, module)?)?;
    module.add_function(wrap_pyfunction!(adam_step, module)?)?;
    Ok(())
}
