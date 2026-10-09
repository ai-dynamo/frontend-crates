// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{DecisionError, Dialect, Distribution};
const TOLERANCE: f64 = 1e-9;
// Engine vocabulary log probabilities are commonly rounded to float32. This
// allows that rounding at unit total mass without relaxing distribution checks.
const VOCABULARY_MASS_TOLERANCE: f64 = 1e-6;

fn invalid(message: &str) -> DecisionError {
    DecisionError::execution(Dialect::OpenAi, message)
}

/// Reduce complete requested-token log probabilities normalized over the full vocabulary.
/// Candidate/head logits must not use this entry point.
pub fn reduce_vocab_logprobs(
    scores: &[f64],
    temperature: f64,
) -> Result<Distribution, DecisionError> {
    if scores.is_empty() || !temperature.is_finite() || temperature <= 0.0 {
        return Err(invalid(
            "candidate scores require a finite positive temperature and nonempty candidates",
        ));
    }
    if scores.iter().any(|x| x.is_nan() || *x > 0.0) {
        return Err(invalid(
            "vocabulary log probabilities must be nonpositive and not NaN",
        ));
    }
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return Err(invalid("all candidates have zero probability"));
    }
    let mass: f64 = scores.iter().map(|x| x.exp()).sum();
    if mass > 1.0 + VOCABULARY_MASS_TOLERANCE {
        return Err(invalid("candidate vocabulary mass exceeds one"));
    }
    // Subtract before dividing to avoid overflow for very small temperatures.
    let weights: Vec<_> = scores
        .iter()
        .map(|x| ((x - maximum) / temperature).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    Ok(Distribution {
        probabilities: weights.iter().map(|x| x / total).collect(),
        label_mass: Some(mass.min(1.0)),
    })
}

pub(crate) fn validate_probabilities(probabilities: &[f64]) -> Result<(), DecisionError> {
    if probabilities.is_empty()
        || probabilities
            .iter()
            .any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
        || (probabilities.iter().sum::<f64>() - 1.0).abs() > TOLERANCE
    {
        return Err(invalid(
            "executor returned an invalid candidate distribution",
        ));
    }
    Ok(())
}

impl Distribution {
    pub fn validate(&self, count: usize) -> Result<(), DecisionError> {
        if self.probabilities.len() != count {
            return Err(invalid(
                "executor candidate cardinality differs from the request",
            ));
        }
        validate_probabilities(&self.probabilities)?;
        if self
            .label_mass
            .is_some_and(|x| !x.is_finite() || !(0.0..=1.0).contains(&x))
        {
            return Err(invalid("executor vocabulary label mass is invalid"));
        }
        Ok(())
    }
}

pub(crate) fn modal_index(probabilities: &[f64]) -> usize {
    probabilities
        .iter()
        .enumerate()
        .fold(0, |best, (index, p)| {
            if *p > probabilities[best] {
                index
            } else {
                best
            }
        })
}

pub fn choice_confidence(probabilities: &[f64]) -> Result<f64, DecisionError> {
    validate_probabilities(probabilities)?;
    if probabilities.len() == 1 {
        return Ok(1.0);
    }
    let n = probabilities.len() as f64;
    Ok(((n * probabilities[modal_index(probabilities)] - 1.0) / (n - 1.0)).clamp(0.0, 1.0))
}

pub fn score_confidence(probabilities: &[f64]) -> Result<f64, DecisionError> {
    validate_probabilities(probabilities)?;
    if probabilities.len() == 1 {
        return Ok(1.0);
    }
    let n = probabilities.len() as f64;
    let mode = modal_index(probabilities) as f64;
    let distance: f64 = probabilities
        .iter()
        .enumerate()
        .map(|(i, p)| p * (i as f64 - mode).abs())
        .sum();
    let uniform_distance = (0..probabilities.len())
        .map(|i| (i as f64 - (n - 1.0) / 2.0).abs())
        .sum::<f64>()
        / n;
    Ok((1.0 - distance / uniform_distance).clamp(0.0, 1.0))
}
