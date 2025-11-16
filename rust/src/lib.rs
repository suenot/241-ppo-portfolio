//! PPO Portfolio Trading
//!
//! Implementation of Proximal Policy Optimization (PPO) for multi-asset
//! portfolio management with Bybit market data integration.

use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

// ============================================================================
// Neural Network Layer
// ============================================================================

/// A simple dense (fully connected) neural network layer.
#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
}

impl DenseLayer {
    /// Create a new dense layer with Xavier initialization.
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (input_size + output_size) as f64).sqrt();

        let weights = Array2::from_shape_fn((input_size, output_size), |_| {
            rng.gen_range(-scale..scale)
        });
        let biases = Array1::zeros(output_size);

        Self { weights, biases }
    }

    /// Forward pass: output = input * weights + biases
    pub fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        input.dot(&self.weights) + &self.biases
    }
}

/// Apply ReLU activation element-wise.
pub fn relu(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v.max(0.0))
}

/// Apply tanh activation element-wise.
pub fn tanh_activation(x: &Array1<f64>) -> Array1<f64> {
    x.mapv(|v| v.tanh())
}

/// Apply softmax to convert logits to portfolio weights.
pub fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_x = x.mapv(|v| (v - max_val).exp());
    let sum_exp: f64 = exp_x.sum();
    exp_x / sum_exp
}

// ============================================================================
// Actor-Critic Network
// ============================================================================

/// Actor-Critic network with shared backbone for PPO.
///
/// Architecture:
/// - Shared backbone: input -> hidden1 -> hidden2
/// - Actor head: hidden2 -> action_mean, action_log_std
/// - Critic head: hidden2 -> value
#[derive(Debug, Clone)]
pub struct ActorCritic {
    /// Shared backbone layers
    pub backbone_layer1: DenseLayer,
    pub backbone_layer2: DenseLayer,
    /// Actor head: outputs action means
    pub actor_mean_layer: DenseLayer,
    /// Actor log standard deviations (learnable parameter per action dim)
    pub actor_log_std: Array1<f64>,
    /// Critic head: outputs state value
    pub critic_layer: DenseLayer,
    /// Number of assets (action dimensions)
    pub num_assets: usize,
}

impl ActorCritic {
    /// Create a new actor-critic network.
    ///
    /// # Arguments
    /// * `state_dim` - Dimension of the state/observation space
    /// * `num_assets` - Number of assets in the portfolio
    /// * `hidden_size` - Size of hidden layers in the shared backbone
    pub fn new(state_dim: usize, num_assets: usize, hidden_size: usize) -> Self {
        Self {
            backbone_layer1: DenseLayer::new(state_dim, hidden_size),
            backbone_layer2: DenseLayer::new(hidden_size, hidden_size),
            actor_mean_layer: DenseLayer::new(hidden_size, num_assets),
            actor_log_std: Array1::from_elem(num_assets, -0.5), // Initial std ~ 0.6
            critic_layer: DenseLayer::new(hidden_size, 1),
            num_assets,
        }
    }

    /// Forward pass through the shared backbone.
    pub fn backbone_forward(&self, state: &Array1<f64>) -> Array1<f64> {
        let h1 = relu(&self.backbone_layer1.forward(state));
        tanh_activation(&self.backbone_layer2.forward(&h1))
    }

    /// Get action distribution parameters (mean, std) and state value.
    pub fn forward(&self, state: &Array1<f64>) -> (Array1<f64>, Array1<f64>, f64) {
        let features = self.backbone_forward(state);
        let action_mean = self.actor_mean_layer.forward(&features);
        let action_std = self.actor_log_std.mapv(|v| v.exp());
        let value = self.critic_layer.forward(&features)[0];
        (action_mean, action_std, value)
    }

    /// Sample an action from the policy distribution.
    pub fn sample_action(&self, state: &Array1<f64>) -> (Array1<f64>, f64, f64) {
        let (mean, std, value) = self.forward(state);
        let mut rng = rand::thread_rng();

        // Sample from Gaussian
        let raw_action = Array1::from_shape_fn(self.num_assets, |i| {
            mean[i] + std[i] * sample_standard_normal(&mut rng)
        });

        // Compute log probability
        let log_prob = gaussian_log_prob(&raw_action, &mean, &std);

        // Convert to portfolio weights via softmax
        let weights = softmax(&raw_action);

        (weights, log_prob, value)
    }

    /// Evaluate actions: compute log probability and value for given state-action pairs.
    pub fn evaluate_action(
        &self,
        state: &Array1<f64>,
        action_raw: &Array1<f64>,
    ) -> (f64, f64, f64) {
        let (mean, std, value) = self.forward(state);
        let log_prob = gaussian_log_prob(action_raw, &mean, &std);
        let entropy = gaussian_entropy(&std);
        (log_prob, value, entropy)
    }

    /// Apply parameter updates using simple SGD.
    pub fn apply_gradients(
        &mut self,
        gradients: &ActorCriticGradients,
        learning_rate: f64,
    ) {
        apply_dense_gradient(&mut self.backbone_layer1, &gradients.backbone1_w, &gradients.backbone1_b, learning_rate);
        apply_dense_gradient(&mut self.backbone_layer2, &gradients.backbone2_w, &gradients.backbone2_b, learning_rate);
        apply_dense_gradient(&mut self.actor_mean_layer, &gradients.actor_mean_w, &gradients.actor_mean_b, learning_rate);
        apply_dense_gradient(&mut self.critic_layer, &gradients.critic_w, &gradients.critic_b, learning_rate);

        // Update log_std
        self.actor_log_std = &self.actor_log_std - &(&gradients.actor_log_std * learning_rate);
    }
}

/// Gradients for the actor-critic network.
#[derive(Debug, Clone)]
pub struct ActorCriticGradients {
    pub backbone1_w: Array2<f64>,
    pub backbone1_b: Array1<f64>,
    pub backbone2_w: Array2<f64>,
    pub backbone2_b: Array1<f64>,
    pub actor_mean_w: Array2<f64>,
    pub actor_mean_b: Array1<f64>,
    pub actor_log_std: Array1<f64>,
    pub critic_w: Array2<f64>,
    pub critic_b: Array1<f64>,
}

fn apply_dense_gradient(
    layer: &mut DenseLayer,
    w_grad: &Array2<f64>,
    b_grad: &Array1<f64>,
    lr: f64,
) {
    layer.weights = &layer.weights - &(w_grad * lr);
    layer.biases = &layer.biases - &(b_grad * lr);
}

// ============================================================================
// Gaussian Distribution Utilities
// ============================================================================

/// Sample from standard normal distribution using Box-Muller transform.
pub fn sample_standard_normal(rng: &mut impl Rng) -> f64 {
    let u1: f64 = rng.gen_range(1e-10..1.0);
    let u2: f64 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Compute log probability of a sample under a diagonal Gaussian.
pub fn gaussian_log_prob(x: &Array1<f64>, mean: &Array1<f64>, std: &Array1<f64>) -> f64 {
    let n = x.len() as f64;
    let mut log_prob = -0.5 * n * (2.0 * PI).ln();
    for i in 0..x.len() {
        let diff = x[i] - mean[i];
        log_prob -= std[i].ln() + 0.5 * (diff / std[i]).powi(2);
    }
    log_prob
}

/// Compute entropy of a diagonal Gaussian distribution.
pub fn gaussian_entropy(std: &Array1<f64>) -> f64 {
    let n = std.len() as f64;
    0.5 * n * (1.0 + (2.0 * PI).ln()) + std.mapv(|s| s.ln()).sum()
}

/// Compute KL divergence between two diagonal Gaussians.
pub fn gaussian_kl_divergence(
    mean1: &Array1<f64>,
    std1: &Array1<f64>,
    mean2: &Array1<f64>,
    std2: &Array1<f64>,
) -> f64 {
    let mut kl = 0.0;
    for i in 0..mean1.len() {
        let var1 = std1[i] * std1[i];
        let var2 = std2[i] * std2[i];
        kl += (std2[i] / std1[i]).ln() + (var1 + (mean1[i] - mean2[i]).powi(2)) / (2.0 * var2)
            - 0.5;
    }
    kl
}

// ============================================================================
// Generalized Advantage Estimation (GAE)
// ============================================================================

/// Compute Generalized Advantage Estimation.
///
/// # Arguments
/// * `rewards` - Sequence of rewards
/// * `values` - Sequence of state values (one more than rewards for bootstrap)
/// * `gamma` - Discount factor
/// * `lambda` - GAE lambda parameter for bias-variance tradeoff
///
/// # Returns
/// Vector of advantage estimates and vector of returns (value targets)
pub fn compute_gae(
    rewards: &[f64],
    values: &[f64],
    gamma: f64,
    lambda: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = rewards.len();
    let mut advantages = vec![0.0; n];
    let mut gae = 0.0;

    for t in (0..n).rev() {
        let next_value = if t + 1 < values.len() {
            values[t + 1]
        } else {
            0.0
        };
        let delta = rewards[t] + gamma * next_value - values[t];
        gae = delta + gamma * lambda * gae;
        advantages[t] = gae;
    }

    // Compute returns as advantages + values
    let returns: Vec<f64> = advantages
        .iter()
        .zip(values.iter())
        .map(|(a, v)| a + v)
        .collect();

    (advantages, returns)
}

// ============================================================================
// PPO Clipped Objective
// ============================================================================

/// PPO hyperparameters.
#[derive(Debug, Clone)]
pub struct PPOConfig {
    /// Clipping parameter epsilon
    pub clip_epsilon: f64,
    /// Discount factor
    pub gamma: f64,
    /// GAE lambda
    pub gae_lambda: f64,
    /// Value function loss coefficient
    pub value_coeff: f64,
    /// Entropy bonus coefficient
    pub entropy_coeff: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Number of optimization epochs per batch
    pub num_epochs: usize,
    /// Minibatch size
    pub minibatch_size: usize,
    /// Maximum KL divergence before early stopping
    pub max_kl: f64,
}

impl Default for PPOConfig {
    fn default() -> Self {
        Self {
            clip_epsilon: 0.2,
            gamma: 0.99,
            gae_lambda: 0.95,
            value_coeff: 0.5,
            entropy_coeff: 0.01,
            learning_rate: 3e-4,
            num_epochs: 4,
            minibatch_size: 64,
            max_kl: 0.015,
        }
    }
}

/// Compute the PPO clipped surrogate loss for a single sample.
///
/// # Arguments
/// * `ratio` - Probability ratio pi_new / pi_old
/// * `advantage` - Advantage estimate
/// * `clip_epsilon` - Clipping parameter
///
/// # Returns
/// The clipped surrogate loss (negated for minimization)
pub fn ppo_clipped_loss(ratio: f64, advantage: f64, clip_epsilon: f64) -> f64 {
    let unclipped = ratio * advantage;
    let clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage;
    -unclipped.min(clipped)
}

/// Compute the combined PPO loss for a batch of experiences.
pub fn compute_ppo_loss(
    old_log_probs: &[f64],
    new_log_probs: &[f64],
    advantages: &[f64],
    values: &[f64],
    returns: &[f64],
    entropies: &[f64],
    config: &PPOConfig,
) -> (f64, f64, f64, f64) {
    let n = old_log_probs.len() as f64;
    let mut policy_loss = 0.0;
    let mut value_loss = 0.0;
    let mut entropy_mean = 0.0;
    let mut approx_kl = 0.0;

    for i in 0..old_log_probs.len() {
        // Policy loss (clipped surrogate)
        let ratio = (new_log_probs[i] - old_log_probs[i]).exp();
        policy_loss += ppo_clipped_loss(ratio, advantages[i], config.clip_epsilon);

        // Value loss
        let vf_loss = (values[i] - returns[i]).powi(2);
        value_loss += vf_loss;

        // Entropy
        entropy_mean += entropies[i];

        // Approximate KL divergence
        approx_kl += old_log_probs[i] - new_log_probs[i];
    }

    policy_loss /= n;
    value_loss /= n;
    entropy_mean /= n;
    approx_kl /= n;

    let total_loss = policy_loss + config.value_coeff * value_loss - config.entropy_coeff * entropy_mean;

    (total_loss, policy_loss, value_loss, approx_kl)
}

// ============================================================================
// Experience Buffer
// ============================================================================

/// A single transition in the environment.
#[derive(Debug, Clone)]
pub struct Transition {
    pub state: Array1<f64>,
    pub action_raw: Array1<f64>,
    pub action_weights: Array1<f64>,
    pub log_prob: f64,
    pub reward: f64,
    pub value: f64,
}

/// Buffer for storing collected experience.
#[derive(Debug, Clone)]
pub struct ExperienceBuffer {
    pub transitions: Vec<Transition>,
}

impl ExperienceBuffer {
    pub fn new() -> Self {
        Self {
            transitions: Vec::new(),
        }
    }

    pub fn push(&mut self, transition: Transition) {
        self.transitions.push(transition);
    }

    pub fn clear(&mut self) {
        self.transitions.clear();
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    /// Compute advantages and returns for the collected buffer.
    pub fn compute_advantages(&self, gamma: f64, lambda: f64) -> (Vec<f64>, Vec<f64>) {
        let rewards: Vec<f64> = self.transitions.iter().map(|t| t.reward).collect();
        let values: Vec<f64> = self.transitions.iter().map(|t| t.value).collect();
        compute_gae(&rewards, &values, gamma, lambda)
    }

    /// Normalize advantages to have zero mean and unit variance.
    pub fn normalize_advantages(advantages: &mut Vec<f64>) {
        let n = advantages.len() as f64;
        if n < 2.0 {
            return;
        }
        let mean = advantages.iter().sum::<f64>() / n;
        let variance = advantages.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std = (variance + 1e-8).sqrt();
        for a in advantages.iter_mut() {
            *a = (*a - mean) / std;
        }
    }
}

impl Default for ExperienceBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Portfolio Environment
// ============================================================================

/// Multi-asset portfolio trading environment.
#[derive(Debug, Clone)]
pub struct PortfolioEnv {
    /// Price data for each asset: assets x timesteps
    pub prices: Vec<Vec<f64>>,
    /// Number of assets
    pub num_assets: usize,
    /// Current time step
    pub current_step: usize,
    /// Lookback window for features
    pub lookback: usize,
    /// Current portfolio weights
    pub weights: Array1<f64>,
    /// Transaction cost rate
    pub transaction_cost: f64,
    /// Portfolio value
    pub portfolio_value: f64,
    /// Number of features per observation
    pub state_dim: usize,
}

impl PortfolioEnv {
    /// Create a new portfolio environment.
    ///
    /// # Arguments
    /// * `prices` - Vector of price series, one per asset
    /// * `transaction_cost` - Transaction cost rate (e.g., 0.001 for 0.1%)
    /// * `lookback` - Number of past periods for feature computation
    pub fn new(prices: Vec<Vec<f64>>, transaction_cost: f64, lookback: usize) -> Self {
        let num_assets = prices.len();
        // Features per asset: returns(1) + volatility(1) + normalized_volume(1) = 3
        // Plus current weights(num_assets)
        let features_per_asset = 3;
        let state_dim = num_assets * features_per_asset + num_assets;

        // Start with equal weights
        let initial_weight = 1.0 / num_assets as f64;
        let weights = Array1::from_elem(num_assets, initial_weight);

        Self {
            prices,
            num_assets,
            current_step: lookback,
            lookback,
            weights,
            transaction_cost,
            portfolio_value: 10000.0,
            state_dim,
        }
    }

    /// Reset the environment to the initial state.
    pub fn reset(&mut self) -> Array1<f64> {
        self.current_step = self.lookback;
        let initial_weight = 1.0 / self.num_assets as f64;
        self.weights = Array1::from_elem(self.num_assets, initial_weight);
        self.portfolio_value = 10000.0;
        self.get_state()
    }

    /// Get the current state observation.
    pub fn get_state(&self) -> Array1<f64> {
        let mut state = Vec::with_capacity(self.state_dim);

        for asset in 0..self.num_assets {
            // Return over lookback period
            let current_price = self.prices[asset][self.current_step];
            let past_price = self.prices[asset][self.current_step - self.lookback];
            let ret = if past_price > 0.0 {
                (current_price / past_price) - 1.0
            } else {
                0.0
            };
            state.push(ret);

            // Simple volatility estimate (std of recent returns)
            let mut returns = Vec::new();
            for i in 1..=self.lookback.min(self.current_step) {
                let p1 = self.prices[asset][self.current_step - i];
                let p0 = self.prices[asset][self.current_step - i + 1];
                if p1 > 0.0 {
                    returns.push((p0 / p1) - 1.0);
                }
            }
            let vol = if returns.len() > 1 {
                let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
                let var = returns
                    .iter()
                    .map(|r| (r - mean_ret).powi(2))
                    .sum::<f64>()
                    / (returns.len() - 1) as f64;
                var.sqrt()
            } else {
                0.0
            };
            state.push(vol);

            // Momentum (short-term return)
            let short_lookback = (self.lookback / 4).max(1);
            let short_past = if self.current_step >= short_lookback {
                self.prices[asset][self.current_step - short_lookback]
            } else {
                self.prices[asset][0]
            };
            let momentum = if short_past > 0.0 {
                (current_price / short_past) - 1.0
            } else {
                0.0
            };
            state.push(momentum);
        }

        // Current portfolio weights
        for i in 0..self.num_assets {
            state.push(self.weights[i]);
        }

        Array1::from_vec(state)
    }

    /// Take a step in the environment with the given portfolio weights.
    ///
    /// # Returns
    /// (next_state, reward, done)
    pub fn step(&mut self, new_weights: &Array1<f64>) -> (Array1<f64>, f64, bool) {
        // Compute transaction costs
        let turnover: f64 = self
            .weights
            .iter()
            .zip(new_weights.iter())
            .map(|(w_old, w_new)| (w_new - w_old).abs())
            .sum();
        let tc = self.transaction_cost * turnover;

        // Move to next step
        self.current_step += 1;
        let done = self.current_step >= self.min_length() - 1;

        // Compute portfolio return
        let mut portfolio_return = 0.0;
        for asset in 0..self.num_assets {
            let price_now = self.prices[asset][self.current_step];
            let price_prev = self.prices[asset][self.current_step - 1];
            let asset_return = if price_prev > 0.0 {
                (price_now / price_prev) - 1.0
            } else {
                0.0
            };
            portfolio_return += new_weights[asset] * asset_return;
        }

        // Net return after transaction costs
        let net_return = portfolio_return - tc;
        self.portfolio_value *= 1.0 + net_return;

        // Update weights (drift due to price changes)
        self.weights = new_weights.clone();

        // Reward: risk-adjusted return (simple Sharpe-like reward)
        let reward = net_return * 100.0; // Scale for easier learning

        let next_state = self.get_state();
        (next_state, reward, done)
    }

    /// Get the minimum length across all asset price series.
    pub fn min_length(&self) -> usize {
        self.prices
            .iter()
            .map(|p| p.len())
            .min()
            .unwrap_or(0)
    }

    /// Get the total number of available steps.
    pub fn total_steps(&self) -> usize {
        let min_len = self.min_length();
        if min_len > self.lookback + 1 {
            min_len - self.lookback - 1
        } else {
            0
        }
    }
}

// ============================================================================
// PPO Trainer
// ============================================================================

/// PPO training agent for portfolio management.
#[derive(Debug)]
pub struct PPOTrainer {
    pub actor_critic: ActorCritic,
    pub config: PPOConfig,
    pub buffer: ExperienceBuffer,
}

impl PPOTrainer {
    /// Create a new PPO trainer.
    pub fn new(state_dim: usize, num_assets: usize, hidden_size: usize, config: PPOConfig) -> Self {
        Self {
            actor_critic: ActorCritic::new(state_dim, num_assets, hidden_size),
            config,
            buffer: ExperienceBuffer::new(),
        }
    }

    /// Collect experience by running the policy in the environment.
    pub fn collect_experience(&mut self, env: &mut PortfolioEnv, num_steps: usize) {
        self.buffer.clear();
        let mut state = env.get_state();

        for _ in 0..num_steps {
            let (action_mean, action_std, value) = self.actor_critic.forward(&state);
            let mut rng = rand::thread_rng();

            // Sample raw action from Gaussian
            let action_raw = Array1::from_shape_fn(env.num_assets, |i| {
                action_mean[i] + action_std[i] * sample_standard_normal(&mut rng)
            });

            let log_prob = gaussian_log_prob(&action_raw, &action_mean, &action_std);
            let action_weights = softmax(&action_raw);

            let (next_state, reward, done) = env.step(&action_weights);

            self.buffer.push(Transition {
                state: state.clone(),
                action_raw: action_raw.clone(),
                action_weights,
                log_prob,
                reward,
                value,
            });

            if done {
                state = env.reset();
            } else {
                state = next_state;
            }
        }
    }

    /// Perform PPO update using collected experience.
    /// Returns (mean_policy_loss, mean_value_loss, mean_kl).
    pub fn update(&mut self) -> (f64, f64, f64) {
        let (mut advantages, returns) =
            self.buffer.compute_advantages(self.config.gamma, self.config.gae_lambda);
        ExperienceBuffer::normalize_advantages(&mut advantages);

        let mut total_policy_loss = 0.0;
        let mut total_value_loss = 0.0;
        let mut total_kl = 0.0;
        let mut update_count = 0.0;

        let old_log_probs: Vec<f64> = self.buffer.transitions.iter().map(|t| t.log_prob).collect();

        for _epoch in 0..self.config.num_epochs {
            // Process full batch (simplified - no minibatch shuffling for clarity)
            let mut new_log_probs = Vec::new();
            let mut new_values = Vec::new();
            let mut new_entropies = Vec::new();

            for t in &self.buffer.transitions {
                let (lp, v, ent) = self.actor_critic.evaluate_action(&t.state, &t.action_raw);
                new_log_probs.push(lp);
                new_values.push(v);
                new_entropies.push(ent);
            }

            let (_, policy_loss, value_loss, approx_kl) = compute_ppo_loss(
                &old_log_probs,
                &new_log_probs,
                &advantages,
                &new_values,
                &returns,
                &new_entropies,
                &self.config,
            );

            total_policy_loss += policy_loss;
            total_value_loss += value_loss;
            total_kl += approx_kl;
            update_count += 1.0;

            // Early stopping if KL divergence too large
            if approx_kl.abs() > self.config.max_kl {
                break;
            }

            // Apply numerical gradient updates (finite difference approximation)
            self.numerical_update(&advantages, &returns);
        }

        (
            total_policy_loss / update_count,
            total_value_loss / update_count,
            total_kl / update_count,
        )
    }

    /// Apply numerical gradient update using finite differences.
    /// This is a simplified approach for demonstration purposes.
    fn numerical_update(&mut self, advantages: &[f64], returns: &[f64]) {
        let lr = self.config.learning_rate;
        let epsilon = 1e-4;

        // Update actor log_std using REINFORCE-style update
        for i in 0..self.actor_critic.actor_log_std.len() {
            let mut grad = 0.0;
            for (j, t) in self.buffer.transitions.iter().enumerate() {
                let (mean, std, _) = self.actor_critic.forward(&t.state);
                let diff = t.action_raw[i] - mean[i];
                // Gradient of log prob w.r.t. log_std
                let dlp_dlogstd = (diff * diff) / (std[i] * std[i]) - 1.0;
                grad += dlp_dlogstd * advantages[j];
            }
            grad /= self.buffer.len() as f64;
            self.actor_critic.actor_log_std[i] += lr * grad;
        }

        // Update actor mean layer biases using policy gradient
        for i in 0..self.actor_critic.actor_mean_layer.biases.len() {
            let mut grad = 0.0;
            for (j, t) in self.buffer.transitions.iter().enumerate() {
                let (mean, std, _) = self.actor_critic.forward(&t.state);
                let diff = t.action_raw[i] - mean[i];
                grad += (diff / (std[i] * std[i])) * advantages[j];
            }
            grad /= self.buffer.len() as f64;
            self.actor_critic.actor_mean_layer.biases[i] += lr * grad;
        }

        // Update critic bias using value gradient
        for i in 0..self.actor_critic.critic_layer.biases.len() {
            let mut grad = 0.0;
            for (j, t) in self.buffer.transitions.iter().enumerate() {
                let (_, _, value) = self.actor_critic.forward(&t.state);
                grad += 2.0 * (value - returns[j]);
            }
            grad /= self.buffer.len() as f64;
            self.actor_critic.critic_layer.biases[i] -= lr * self.config.value_coeff * grad;
        }

        // Perturb backbone layer1 biases with finite difference
        for i in 0..self.actor_critic.backbone_layer1.biases.len() {
            let original = self.actor_critic.backbone_layer1.biases[i];

            // Forward
            self.actor_critic.backbone_layer1.biases[i] = original + epsilon;
            let loss_plus = self.evaluate_total_loss(advantages, returns);

            // Backward
            self.actor_critic.backbone_layer1.biases[i] = original - epsilon;
            let loss_minus = self.evaluate_total_loss(advantages, returns);

            self.actor_critic.backbone_layer1.biases[i] = original;

            let grad = (loss_plus - loss_minus) / (2.0 * epsilon);
            self.actor_critic.backbone_layer1.biases[i] -= lr * grad;
        }
    }

    /// Evaluate the total loss for current parameters.
    fn evaluate_total_loss(&self, advantages: &[f64], returns: &[f64]) -> f64 {
        let mut total = 0.0;
        let old_log_probs: Vec<f64> = self.buffer.transitions.iter().map(|t| t.log_prob).collect();

        for (j, t) in self.buffer.transitions.iter().enumerate() {
            let (lp, v, ent) = self.actor_critic.evaluate_action(&t.state, &t.action_raw);
            let ratio = (lp - old_log_probs[j]).exp();
            let policy_loss = ppo_clipped_loss(ratio, advantages[j], self.config.clip_epsilon);
            let value_loss = (v - returns[j]).powi(2);
            total += policy_loss + self.config.value_coeff * value_loss - self.config.entropy_coeff * ent;
        }
        total / self.buffer.len() as f64
    }

    /// Run a full training loop.
    pub fn train(
        &mut self,
        env: &mut PortfolioEnv,
        num_iterations: usize,
        steps_per_iteration: usize,
    ) -> Vec<f64> {
        let mut episode_values = Vec::new();

        for iteration in 0..num_iterations {
            // Collect experience
            self.collect_experience(env, steps_per_iteration);

            // Update policy
            let (policy_loss, value_loss, kl) = self.update();

            episode_values.push(env.portfolio_value);

            if iteration % 10 == 0 {
                println!(
                    "Iteration {}: portfolio_value={:.2}, policy_loss={:.4}, value_loss={:.4}, kl={:.6}",
                    iteration, env.portfolio_value, policy_loss, value_loss, kl
                );
            }
        }

        episode_values
    }
}

// ============================================================================
// Bybit API Integration
// ============================================================================

/// Bybit kline response structure.
#[derive(Debug, Deserialize, Serialize)]
pub struct BybitResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitResult,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct BybitResult {
    pub symbol: Option<String>,
    pub category: Option<String>,
    pub list: Vec<Vec<String>>,
}

/// Parsed OHLCV candle data.
#[derive(Debug, Clone)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Fetch kline data from Bybit API.
///
/// # Arguments
/// * `symbol` - Trading pair (e.g., "BTCUSDT")
/// * `interval` - Candle interval (e.g., "60" for 1h, "D" for daily)
/// * `limit` - Number of candles to fetch (max 200)
pub fn fetch_bybit_klines(symbol: &str, interval: &str, limit: usize) -> Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );

    let client = reqwest::blocking::Client::new();
    let response: BybitResponse = client
        .get(&url)
        .header("User-Agent", "ppo-portfolio-trader/0.1")
        .send()?
        .json()?;

    if response.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", response.ret_msg);
    }

    let mut candles: Vec<Candle> = response
        .result
        .list
        .iter()
        .filter_map(|item| {
            if item.len() >= 6 {
                Some(Candle {
                    timestamp: item[0].parse().unwrap_or(0),
                    open: item[1].parse().unwrap_or(0.0),
                    high: item[2].parse().unwrap_or(0.0),
                    low: item[3].parse().unwrap_or(0.0),
                    close: item[4].parse().unwrap_or(0.0),
                    volume: item[5].parse().unwrap_or(0.0),
                })
            } else {
                None
            }
        })
        .collect();

    // Bybit returns newest first, reverse to chronological order
    candles.reverse();
    Ok(candles)
}

/// Fetch multi-asset data from Bybit.
///
/// # Arguments
/// * `symbols` - List of trading pairs
/// * `interval` - Candle interval
/// * `limit` - Number of candles per symbol
///
/// # Returns
/// Vector of close price series, one per symbol (aligned by index)
pub fn fetch_multi_asset_data(
    symbols: &[&str],
    interval: &str,
    limit: usize,
) -> Result<Vec<Vec<f64>>> {
    let mut all_prices = Vec::new();

    for symbol in symbols {
        let candles = fetch_bybit_klines(symbol, interval, limit)?;
        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        all_prices.push(closes);
    }

    // Align to shortest series
    let min_len = all_prices.iter().map(|p| p.len()).min().unwrap_or(0);
    for prices in &mut all_prices {
        prices.truncate(min_len);
    }

    Ok(all_prices)
}

// ============================================================================
// Performance Metrics
// ============================================================================

/// Compute the Sharpe ratio from a return series.
pub fn compute_sharpe_ratio(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let std = variance.sqrt();
    if std < 1e-10 {
        0.0
    } else {
        mean / std * (252.0_f64).sqrt() // Annualized assuming daily returns
    }
}

/// Compute portfolio returns from price data and weights over time.
pub fn compute_portfolio_returns(
    prices: &[Vec<f64>],
    weights_history: &[Array1<f64>],
) -> Vec<f64> {
    let num_assets = prices.len();
    let mut returns = Vec::new();

    for t in 1..weights_history.len().min(prices[0].len()) {
        let mut port_return = 0.0;
        for asset in 0..num_assets {
            if prices[asset][t - 1] > 0.0 {
                let asset_ret = (prices[asset][t] / prices[asset][t - 1]) - 1.0;
                port_return += weights_history[t - 1][asset] * asset_ret;
            }
        }
        returns.push(port_return);
    }

    returns
}

/// Compute equal-weight portfolio returns (benchmark).
pub fn compute_equal_weight_returns(prices: &[Vec<f64>]) -> Vec<f64> {
    let num_assets = prices.len();
    let weight = 1.0 / num_assets as f64;
    let min_len = prices.iter().map(|p| p.len()).min().unwrap_or(0);
    let mut returns = Vec::new();

    for t in 1..min_len {
        let mut port_return = 0.0;
        for asset in 0..num_assets {
            if prices[asset][t - 1] > 0.0 {
                let asset_ret = (prices[asset][t] / prices[asset][t - 1]) - 1.0;
                port_return += weight * asset_ret;
            }
        }
        returns.push(port_return);
    }

    returns
}

/// Generate synthetic price data for testing.
pub fn generate_synthetic_prices(num_assets: usize, num_steps: usize, seed: u64) -> Vec<Vec<f64>> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let mut all_prices = Vec::new();
    for _ in 0..num_assets {
        let mut prices = Vec::with_capacity(num_steps);
        let mut price = 100.0;
        let drift: f64 = 0.0001 + rng.gen_range(-0.0001..0.0002);
        let vol: f64 = 0.01 + rng.gen_range(0.0..0.02);

        for _ in 0..num_steps {
            prices.push(price);
            let ret = drift + vol * sample_standard_normal(&mut rng);
            price *= 1.0 + ret;
            price = price.max(1.0); // Floor to prevent negative prices
        }
        all_prices.push(prices);
    }

    all_prices
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let probs = softmax(&logits);

        // All probabilities positive
        assert!(probs.iter().all(|&p| p > 0.0));

        // Sum to 1
        let sum: f64 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Largest logit gets largest probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_gaussian_log_prob() {
        let x = Array1::from_vec(vec![0.0]);
        let mean = Array1::from_vec(vec![0.0]);
        let std = Array1::from_vec(vec![1.0]);

        let lp = gaussian_log_prob(&x, &mean, &std);
        let expected = -0.5 * (2.0 * PI).ln(); // log(1/sqrt(2*pi))
        assert!((lp - expected).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_entropy() {
        let std = Array1::from_vec(vec![1.0]);
        let entropy = gaussian_entropy(&std);
        let expected = 0.5 * (1.0 + (2.0 * PI).ln());
        assert!((entropy - expected).abs() < 1e-10);
    }

    #[test]
    fn test_gae_computation() {
        let rewards = vec![1.0, 1.0, 1.0, 1.0];
        let values = vec![0.5, 0.5, 0.5, 0.5, 0.0]; // One extra for bootstrap
        let gamma = 0.99;
        let lambda = 0.95;

        let (advantages, returns) = compute_gae(&rewards, &values, gamma, lambda);

        assert_eq!(advantages.len(), 4);
        assert_eq!(returns.len(), 4);

        // Advantages should decrease from first to last (for constant rewards)
        // because earlier states have more future reward
        assert!(advantages[0] > advantages[3]);

        // Returns should be positive (rewards are all positive)
        assert!(returns.iter().all(|&r| r > 0.0));
    }

    #[test]
    fn test_ppo_clipped_loss() {
        let clip_epsilon = 0.2;

        // When advantage is positive and ratio is within clip range
        let loss1 = ppo_clipped_loss(1.0, 1.0, clip_epsilon);
        assert!((loss1 - (-1.0)).abs() < 1e-10);

        // When ratio exceeds clip range (positive advantage)
        let loss2 = ppo_clipped_loss(1.5, 1.0, clip_epsilon);
        // Should be clipped to -(1+eps)*advantage
        assert!((loss2 - (-1.2)).abs() < 1e-10);

        // When advantage is negative
        let loss3 = ppo_clipped_loss(0.5, -1.0, clip_epsilon);
        // Clipped to -(1-eps)*advantage = -0.8 * (-1) = 0.8
        // Unclipped: -0.5 * (-1) = 0.5
        // min(0.5, 0.8) = 0.5, negated = ... actually min is taken before negation
        // ppo_clipped_loss = -min(0.5*(-1), 0.8*(-1)) = -min(-0.5, -0.8) = -(-0.8) = 0.8
        assert!((loss3 - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_portfolio_env() {
        let prices = generate_synthetic_prices(2, 200, 42);
        let mut env = PortfolioEnv::new(prices, 0.001, 20);

        let state = env.reset();
        assert_eq!(state.len(), env.state_dim);

        // Take a step with equal weights
        let weights = Array1::from_vec(vec![0.5, 0.5]);
        let (next_state, reward, done) = env.step(&weights);

        assert_eq!(next_state.len(), env.state_dim);
        assert!(!done); // Should not be done after one step
        // Reward should be finite
        assert!(reward.is_finite());
    }

    #[test]
    fn test_actor_critic_forward() {
        let state_dim = 8;
        let num_assets = 2;
        let hidden_size = 32;
        let ac = ActorCritic::new(state_dim, num_assets, hidden_size);

        let state = Array1::from_vec(vec![0.1, -0.2, 0.3, 0.01, 0.5, 0.5, 0.02, -0.01]);
        let (mean, std, value) = ac.forward(&state);

        assert_eq!(mean.len(), num_assets);
        assert_eq!(std.len(), num_assets);
        assert!(std.iter().all(|&s| s > 0.0)); // Std should be positive
        assert!(value.is_finite());
    }

    #[test]
    fn test_kl_divergence() {
        let mean1 = Array1::from_vec(vec![0.0, 0.0]);
        let std1 = Array1::from_vec(vec![1.0, 1.0]);
        let mean2 = Array1::from_vec(vec![0.0, 0.0]);
        let std2 = Array1::from_vec(vec![1.0, 1.0]);

        // KL divergence of identical distributions should be 0
        let kl = gaussian_kl_divergence(&mean1, &std1, &mean2, &std2);
        assert!(kl.abs() < 1e-10);

        // KL divergence of different distributions should be positive
        let mean3 = Array1::from_vec(vec![1.0, 1.0]);
        let kl2 = gaussian_kl_divergence(&mean1, &std1, &mean3, &std2);
        assert!(kl2 > 0.0);
    }

    #[test]
    fn test_sharpe_ratio() {
        // Positive mean with some variance
        let mut returns1 = Vec::new();
        for i in 0..100 {
            returns1.push(0.01 + 0.001 * (i as f64 % 3.0 - 1.0));
        }
        let sharpe1 = compute_sharpe_ratio(&returns1);
        assert!(sharpe1 > 1.0); // Should have positive Sharpe

        // Zero returns => zero Sharpe
        let returns2 = vec![0.0; 100];
        let sharpe2 = compute_sharpe_ratio(&returns2);
        assert!(sharpe2.abs() < 1e-10);

        // Mixed returns
        let returns3 = vec![0.01, -0.005, 0.02, -0.01, 0.015];
        let sharpe3 = compute_sharpe_ratio(&returns3);
        assert!(sharpe3.is_finite());
    }

    #[test]
    fn test_experience_buffer() {
        let mut buffer = ExperienceBuffer::new();
        assert!(buffer.is_empty());

        buffer.push(Transition {
            state: Array1::from_vec(vec![1.0, 2.0]),
            action_raw: Array1::from_vec(vec![0.1]),
            action_weights: Array1::from_vec(vec![1.0]),
            log_prob: -0.5,
            reward: 1.0,
            value: 0.8,
        });

        assert_eq!(buffer.len(), 1);
        assert!(!buffer.is_empty());

        buffer.clear();
        assert!(buffer.is_empty());
    }
}
