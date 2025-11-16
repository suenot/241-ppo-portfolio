//! PPO Portfolio Trading Example
//!
//! Demonstrates PPO-based portfolio management with multi-asset data.
//! Uses synthetic data by default, with optional Bybit live data.

use ndarray::Array1;
use ppo_portfolio::*;

fn main() {
    println!("=== PPO Portfolio Trading ===\n");

    // Try to fetch live data from Bybit, fall back to synthetic
    let (prices, asset_names) = match fetch_live_data() {
        Ok((p, names)) => {
            println!("Using live Bybit data for BTC and ETH");
            (p, names)
        }
        Err(e) => {
            println!("Could not fetch Bybit data ({}), using synthetic data", e);
            let prices = generate_synthetic_prices(2, 500, 42);
            (prices, vec!["Asset_A".to_string(), "Asset_B".to_string()])
        }
    };

    let num_assets = prices.len();
    let data_length = prices.iter().map(|p| p.len()).min().unwrap_or(0);
    println!(
        "Assets: {:?}, Data points: {}\n",
        asset_names, data_length
    );

    // Print price summary
    for (i, name) in asset_names.iter().enumerate() {
        let first = prices[i].first().unwrap_or(&0.0);
        let last = prices[i].last().unwrap_or(&0.0);
        let ret = if *first > 0.0 {
            (last / first - 1.0) * 100.0
        } else {
            0.0
        };
        println!("{}: start={:.2}, end={:.2}, return={:.2}%", name, first, last, ret);
    }
    println!();

    // Split data into train and test sets
    let split_point = (data_length as f64 * 0.7) as usize;
    let train_prices: Vec<Vec<f64>> = prices.iter().map(|p| p[..split_point].to_vec()).collect();
    let test_prices: Vec<Vec<f64>> = prices.iter().map(|p| p[split_point..].to_vec()).collect();

    println!(
        "Train: {} steps, Test: {} steps\n",
        split_point,
        data_length - split_point
    );

    // Create environment and PPO trainer
    let lookback = 20;
    let transaction_cost = 0.001; // 0.1% per trade
    let mut train_env = PortfolioEnv::new(train_prices.clone(), transaction_cost, lookback);

    let state_dim = train_env.state_dim;
    let hidden_size = 64;
    let config = PPOConfig {
        clip_epsilon: 0.2,
        gamma: 0.99,
        gae_lambda: 0.95,
        value_coeff: 0.5,
        entropy_coeff: 0.01,
        learning_rate: 1e-3,
        num_epochs: 4,
        minibatch_size: 64,
        max_kl: 0.02,
    };

    let mut trainer = PPOTrainer::new(state_dim, num_assets, hidden_size, config);

    // Training
    println!("--- Training PPO Agent ---");
    let num_iterations = 50;
    let steps_per_iteration = 100;
    let portfolio_values = trainer.train(&mut train_env, num_iterations, steps_per_iteration);

    println!(
        "\nFinal training portfolio value: {:.2}",
        portfolio_values.last().unwrap_or(&0.0)
    );

    // Evaluation on test data
    println!("\n--- Evaluating on Test Data ---");
    let mut test_env = PortfolioEnv::new(test_prices.clone(), transaction_cost, lookback);
    let mut state = test_env.reset();
    let mut weights_history: Vec<Array1<f64>> = Vec::new();
    let initial_weight = 1.0 / num_assets as f64;
    weights_history.push(Array1::from_elem(num_assets, initial_weight));

    let test_steps = test_env.total_steps();
    for _ in 0..test_steps {
        let (weights, _, _) = trainer.actor_critic.sample_action(&state);
        let (next_state, _, done) = test_env.step(&weights);
        weights_history.push(weights);
        if done {
            break;
        }
        state = next_state;
    }

    // Compare PPO vs Equal-Weight
    println!("\n--- Portfolio Weight Evolution (last 10 steps) ---");
    let start = if weights_history.len() > 10 {
        weights_history.len() - 10
    } else {
        0
    };
    for (i, w) in weights_history[start..].iter().enumerate() {
        let step = start + i;
        let weight_strs: Vec<String> = asset_names
            .iter()
            .enumerate()
            .map(|(j, name)| format!("{}={:.3}", name, w[j]))
            .collect();
        println!("  Step {}: [{}]", step, weight_strs.join(", "));
    }

    // Compute Sharpe ratios
    let ppo_returns = compute_portfolio_returns(&test_prices, &weights_history);
    let equal_returns = compute_equal_weight_returns(&test_prices);

    let ppo_sharpe = compute_sharpe_ratio(&ppo_returns);
    let equal_sharpe = compute_sharpe_ratio(&equal_returns);

    let ppo_total_return: f64 = ppo_returns
        .iter()
        .fold(1.0, |acc, r| acc * (1.0 + r))
        - 1.0;
    let equal_total_return: f64 = equal_returns
        .iter()
        .fold(1.0, |acc, r| acc * (1.0 + r))
        - 1.0;

    println!("\n--- Performance Comparison ---");
    println!(
        "PPO Portfolio:   Total Return={:.4}%, Sharpe Ratio={:.4}",
        ppo_total_return * 100.0,
        ppo_sharpe
    );
    println!(
        "Equal Weight:    Total Return={:.4}%, Sharpe Ratio={:.4}",
        equal_total_return * 100.0,
        equal_sharpe
    );

    if ppo_sharpe > equal_sharpe {
        println!("\nPPO agent outperformed equal-weight baseline in Sharpe ratio!");
    } else {
        println!("\nEqual-weight baseline performed better. PPO may need more training.");
    }

    println!("\n=== Done ===");
}

/// Attempt to fetch live data from Bybit API.
fn fetch_live_data() -> anyhow::Result<(Vec<Vec<f64>>, Vec<String>)> {
    let symbols = vec!["BTCUSDT", "ETHUSDT"];
    let interval = "60"; // 1-hour candles
    let limit = 200;

    let prices = fetch_multi_asset_data(&symbols, interval, limit)?;

    if prices.iter().any(|p| p.is_empty()) {
        anyhow::bail!("Empty price data received");
    }

    let names = symbols.iter().map(|s| s.to_string()).collect();
    Ok((prices, names))
}
