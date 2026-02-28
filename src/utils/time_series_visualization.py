"""Visualization utilities for time series predictions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path


def plot_trajectory(
    timepoints: List[int],
    predicted_deltas: np.ndarray,
    actual_deltas: Optional[np.ndarray] = None,
    baseline: Optional[float] = None,
    final_delta: Optional[float] = None,
    uncertainties: Optional[np.ndarray] = None,
    title: str = "Drug Response Trajectory",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot predicted trajectory over time.
    
    Args:
        timepoints: List of days
        predicted_deltas: Predicted delta values
        actual_deltas: Actual delta values (if available)
        baseline: Baseline value (to show absolute values)
        final_delta: Final delta target
        uncertainties: Uncertainty estimates (for confidence intervals)
        title: Plot title
        save_path: Path to save plot
        show: Whether to display plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Delta over time
    ax1 = axes[0]
    ax1.plot(timepoints, predicted_deltas, 'b-', linewidth=2, label='Predicted Delta', marker='o')
    
    if actual_deltas is not None:
        ax1.plot(timepoints, actual_deltas, 'r--', linewidth=2, label='Actual Delta', marker='s')
    
    if final_delta is not None:
        ax1.axhline(y=final_delta, color='g', linestyle=':', linewidth=2, label=f'Final Delta Target ({final_delta:.1f})')
    
    if uncertainties is not None:
        upper = predicted_deltas + 1.96 * uncertainties
        lower = predicted_deltas - 1.96 * uncertainties
        ax1.fill_between(timepoints, lower, upper, alpha=0.3, color='blue', label='95% Confidence Interval')
    
    ax1.set_xlabel('Days on Drug')
    ax1.set_ylabel('Delta (Change from Baseline)')
    ax1.set_title('Delta Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Absolute values (if baseline provided)
    if baseline is not None:
        ax2 = axes[1]
        absolute_predicted = baseline + predicted_deltas
        ax2.plot(timepoints, absolute_predicted, 'b-', linewidth=2, label='Predicted Value', marker='o')
        
        if actual_deltas is not None:
            absolute_actual = baseline + actual_deltas
            ax2.plot(timepoints, absolute_actual, 'r--', linewidth=2, label='Actual Value', marker='s')
        
        ax2.axhline(y=baseline, color='gray', linestyle='--', linewidth=1, label=f'Baseline ({baseline:.1f})')
        
        if uncertainties is not None:
            upper_abs = baseline + predicted_deltas + 1.96 * uncertainties
            lower_abs = baseline + predicted_deltas - 1.96 * uncertainties
            ax2.fill_between(timepoints, lower_abs, upper_abs, alpha=0.3, color='blue', label='95% CI')
        
        ax2.set_xlabel('Days on Drug')
        ax2.set_ylabel('Absolute Value')
        ax2.set_title('Absolute Value Trajectory')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        axes[1].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_multi_metric_trajectories(
    predictions: pd.DataFrame,
    patient_id: str,
    metric_names: List[str],
    timepoints: List[int],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot trajectories for multiple metrics simultaneously.
    
    Args:
        predictions: DataFrame with predictions (columns: day_10, day_20, etc.)
        patient_id: Patient ID to plot
        metric_names: List of metric names
        timepoints: List of timepoints
        save_path: Path to save plot
        show: Whether to display plot
    """
    n_metrics = len(metric_names)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    patient_data = predictions[predictions['patient_id'] == patient_id]
    
    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        
        # Extract trajectory
        trajectory = [patient_data[f'day_{day}'].values[0] for day in timepoints]
        
        ax.plot(timepoints, trajectory, 'b-', linewidth=2, marker='o')
        ax.set_xlabel('Days on Drug')
        ax.set_ylabel(f'{metric} Delta')
        ax.set_title(f'{metric} Trajectory')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Multi-Metric Trajectories - Patient {patient_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison_grid(
    predictions: pd.DataFrame,
    actuals: Optional[pd.DataFrame],
    metric_name: str,
    timepoints: List[int],
    n_samples: int = 9,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot a grid of trajectory comparisons for multiple patients.
    
    Args:
        predictions: DataFrame with predictions
        actuals: DataFrame with actual values (optional)
        metric_name: Metric to plot
        timepoints: List of timepoints
        n_samples: Number of patients to plot
        save_path: Path to save plot
        show: Whether to display plot
    """
    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    
    # Sample patients
    patient_ids = predictions['patient_id'].unique()[:n_samples]
    
    for idx, patient_id in enumerate(patient_ids):
        ax = axes[idx]
        
        patient_pred = predictions[predictions['patient_id'] == patient_id]
        pred_trajectory = [patient_pred[f'day_{day}'].values[0] for day in timepoints]
        
        ax.plot(timepoints, pred_trajectory, 'b-', linewidth=2, marker='o', label='Predicted')
        
        if actuals is not None:
            patient_actual = actuals[actuals['patient_id'] == patient_id]
            if len(patient_actual) > 0:
                actual_trajectory = [patient_actual[f'day_{day}'].values[0] for day in timepoints]
                ax.plot(timepoints, actual_trajectory, 'r--', linewidth=2, marker='s', label='Actual')
        
        ax.set_xlabel('Days')
        ax.set_ylabel('Delta')
        ax.set_title(f'Patient {patient_id}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(patient_ids), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'{metric_name} Trajectory Predictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_summary_report(
    predictions: pd.DataFrame,
    actuals: Optional[pd.DataFrame],
    timepoints: List[int],
    output_path: str
):
    """
    Create a summary report with statistics and visualizations.
    
    Args:
        predictions: DataFrame with predictions
        actuals: DataFrame with actual values (optional)
        timepoints: List of timepoints
        output_path: Path to save report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate statistics
    metrics = []
    for day in timepoints:
        col = f'day_{day}'
        if col in predictions.columns:
            pred_mean = predictions[col].mean()
            pred_std = predictions[col].std()
            
            metric = {
                'day': day,
                'predicted_mean': pred_mean,
                'predicted_std': pred_std,
            }
            
            if actuals is not None and col in actuals.columns:
                actual_mean = actuals[col].mean()
                actual_std = actuals[col].std()
                error = abs(pred_mean - actual_mean)
                
                metric.update({
                    'actual_mean': actual_mean,
                    'actual_std': actual_std,
                    'error': error,
                    'error_pct': (error / abs(actual_mean)) * 100 if actual_mean != 0 else 0
                })
            
            metrics.append(metric)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(metrics)
    
    # Save to CSV
    summary_path = output_path.with_suffix('.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(summary_df['day'], summary_df['predicted_mean'], 'b-', linewidth=2, marker='o', label='Predicted Mean')
    ax.fill_between(
        summary_df['day'],
        summary_df['predicted_mean'] - summary_df['predicted_std'],
        summary_df['predicted_mean'] + summary_df['predicted_std'],
        alpha=0.3, color='blue', label='Predicted ±1 SD'
    )
    
    if 'actual_mean' in summary_df.columns:
        ax.plot(summary_df['day'], summary_df['actual_mean'], 'r--', linewidth=2, marker='s', label='Actual Mean')
        ax.fill_between(
            summary_df['day'],
            summary_df['actual_mean'] - summary_df['actual_std'],
            summary_df['actual_mean'] + summary_df['actual_std'],
            alpha=0.3, color='red', label='Actual ±1 SD'
        )
    
    ax.set_xlabel('Days on Drug')
    ax.set_ylabel('Delta (Change from Baseline)')
    ax.set_title('Summary: Mean Trajectory with Uncertainty')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plot_path = output_path.with_suffix('.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")
    plt.close()

