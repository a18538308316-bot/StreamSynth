#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training visualization and analysis module.
Used to generate line charts and tables for rewards and losses.
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Configure fonts and plotting style (fallback to English fonts if needed)
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    USE_CHINESE = True
except:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    USE_CHINESE = False
sns.set_style("whitegrid")

class TrainingVisualizer:
    """Training process visualizer."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Data storage for training metrics
        self.training_metrics = {
            'steps': [],
            'total_rewards': [],
            'losses': [],
            'sample_rewards': {
                'sentiment_consistency': [],
                'attribute_compliance': [],
                'length_compliance': []
            },
            'batch_rewards': {
                'yelp_semantic_diversity': [],
                'inter_sample_diversity': []
            }
        }
        
    def record_reward_data(self, step: int, reward_type: str, rewards, completions):
        """Record detailed data for reward functions."""
        import torch
        
        # Ensure the current step exists in the records
        if step not in self.training_metrics['steps']:
            self.training_metrics['steps'].append(step)
            # Fill default values for other metrics when needed
            if len(self.training_metrics['total_rewards']) < len(self.training_metrics['steps']):
                self.training_metrics['total_rewards'].append(0.0)
            if len(self.training_metrics['losses']) < len(self.training_metrics['steps']):
                self.training_metrics['losses'].append(0.0)
        
        # Compute reward statistics
        mean_reward = float(torch.mean(rewards))
        
        # Record data according to reward type
        if reward_type in self.training_metrics['sample_rewards']:
            # For sample-level rewards, update if step exists, otherwise append
            step_index = self.training_metrics['steps'].index(step)
            if step_index < len(self.training_metrics['sample_rewards'][reward_type]):
                self.training_metrics['sample_rewards'][reward_type][step_index] = mean_reward
            else:
                self.training_metrics['sample_rewards'][reward_type].append(mean_reward)
                
        elif reward_type in self.training_metrics['batch_rewards']:
            # For batch-level rewards
            step_index = self.training_metrics['steps'].index(step)
            if step_index < len(self.training_metrics['batch_rewards'][reward_type]):
                self.training_metrics['batch_rewards'][reward_type][step_index] = mean_reward
            else:
                self.training_metrics['batch_rewards'][reward_type].append(mean_reward)
        
        # Print debug information
        print(f"📊 Record reward data - Step {step}, Type: {reward_type}, Mean: {mean_reward:.4f}")
        
    def log_step_metrics(self, step: int, metrics: Dict[str, Any]):
        """Record per-step training metrics."""
        self.training_metrics['steps'].append(step)
        
        # Total rewards and losses
        self.training_metrics['total_rewards'].append(metrics.get('total_reward', 0.0))
        self.training_metrics['losses'].append(metrics.get('loss', 0.0))
        
        # Sample-level rewards
        sample_rewards = metrics.get('sample_rewards', {})
        for reward_type in self.training_metrics['sample_rewards']:
            value = sample_rewards.get(reward_type, 0.0)
            self.training_metrics['sample_rewards'][reward_type].append(value)
        
        # Batch-level rewards
        batch_rewards = metrics.get('batch_rewards', {})
        for reward_type in self.training_metrics['batch_rewards']:
            value = batch_rewards.get(reward_type, 0.0)
            self.training_metrics['batch_rewards'][reward_type].append(value)
    
    def create_reward_plots(self):
        """Create line plots for reward changes during training."""
        steps = self.training_metrics['steps']
        if not steps:
            print("⚠️ No training data, skip visualization.")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🎯 GRPO training reward trends', fontsize=16, fontweight='bold')
        
        # 1. Total reward trend
        ax1 = axes[0, 0]
        ax1.plot(steps, self.training_metrics['total_rewards'], 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_title('Total reward change', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Training steps')
        ax1.set_ylabel('Total reward')
        ax1.grid(True, alpha=0.3)
        
        # 2. Sample-level rewards
        ax2 = axes[0, 1]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for i, (reward_type, values) in enumerate(self.training_metrics['sample_rewards'].items()):
            if values:  # Only plot rewards that have data
                ax2.plot(steps, values, color=colors[i % len(colors)], linewidth=2, 
                        marker='o', markersize=3, label=reward_type.replace('_', ' ').title())
        ax2.set_title('Sample-level rewards', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Training steps')
        ax2.set_ylabel('Reward value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Batch-level rewards
        ax3 = axes[1, 0]
        colors = ['#96CEB4', '#FFEAA7']
        for i, (reward_type, values) in enumerate(self.training_metrics['batch_rewards'].items()):
            if values:  # Only plot rewards that have data
                ax3.plot(steps, values, color=colors[i % len(colors)], linewidth=2,
                        marker='s', markersize=3, label=reward_type.replace('_', ' ').title())
        ax3.set_title('Batch-level rewards', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Training steps')
        ax3.set_ylabel('Reward value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Loss change
        ax4 = axes[1, 1]
        if self.training_metrics['losses']:
            ax4.plot(steps, self.training_metrics['losses'], 'r-', linewidth=2, marker='x', markersize=4)
            ax4.set_title('Training loss change', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Training steps')
            ax4.set_ylabel('Loss value')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No loss data yet', ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Training loss change', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plots
        plot_path = self.output_dir / 'training_rewards_trends.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Reward trend plots saved to: {plot_path}")
        plt.close()
    
    def create_summary_table(self):
        """Create a summary table for training metrics."""
        if not self.training_metrics['steps']:
            print("⚠️ No training data, skip summary table generation.")
            return
        
        # Prepare data
        summary_data = []
        
        # Basic statistics
        total_steps = len(self.training_metrics['steps'])
        if total_steps > 0:
            final_total_reward = self.training_metrics['total_rewards'][-1] if self.training_metrics['total_rewards'] else 0
            avg_total_reward = np.mean(self.training_metrics['total_rewards']) if self.training_metrics['total_rewards'] else 0
            final_loss = self.training_metrics['losses'][-1] if self.training_metrics['losses'] else 0
            avg_loss = np.mean(self.training_metrics['losses']) if self.training_metrics['losses'] else 0
            
            summary_data.append({
                'metric_type': 'training_overview',
                'metric_name': 'total_steps',
                'final_value': total_steps,
                'average_value': total_steps,
                'max_value': total_steps,
                'min_value': total_steps
            })
            
            summary_data.append({
                'metric_type': 'overall_reward',
                'metric_name': 'total_reward',
                'final_value': f"{final_total_reward:.4f}",
                'average_value': f"{avg_total_reward:.4f}",
                'max_value': f"{max(self.training_metrics['total_rewards']):.4f}" if self.training_metrics['total_rewards'] else "0",
                'min_value': f"{min(self.training_metrics['total_rewards']):.4f}" if self.training_metrics['total_rewards'] else "0"
            })
            
            if self.training_metrics['losses']:
                summary_data.append({
                    'metric_type': 'training_loss',
                    'metric_name': 'loss',
                    'final_value': f"{final_loss:.4f}",
                    'average_value': f"{avg_loss:.4f}",
                    'max_value': f"{max(self.training_metrics['losses']):.4f}",
                    'min_value': f"{min(self.training_metrics['losses']):.4f}"
                })
        
        # Sample-level reward statistics
        for reward_type, values in self.training_metrics['sample_rewards'].items():
            if values:
                summary_data.append({
                    'metric_type': 'sample_reward',
                    'metric_name': reward_type.replace('_', ' ').title(),
                    'final_value': f"{values[-1]:.4f}",
                    'average_value': f"{np.mean(values):.4f}",
                    'max_value': f"{max(values):.4f}",
                    'min_value': f"{min(values):.4f}"
                })
        
        # Batch-level reward statistics
        for reward_type, values in self.training_metrics['batch_rewards'].items():
            if values:
                summary_data.append({
                    'metric_type': 'batch_reward',
                    'metric_name': reward_type.replace('_', ' ').title(),
                    'final_value': f"{values[-1]:.4f}",
                    'average_value': f"{np.mean(values):.4f}",
                    'max_value': f"{max(values):.4f}",
                    'min_value': f"{min(values):.4f}"
                })
        
        # Create DataFrame
        df = pd.DataFrame(summary_data)
        
        # Save as CSV
        csv_path = self.output_dir / 'training_summary.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"📋 Training summary CSV saved to: {csv_path}")
        
        # Save as HTML
        html_path = self.output_dir / 'training_summary.html'
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>GRPO Training Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2E86AB; text-align: center; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #e8f4fd; }}
                .metric-type {{ font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>🎯 GRPO Training Summary Report</h1>
            <p><strong>Generated at:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            {df.to_html(index=False, escape=False, classes='summary-table')}
        </body>
        </html>
        """
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"🌐 Training summary HTML saved to: {html_path}")
        
        return df
    
    def create_detailed_metrics_plot(self):
        """Create detailed comparison plots for training metrics."""
        if not self.training_metrics['steps']:
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        fig.suptitle('📈 GRPO detailed metric analysis', fontsize=16, fontweight='bold')
        
        steps = self.training_metrics['steps']
        
        # 1. Comparison of all sample-level rewards
        ax1 = axes[0]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for i, (reward_type, values) in enumerate(self.training_metrics['sample_rewards'].items()):
            if values:
                ax1.plot(steps, values, color=colors[i % len(colors)], linewidth=2,
                        marker='o', markersize=3, label=reward_type)
        ax1.set_title('Detailed comparison of sample-level rewards', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Training steps')
        ax1.set_ylabel('Reward value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Comparison of all batch-level rewards
        ax2 = axes[1]
        colors = ['#96CEB4', '#FFEAA7']
        for i, (reward_type, values) in enumerate(self.training_metrics['batch_rewards'].items()):
            if values:
                ax2.plot(steps, values, color=colors[i % len(colors)], linewidth=2,
                        marker='s', markersize=3, label=reward_type)
        ax2.set_title('Detailed comparison of batch-level rewards', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Training steps')
        ax2.set_ylabel('Reward value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Dual Y-axis: total reward vs training loss
        ax3 = axes[2]
        ax3_twin = ax3.twinx()
        
        if self.training_metrics['total_rewards']:
            line1 = ax3.plot(steps, self.training_metrics['total_rewards'], 'b-', linewidth=2,
                            marker='o', markersize=4, label='Total reward')
            ax3.set_ylabel('Total reward', color='b')
            ax3.tick_params(axis='y', labelcolor='b')
        
        if self.training_metrics['losses']:
            line2 = ax3_twin.plot(steps, self.training_metrics['losses'], 'r-', linewidth=2,
                                 marker='x', markersize=4, label='Training loss')
            ax3_twin.set_ylabel('Training loss', color='r')
            ax3_twin.tick_params(axis='y', labelcolor='r')
        
        ax3.set_title('Total reward vs training loss', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Training steps')
        ax3.grid(True, alpha=0.3)
        
        # Add legend
        lines1 = ax3.get_lines() if hasattr(ax3, 'get_lines') else []
        lines2 = ax3_twin.get_lines() if hasattr(ax3_twin, 'get_lines') else []
        labels1 = [l.get_label() for l in lines1]
        labels2 = [l.get_label() for l in lines2]
        if lines1 or lines2:
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'detailed_metrics_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Detailed metric analysis plot saved to: {plot_path}")
        plt.close()
    
    def save_raw_data(self):
        """Save raw training data to JSON."""
        raw_data_path = self.output_dir / 'raw_training_metrics.json'
        with open(raw_data_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_metrics, f, indent=2, ensure_ascii=False)
        print(f"💾 Raw training data saved to: {raw_data_path}")
    
    def generate_final_report(self):
        """Generate the final training report with plots and tables."""
        print("\n" + "="*80)
        print("🎉 Generating training visualization report...")
        print("="*80)
        
        # Generate all visualizations and summary tables
        self.create_reward_plots()
        self.create_detailed_metrics_plot()
        summary_df = self.create_summary_table()
        self.save_raw_data()
        
        print("\n📊 Training visualization report generation complete!")
        print(f"📁 All files saved to: {self.output_dir}")
        print("📋 Included files:")
        print("   - training_rewards_trends.png: reward trends plot")
        print("   - detailed_metrics_analysis.png: detailed metrics analysis plot")
        print("   - training_summary.csv: training summary table (CSV)")
        print("   - training_summary.html: training summary report (HTML)")
        print("   - raw_training_metrics.json: raw training metrics data")
        
        # Print brief statistics
        if summary_df is not None and not summary_df.empty:
            print("\n📈 Brief training statistics:")
            total_reward_row = summary_df[summary_df['metric_name'] == 'total_reward']
            if not total_reward_row.empty:
                print(f"   Final total reward: {total_reward_row.iloc[0]['final_value']}")
                print(f"   Average total reward: {total_reward_row.iloc[0]['average_value']}")
        
        print("="*80)

# Global visualizer instance
training_visualizer = None

def initialize_visualizer(output_dir: str):
    """Initialize the global training visualizer."""
    global training_visualizer
    training_visualizer = TrainingVisualizer(output_dir)
    print(f"✅ Training visualizer initialized, output directory: {output_dir}")
    return training_visualizer

def log_training_step(step: int, metrics: Dict[str, Any]):
    """Record metrics for a single training step."""
    global training_visualizer
    if training_visualizer:
        training_visualizer.log_step_metrics(step, metrics)

def generate_final_training_report():
    """Generate the final training report via the global visualizer."""
    global training_visualizer
    if training_visualizer:
        training_visualizer.generate_final_report()
    else:
        print("⚠️ Training visualizer is not initialized, skip report generation.")