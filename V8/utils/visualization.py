import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64

class VisualizationUtils:
    """Utility class for creating visualizations"""
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names=None, title="Confusion Matrix"):
        """Plot confusion matrix"""
        if class_names is None:
            class_names = ['Bad', 'Moderate', 'Good']
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def plot_model_comparison(results_dict, metric='accuracy'):
        """Compare multiple models"""
        models = list(results_dict.keys())
        values = [results_dict[m][metric] for m in models]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, values, color='skyblue', edgecolor='navy')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title(f'Model Comparison - {metric.capitalize()}', fontsize=16, fontweight='bold')
        plt.xlabel('Models', fontsize=12)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def plot_training_history(history_dict, model_name="Model"):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Accuracy
        axes[0, 0].plot(history_dict['accuracy'], label='Train', linewidth=2)
        if 'val_accuracy' in history_dict:
            axes[0, 0].plot(history_dict['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history_dict['loss'], label='Train', linewidth=2)
        if 'val_loss' in history_dict:
            axes[0, 1].plot(history_dict['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate if available
        if 'lr' in history_dict:
            axes[1, 0].plot(history_dict['lr'], linewidth=2, color='green')
            axes[1, 0].set_title('Learning Rate', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Precision/Recall/F1 if available
        if 'precision' in history_dict:
            axes[1, 1].plot(history_dict['precision'], label='Precision', linewidth=2)
            axes[1, 1].plot(history_dict['recall'], label='Recall', linewidth=2)
            axes[1, 1].plot(history_dict['f1_score'], label='F1-Score', linewidth=2)
            axes[1, 1].set_title('Precision/Recall/F1', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_feature_importance(feature_names, importance_scores, title="Feature Importance", top_n=20):
        """Plot feature importance"""
        # Sort by importance
        indices = np.argsort(importance_scores)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importance_scores[indices], color='teal')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance Score')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def plot_class_distribution(y, class_names=None, title="Class Distribution"):
        """Plot class distribution"""
        if class_names is None:
            class_names = ['Bad', 'Moderate', 'Good']
        
        unique, counts = np.unique(y, return_counts=True)
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar([class_names[i] for i in unique], counts, 
                       color=['#ff6b6b', '#feca57', '#48dbfb'])
        
        # Add count labels
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def create_interactive_confusion_matrix(y_true, y_pred, class_names=None):
        """Create interactive confusion matrix with plotly"""
        if class_names is None:
            class_names = ['Bad', 'Moderate', 'Good']
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            width=600,
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_radar_chart(metrics_dict, model_name="Model"):
        """Create radar chart for multiple metrics"""
        categories = list(metrics_dict.keys())
        values = list(metrics_dict.values())
        
        # Close the polygon
        categories = categories + [categories[0]]
        values = values + [values[0]]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=model_name,
            line_color='royalblue'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=f'{model_name} - Performance Radar',
            width=600,
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_performance_over_time(timestamps, scores, title="Performance Over Time"):
        """Plot performance metrics over time"""
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, scores, marker='o', linewidth=2, markersize=8)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def create_heatmap(data, x_labels, y_labels, title="Heatmap"):
        """Create heatmap"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(data, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=x_labels, yticklabels=y_labels)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return plt.gcf()
    
    @staticmethod
    def fig_to_base64(fig):
        """Convert matplotlib figure to base64 string for web display"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str
    
    @staticmethod
    def generate_classification_report(y_true, y_pred, class_names=None):
        """Generate formatted classification report"""
        if class_names is None:
            class_names = ['Bad', 'Moderate', 'Good']
        
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Convert to DataFrame for better display
        df_report = pd.DataFrame(report).transpose()
        
        return df_report