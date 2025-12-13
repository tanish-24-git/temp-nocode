"""
Model comparison utilities.
"""
from typing import Dict, Any, List
from app.utils.logging import get_logger

logger = get_logger(__name__)


def compare_models(
    base_metrics: Dict[str, Any],
    finetuned_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare metrics between base and fine-tuned models.
    
    Args:
        base_metrics: Metrics from base model
        finetuned_metrics: Metrics from fine-tuned model
    
    Returns:
        Comparison dictionary with improvements
    """
    comparison = {
        "base": base_metrics,
        "finetuned": finetuned_metrics,
        "improvements": {}
    }
    
    # Calculate improvements for each metric
    for metric_name in finetuned_metrics:
        if metric_name in base_metrics:
            base_val = base_metrics[metric_name]
            ft_val = finetuned_metrics[metric_name]
            
            if isinstance(base_val, (int, float)) and isinstance(ft_val, (int, float)):
                # For perplexity, lower is better
                if metric_name == "perplexity":
                    improvement = ((base_val - ft_val) / base_val) * 100 if base_val > 0 else 0
                else:
                    # For other metrics, higher is better
                    improvement = ((ft_val - base_val) / base_val) * 100 if base_val > 0 else 0
                
                comparison["improvements"][metric_name] = {
                    "absolute": ft_val - base_val,
                    "relative_percent": improvement
                }
    
    return comparison


def generate_comparison_report(comparison: Dict[str, Any]) -> str:
    """
    Generate a Markdown comparison report.
    
    Args:
        comparison: Comparison dictionary from compare_models()
    
    Returns:
        Markdown formatted report
    """
    report = "# Model Comparison Report\n\n"
    
    report += "## Metrics Comparison\n\n"
    report += "| Metric | Base Model | Fine-tuned Model | Improvement |\n"
    report += "|--------|------------|------------------|-------------|\n"
    
    base = comparison["base"]
    finetuned = comparison["finetuned"]
    improvements = comparison["improvements"]
    
    for metric_name in sorted(finetuned.keys()):
        if metric_name in base:
            base_val = base[metric_name]
            ft_val = finetuned[metric_name]
            
            if metric_name in improvements:
                imp = improvements[metric_name]["relative_percent"]
                imp_str = f"+{imp:.2f}%" if imp > 0 else f"{imp:.2f}%"
            else:
                imp_str = "N/A"
            
            report += f"| {metric_name} | {base_val:.4f} | {ft_val:.4f} | {imp_str} |\n"
    
    report += "\n## Summary\n\n"
    
    # Count improvements
    positive_improvements = sum(1 for imp in improvements.values() if imp["relative_percent"] > 0)
    total_metrics = len(improvements)
    
    report += f"- **Metrics improved**: {positive_improvements}/{total_metrics}\n"
    
    if improvements:
        best_metric = max(improvements.items(), key=lambda x: x[1]["relative_percent"])
        report += f"- **Best improvement**: {best_metric[0]} ({best_metric[1]['relative_percent']:.2f}%)\n"
    
    return report
