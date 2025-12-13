"""
Evaluation metrics computation.
Supports BLEU, ROUGE, perplexity, and accuracy metrics.
"""
import torch
import numpy as np
from typing import List, Dict, Any
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from app.utils.logging import get_logger

logger = get_logger(__name__)


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """
    Compute BLEU score.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
    
    Returns:
        BLEU score (0-100)
    """
    try:
        # sacrebleu expects references as list of lists
        refs = [[ref] for ref in references]
        score = corpus_bleu(predictions, refs)
        return score.score
    except Exception as e:
        logger.error("BLEU computation failed", error=str(e))
        return 0.0


def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
    
    Returns:
        Dictionary with ROUGE scores
    """
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        
        for pred, ref in zip(predictions, references):
            score = scorer.score(ref, pred)
            scores['rouge1'].append(score['rouge1'].fmeasure)
            scores['rouge2'].append(score['rouge2'].fmeasure)
            scores['rougeL'].append(score['rougeL'].fmeasure)
        
        # Average scores
        return {
            'rouge1': np.mean(scores['rouge1']),
            'rouge2': np.mean(scores['rouge2']),
            'rougeL': np.mean(scores['rougeL'])
        }
    except Exception as e:
        logger.error("ROUGE computation failed", error=str(e))
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}


def compute_perplexity(model, tokenizer, texts: List[str], device: str = "cpu") -> float:
    """
    Compute perplexity on a list of texts.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        texts: List of texts to evaluate
        device: Device to use ('cpu' or 'cuda')
    
    Returns:
        Perplexity score
    """
    try:
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(device)
                
                # Get loss
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                # Accumulate
                num_tokens = inputs["input_ids"].size(1)
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        # Compute perplexity
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    except Exception as e:
        logger.error("Perplexity computation failed", error=str(e))
        return float('inf')


def compute_accuracy(predictions: List[str], references: List[str]) -> float:
    """
    Compute exact match accuracy.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
    
    Returns:
        Accuracy score (0-1)
    """
    try:
        correct = sum(1 for pred, ref in zip(predictions, references) if pred.strip() == ref.strip())
        accuracy = correct / len(predictions) if predictions else 0.0
        return accuracy
    except Exception as e:
        logger.error("Accuracy computation failed", error=str(e))
        return 0.0


def compute_all_metrics(
    model,
    tokenizer,
    predictions: List[str],
    references: List[str],
    device: str = "cpu",
    compute_ppl: bool = True
) -> Dict[str, Any]:
    """
    Compute all available metrics.
    
    Args:
        model: Language model (for perplexity)
        tokenizer: Tokenizer (for perplexity)
        predictions: List of predicted texts
        references: List of reference texts
        device: Device to use
        compute_ppl: Whether to compute perplexity (expensive)
    
    Returns:
        Dictionary with all metrics
    """
    logger.info("Computing evaluation metrics", num_samples=len(predictions))
    
    metrics = {}
    
    # BLEU
    logger.info("Computing BLEU score")
    metrics['bleu'] = compute_bleu(predictions, references)
    
    # ROUGE
    logger.info("Computing ROUGE scores")
    rouge_scores = compute_rouge(predictions, references)
    metrics.update(rouge_scores)
    
    # Accuracy
    logger.info("Computing accuracy")
    metrics['accuracy'] = compute_accuracy(predictions, references)
    
    # Perplexity (optional, expensive)
    if compute_ppl and model is not None:
        logger.info("Computing perplexity")
        metrics['perplexity'] = compute_perplexity(model, tokenizer, references, device)
    
    logger.info("Metrics computed", metrics=metrics)
    
    return metrics
