"""
FREE Multi-Model Conversation Generator using Ollama & HuggingFace

Generates synthetic conversations using 100% FREE open-source models:
- Llama 2 / Llama 3 (Meta)
- Mistral (Mistral AI)
- Neural Chat (Intel)
- Orca (Microsoft)
- Phi (Microsoft)

ZERO API COSTS - All models run locally via Ollama

Usage:
    # Estimate (no costs)
    python multimodel_api_orchestrator.py --model-pairs llama_mistral neural_phi --estimate-only
    
    # Generate (completely free)
    python multimodel_api_orchestrator.py --model-pairs llama_mistral neural_phi --num-conversations 500

Requirements:
    1. Ollama: https://ollama.ai/download
    2. Models: ollama pull llama2 mistral neural-chat phi
    3. Python: pip install requests
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """Configuration for different LLM APIs."""
    
    model_name: str
    api_provider: str  # 'openai', 'anthropic', 'ollama'
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_id: Optional[str] = None
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    max_tokens: int = 1000
    temperature: float = 0.7
    

# Model configurations - ALL FREE MODELS
MODEL_CONFIGS = {
    'llama': APIConfig(
        model_name='llama',
        api_provider='ollama',
        base_url='http://localhost:11434',
        model_id='llama2',
        cost_per_1k_input=0.0,  # FREE
        cost_per_1k_output=0.0,
        temperature=0.7,
    ),
    'llama3': APIConfig(
        model_name='llama3',
        api_provider='ollama',
        base_url='http://localhost:11434',
        model_id='llama2:13b',  # Larger version for better quality
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        temperature=0.7,
    ),
    'mistral': APIConfig(
        model_name='mistral',
        api_provider='ollama',
        base_url='http://localhost:11434',
        model_id='mistral',
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        temperature=0.7,
    ),
    'neural': APIConfig(
        model_name='neural',
        api_provider='ollama',
        base_url='http://localhost:11434',
        model_id='neural-chat',  # Optimized for conversations
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        temperature=0.7,
    ),
    'phi': APIConfig(
        model_name='phi',
        api_provider='ollama',
        base_url='http://localhost:11434',
        model_id='phi',  # Microsoft's efficient model
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        temperature=0.7,
    ),
    'orca': APIConfig(
        model_name='orca',
        api_provider='ollama',
        base_url='http://localhost:11434',
        model_id='orca-mini',  # Smaller, faster variant
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        temperature=0.7,
    ),
}


class APIOrchestrator:
    """Orchestrate multi-model API calls for conversation generation."""
    
    CONVERSATION_PROMPTS = {
        "user_base": "You are participating in an AI discussion. Keep responses concise (2-3 sentences) and substantive. Topic: {topic}",
        "assistant_continuation": "Respond thoughtfully to the previous message and advance the conversation. Keep it concise.",
    }
    
    def __init__(self, budget_limit: float = 0.0, output_dir: Path = None):
        """
        Initialize API orchestrator (completely FREE - no API keys needed).
        
        Args:
            budget_limit: Not used for free models (kept for compatibility)
            output_dir: Directory to save conversations
        """
        self.budget_limit = budget_limit  # Not used
        self.budget_spent = 0.0  # Always zero
        self.output_dir = Path(output_dir or "data/synthetic")
        
        logger.info("✅ Using FREE open-source models via Ollama")
        logger.info("⚠️  Make sure Ollama is running: ollama serve")
        
        self.conversation_history = []
    
    def estimate_cost(self, model_a: str, model_b: str, num_conversations: int) -> Dict[str, float]:
        """
        Estimate cost for generating conversations.
        
        Returns: All zeros because using FREE models!
        """
        return {
            model_a: 0.0,
            model_b: 0.0,
            "total": 0.0,
        }
    
    def call_openai(self, model: str, system_prompt: str, user_message: str) -> Tuple[str, Dict[str, Any]]:
        """OpenAI API removed - using FREE Ollama instead."""
        raise NotImplementedError("Use FREE models instead: llama, mistral, neural, phi")
    
    def call_anthropic(self, model: str, system_prompt: str, user_message: str) -> Tuple[str, Dict[str, Any]]:
        """Anthropic API removed - using FREE Ollama instead."""
        raise NotImplementedError("Use FREE models instead: llama, mistral, neural, phi")
    
    def call_ollama(self, model: str, system_prompt: str, user_message: str) -> Tuple[str, Dict[str, Any]]:
        """
        Call local Ollama model.
        
        Returns:
            (response_text, usage_dict)
        """
        try:
            import requests
        except ImportError:
            logger.error("requests package not installed. Run: pip install requests")
            return "", {"error": "requests not installed"}
        
        config = MODEL_CONFIGS[model]
        url = f"{config.base_url}/api/generate"
        
        logger.debug(f"Calling Ollama {config.model_id}: {user_message[:50]}...")
        
        prompt = f"{system_prompt}\n\nUser: {user_message}\n\nAssistant:"
        
        try:
            response = requests.post(
                url,
                json={"model": config.model_id, "prompt": prompt, "stream": False},
                timeout=180,
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to call Ollama: {e}")
            logger.error(f"Make sure Ollama is running: ollama serve")
            return "", {"error": str(e)}
        
        result = response.json()
        text = result.get("response", "")
        
        usage = {
            "input_tokens": len(prompt.split()),
            "output_tokens": len(text.split()),
            "model": config.model_id,
        }
        
        return text, usage
    
    def call_model(self, model: str, system_prompt: str, user_message: str) -> Tuple[str, Dict[str, Any]]:
        """
        Call FREE Ollama model (only option - no API costs).
        """
        config = MODEL_CONFIGS.get(model)
        if not config:
            logger.error(f"Unknown model: {model}")
            return "", {}
        
        if config.api_provider != "ollama":
            logger.error(f"Only free Ollama models supported: {list(MODEL_CONFIGS.keys())}")
            return "", {}
        
        return self.call_ollama(model, system_prompt, user_message)
    
    def generate_conversation(
        self,
        model_a: str,
        model_b: str,
        topic: str,
        num_turns: int = 3,
    ) -> Dict[str, Any]:
        """
        Generate full conversation between two models.
        
        Args:
            model_a: First model
            model_b: Second model
            topic: Conversation topic
            num_turns: Number of back-and-forth turns
        
        Returns:
            Conversation object
        """
        
        conversation_id = f"synthetic_{model_a}_{model_b}_{int(time.time() * 1000)}"
        logger.info(f"Generating: {model_a} ↔ {model_b} | Turns: {num_turns}")
        
        messages = []
        context = ""
        
        system_prompt_a = f"You are an AI researcher ({model_a}). Be thoughtful and concise."
        system_prompt_b = f"You are an AI researcher ({model_b}). Be thoughtful and concise."
        
        for turn in range(num_turns):
            # Model A speaks
            if turn == 0:
                user_msg_a = f"{self.CONVERSATION_PROMPTS['user_base'].format(topic=topic)}\nContext: {context}"
            else:
                # Check if messages is not empty before accessing
                if messages:
                    user_msg_a = f"{self.CONVERSATION_PROMPTS['assistant_continuation']}\nPrevious: {messages[-1]['content']}"
                else:
                    # If no messages from previous turn, skip this turn
                    logger.warning(f"No previous messages, skipping turn {turn}")
                    continue
            
            response_a, usage_a = self.call_model(model_a, system_prompt_a, user_msg_a)
            
            if response_a:
                messages.append({
                    "turn": turn + 1,
                    "speaker": model_a,
                    "content": response_a,
                    "usage": usage_a,
                })
                context = response_a[:200]
            else:
                logger.warning(f"Model {model_a} returned empty response on turn {turn}")
            
            time.sleep(0.5)  # Rate limiting
            
            # Model B responds
            user_msg_b = f"{self.CONVERSATION_PROMPTS['assistant_continuation']}\nPrevious from {model_a}: {messages[-1]['content'] if messages else topic}"
            
            response_b, usage_b = self.call_model(model_b, system_prompt_b, user_msg_b)
            
            if response_b:
                messages.append({
                    "turn": turn + 1,
                    "speaker": model_b,
                    "content": response_b,
                    "usage": usage_b,
                })
                context = response_b[:200]
            
            time.sleep(0.5)
        
        conversation = {
            "conversation_id": conversation_id,
            "models": [model_a, model_b],
            "topic": topic,
            "num_turns": num_turns,
            "messages": messages,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "budget_spent": self.budget_spent,
        }
        
        return conversation


def main():
    parser = argparse.ArgumentParser(description="Generate FREE multi-model conversations (no API costs!)")
    parser.add_argument("--model-pairs", nargs="+", default=["llama_mistral", "neural_phi"], help="FREE model pairs")
    parser.add_argument("--num-conversations", type=int, default=100, help="Conversations per pair")
    parser.add_argument("--total-budget", type=float, default=0.0, help="Budget (not used for free models)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"), help="Output directory")
    parser.add_argument("--estimate-only", action="store_true", help="Only estimate (nothing to cost)")
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("FREE MULTI-MODEL CONVERSATION GENERATOR")
    logger.info("="*70)
    logger.info("✅ All models are FREE - running locally via Ollama")
    logger.info("⚠️  Make sure Ollama is running: ollama serve")
    logger.info("")
    
    orchestrator = APIOrchestrator(budget_limit=0.0, output_dir=args.output_dir)
    
    # Parse model pairs
    model_pairs = []
    valid_models = set(MODEL_CONFIGS.keys())
    
    for pair_str in args.model_pairs:
        if "_" in pair_str:
            a, b = pair_str.split("_", 1)
            if a not in valid_models or b not in valid_models:
                logger.error(f"Invalid model pair: {pair_str}")
                logger.error(f"Valid models: {', '.join(sorted(valid_models))}")
                continue
            model_pairs.append((a, b))
    
    if not model_pairs:
        logger.error("No valid model pairs specified")
        logger.info(f"Available: {', '.join(sorted(valid_models))}")
        return
    
    # Cost estimation (all free)
    logger.info("Cost Estimation:")
    logger.info("-" * 70)
    total_estimated_cost = 0.0
    for model_a, model_b in model_pairs:
        cost = orchestrator.estimate_cost(model_a, model_b, args.num_conversations)
        logger.info(f"{model_a:10} ↔ {model_b:10} | {args.num_conversations:4} convos | Cost: FREE! 🎉")
    
    logger.info("-" * 70)
    logger.info(f"{'TOTAL COST':30} FREE! (Completely open-source)")
    logger.info(f"{'OLLAMA SETUP TIME':30} ~5 minutes per model")
    logger.info("-" * 70)
    
    if args.estimate_only:
        logger.info("\n✓ Cost estimation complete (--estimate-only flag set)")
        logger.info("\nReady to generate? Run without --estimate-only flag")
        return
    
    # Generate conversations
    logger.info("\nGenerating Conversations:")
    logger.info("-" * 70)
    
    all_conversations = []
    for model_a, model_b in model_pairs:
        pair_conversations = []
        for i in range(args.num_conversations):
            topics = [
                "sentiment analysis challenges",
                "preprocessing importance",
                "ensemble vs single models",
                "robustness validation",
                "AI ethics in discourse",
            ]
            topic = topics[i % len(topics)]
            
            conv = orchestrator.generate_conversation(model_a, model_b, topic, num_turns=3)
            pair_conversations.append(conv)
            all_conversations.append(conv)
            
            logger.info(f"  [{i+1}/{args.num_conversations}] {model_a}-{model_b} | Cost: FREE 💰")
    
    # Save conversations
    output_dir = Path(args.output_dir) / "conversations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"multimodel_conversations_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    with open(output_file, 'w') as f:
        for conv in all_conversations:
            f.write(json.dumps(conv) + '\n')
    
    logger.info(f"\n✓ Saved {len(all_conversations)} conversations to {output_file}")
    logger.info(f"✓ Total API cost: $0.00 (100% FREE!)")
    logger.info("="*70)


if __name__ == "__main__":
    main()
