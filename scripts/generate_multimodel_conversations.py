"""
Multi-Model Conversation Generator for MoltBook Sentiment Research

Generates synthetic conversations between different LLMs (GPT-4, Claude, Llama)
to create diverse, realistic AI-to-AI dialogue data for robustness validation.

Usage:
    python generate_multimodel_conversations.py \
        --model-pairs gpt4_claude gpt4_llama claude_mistral \
        --conversations-per-pair 500 \
        --output-dir data/synthetic

Requirements:
    - OpenAI API key: OPENAI_API_KEY
    - Anthropic API key: ANTHROPIC_API_KEY (for Claude)
    - (Optional) Ollama running locally for Llama/Mistral
"""

import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiModelConversationGenerator:
    """Generate conversations between multiple LLM models."""
    
    # Conversation topics for diversity
    TOPICS = [
        "Discuss the challenges of sentiment analysis in social media conversations",
        "Debate whether neutral sentiment should be a distinct class or collapsed with positive",
        "Discuss trade-offs between rule-based and ML-based sentiment analysis approaches",
        "Explain how preprocessing affects sentiment analysis robustness",
        "Discuss bias and fairness issues in multi-agent AI systems",
        "Evaluate different approaches to handling sarcasm in text",
        "Discuss sentiment dynamics in threaded conversations",
        "Debate the value of ensemble vs single-model sentiment scoring",
        "Explain how to validate robustness of NLP pipelines",
        "Discuss emerging patterns in AI-to-AI discourse communities",
    ]
    
    # Model-specific personas to encourage natural variation
    PERSONAS = {
        'gpt4': "You are a precise, analytical AI researcher. Use formal language, cite methodologies, and maintain academic rigor.",
        'claude': "You are a thoughtful AI assistant. Be nuanced, acknowledge limitations, and consider ethical implications.",
        'llama': "You are a straightforward AI. Be direct, practical, and focus on implementation details.",
        'mistral': "You are a resourceful AI. Offer creative solutions and alternative perspectives.",
    }
    
    def __init__(self, output_dir: Path = None):
        """Initialize conversation generator.
        
        Args:
            output_dir: Base directory for saving conversations
        """
        self.output_dir = Path(output_dir or "data/synthetic")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata = {
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "model_pairs": {},
            "total_conversations": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }
    
    def generate_pair_conversation(
        self,
        model_a: str,
        model_b: str,
        topic: str,
        num_turns: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate conversation between two model pairs.
        
        Args:
            model_a: First model name (gpt4, claude, llama, mistral)
            model_b: Second model name
            topic: Conversation topic
            num_turns: Number of conversation turns (each turn = 2 messages)
        
        Returns:
            Conversation object with metadata
        """
        
        conversation_id = f"synthetic_{model_a}_{model_b}_{int(time.time())}"
        
        logger.info(f"Generating: {model_a} ↔ {model_b} | Topic: {topic[:50]}...")
        
        # =========== TEMPLATE FOR ACTUAL IMPLEMENTATION ===========
        # Replace this mock with actual API calls
        
        # Mock conversation for demonstration
        messages = [
            {
                "role": f"{model_a}_speaker",
                "model": model_a,
                "content": f"[{model_a}] perspective on: {topic}. This is turn 1.",
                "turn": 1,
            },
            {
                "role": f"{model_b}_speaker",
                "model": model_b,
                "content": f"[{model_b}] responds: That's interesting. Let me add: {topic}. This is turn 1.",
                "turn": 1,
            },
            {
                "role": f"{model_a}_speaker",
                "model": model_a,
                "content": f"[{model_a}] further comment on the topic with analytical depth. Turn 2.",
                "turn": 2,
            },
            {
                "role": f"{model_b}_speaker",
                "model": model_b,
                "content": f"[{model_b}] concurs but adds nuance and counterpoints. Turn 2.",
                "turn": 2,
            },
        ]
        
        # ========================================================
        # In real implementation, replace above with:
        #
        # messages = []
        # for turn in range(num_turns):
        #     # Model A speaks
        #     a_response = self._call_model(model_a, persona, topic, previous_context)
        #     messages.append(a_response)
        #     
        #     # Model B responds
        #     b_response = self._call_model(model_b, persona, topic, previous_context)
        #     messages.append(b_response)
        #
        # ========================================================
        
        conversation = {
            "conversation_id": conversation_id,
            "models": [model_a, model_b],
            "model_sequence": [model_a, model_b] * num_turns,
            "topic": topic,
            "num_turns": num_turns,
            "num_messages": len(messages),
            "messages": messages,
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
            "estimated_tokens": sum(len(msg["content"].split()) for msg in messages) * 1.3,
        }
        
        return conversation
    
    def generate_dataset(
        self,
        model_pairs: List[Tuple[str, str]],
        conversations_per_pair: int = 500,
    ) -> Dict[str, Any]:
        """
        Generate complete multi-model conversation dataset.
        
        Args:
            model_pairs: List of (model_a, model_b) tuples
            conversations_per_pair: Conversations to generate per pair
        
        Returns:
            Metadata about generated dataset
        """
        
        all_conversations = []
        
        for model_a, model_b in model_pairs:
            pair_name = f"{model_a}_{model_b}"
            pair_dir = self.output_dir / "conversations" / pair_name
            pair_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"\n{'='*70}")
            logger.info(f"Generating {conversations_per_pair} conversations: {model_a} ↔ {model_b}")
            logger.info(f"Output: {pair_dir}")
            logger.info(f"{'='*70}")
            
            pair_conversations = []
            pair_metadata = {
                "model_pair": (model_a, model_b),
                "conversations_generated": 0,
                "tokens_used": 0,
                "estimated_cost": 0.0,
                "start_time": datetime.now(timezone.utc).isoformat(),
            }
            
            for i in range(conversations_per_pair):
                # Cycle through topics
                topic = self.TOPICS[i % len(self.TOPICS)]
                
                # Generate conversation
                conversation = self.generate_pair_conversation(
                    model_a=model_a,
                    model_b=model_b,
                    topic=topic,
                    num_turns=5,
                )
                
                pair_conversations.append(conversation)
                all_conversations.append(conversation)
                
                # Save periodically (every 100 conversations)
                if (i + 1) % 100 == 0:
                    logger.info(f"  {i+1}/{conversations_per_pair} conversations generated")
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
            
            pair_metadata["conversations_generated"] = len(pair_conversations)
            pair_metadata["end_time"] = datetime.now(timezone.utc).isoformat()
            
            # Save pair conversations to JSONL
            pair_output = pair_dir / f"conversations_{pair_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
            with open(pair_output, 'w') as f:
                for conv in pair_conversations:
                    f.write(json.dumps(conv) + '\n')
            
            logger.info(f"✓ Saved {len(pair_conversations)} conversations to {pair_output.name}")
            
            # Save pair metadata
            self.metadata["model_pairs"][pair_name] = pair_metadata
        
        self.metadata["total_conversations"] = len(all_conversations)
        
        # Save global metadata
        metadata_file = self.output_dir / "metadata" / f"generation_metadata_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✓ Dataset generation complete!")
        logger.info(f"Total conversations: {self.metadata['total_conversations']}")
        logger.info(f"Metadata saved to: {metadata_file}")
        logger.info(f"{'='*70}\n")
        
        return self.metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-model conversations for sentiment analysis validation"
    )
    parser.add_argument(
        "--model-pairs",
        nargs="+",
        default=["gpt4_claude", "gpt4_llama", "claude_mistral"],
        help="Model pairs to generate conversations for",
    )
    parser.add_argument(
        "--conversations-per-pair",
        type=int,
        default=500,
        help="Number of conversations to generate per model pair",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synthetic"),
        help="Output directory for synthetic data",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate metadata without actual API calls",
    )
    
    args = parser.parse_args()
    
    # Parse model pairs
    model_pairs = []
    for pair_str in args.model_pairs:
        if "_" in pair_str:
            a, b = pair_str.split("_", 1)
            model_pairs.append((a, b))
        else:
            logger.warning(f"Invalid model pair format: {pair_str}")
    
    if not model_pairs:
        logger.error("No valid model pairs specified")
        return
    
    logger.info("="*70)
    logger.info("MULTI-MODEL CONVERSATION GENERATOR")
    logger.info("="*70)
    logger.info(f"Model pairs: {model_pairs}")
    logger.info(f"Conversations per pair: {args.conversations_per_pair}")
    logger.info(f"Total conversations: {len(model_pairs) * args.conversations_per_pair}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("="*70 + "\n")
    
    # Initialize generator
    generator = MultiModelConversationGenerator(output_dir=args.output_dir)
    
    # Generate dataset
    metadata = generator.generate_dataset(
        model_pairs=model_pairs,
        conversations_per_pair=args.conversations_per_pair,
    )
    
    # Print summary
    logger.info("\nGeneration Summary:")
    logger.info(f"  Total conversations: {metadata['total_conversations']}")
    logger.info(f"  Model pairs: {list(metadata['model_pairs'].keys())}")
    
    logger.info("\n" + "="*70)
    logger.info("NEXT STEPS:")
    logger.info("="*70)
    logger.info("1. Apply sentiment pipeline to synthetic data:")
    logger.info("   python scripts/run_moltbook_rule_based.py \\")
    logger.info("       --input data/synthetic/conversations/*/*.jsonl")
    logger.info("\n2. Compare with real data:")
    logger.info("   python scripts/compare_synthetic_vs_real.py \\")
    logger.info("       --real-data data/staged/moltbook_comments_all.jsonl \\")
    logger.info("       --synthetic-data data/synthetic/conversations/*/*.jsonl")
    logger.info("\n3. Analyze model interaction patterns:")
    logger.info("   python scripts/analyze_model_interactions.py \\")
    logger.info("       --input data/synthetic/metadata/*.json")
    logger.info("="*70)


if __name__ == "__main__":
    main()
