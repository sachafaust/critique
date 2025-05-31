from typing import Dict, List, Optional, Any
from datetime import datetime
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import uuid

from .models import LLMClient
from .chains import IterativeWorkflowChain
from langchain_core.prompts import ChatPromptTemplate

console = Console()

class ResponseSynthesizer:
    """Manages the creator-critic iteration workflow."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        max_iterations: int = 1,
        confidence_threshold: float = 0.8
    ):
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
    
    async def synthesize(
        self,
        prompt: str,
        models: List[str],
        creator_model: str,
        output_format: str = "human"
    ) -> Dict[str, Any]:
        """Execute the full creator-critic iteration workflow."""
        start_time = time.time()
        
        # Get creator model
        creator_llm = self.llm_client.get_model(creator_model)
        
        # Get critic models
        critic_llms = []
        critic_names = []
        for model_name in models:
            if model_name != creator_model:  # Don't use creator as critic
                try:
                    critic_llms.append(self.llm_client.get_model(model_name))
                    critic_names.append(model_name)
                except ValueError:
                    console.print(f"[yellow]Warning: Model {model_name} not available, skipping[/yellow]")
        
        # If no critics available, use all available models as critics
        if not critic_llms:
            critic_llms = [self.llm_client.get_model(name) for name in models]
            critic_names = models
        
        # Initialize workflow chain
        workflow_chain = IterativeWorkflowChain(
            creator_model=creator_llm,
            critic_models=critic_llms,
            critic_model_names=critic_names
        )
        
        # Execute workflow
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            progress.add_task("Running creator-critic iterations...", total=None)
            
            workflow_results = await workflow_chain._call({
                "prompt": prompt,
                "max_iterations": self.max_iterations
            })
        
        # Calculate total duration and cost
        total_duration_ms = (time.time() - start_time) * 1000
        total_cost = 0.0  # TODO: Implement cost calculation for new workflow
        
        # Extract final iteration data
        final_iteration = workflow_results["iterations"][-1] if workflow_results["iterations"] else {}
        final_content = workflow_results["final_content"]
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(workflow_results)
        
        # Prepare results
        results = {
            "execution_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "input": {
                "prompt": prompt,
                "creator_model": creator_model,
                "critic_models": critic_names,
                "max_iterations": self.max_iterations
            },
            "results": {
                "workflow_results": workflow_results,
                "final_answer": final_content,
                "confidence_score": quality_metrics["average_confidence"],
                "consensus_score": quality_metrics["average_quality"],
                "total_iterations": workflow_results["total_iterations"],
                "convergence_achieved": workflow_results["convergence_achieved"]
            },
            "performance": {
                "total_duration_ms": total_duration_ms,
                "estimated_cost_usd": total_cost
            },
            "quality_metrics": quality_metrics
        }
        
        # Format output
        if output_format == "human":
            self._print_human_output(results)
        
        return results
    
    def _calculate_quality_metrics(self, workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics from workflow results."""
        if not workflow_results["iterations"]:
            return {
                "average_quality": 0.5,
                "average_confidence": 0.5,
                "quality_progression": [],
                "confidence_progression": []
            }
        
        quality_progression = []
        confidence_progression = []
        
        for iteration in workflow_results["iterations"]:
            # Extract quality scores from critics
            if iteration["critics"]:
                quality_scores = [
                    critic["critic_response"].get("quality_score", 0.5)
                    for critic in iteration["critics"]
                ]
                confidence_scores = [
                    critic["critic_response"].get("confidence", 0.5)
                    for critic in iteration["critics"]
                ]
                
                avg_quality = sum(quality_scores) / len(quality_scores)
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                
                quality_progression.append(avg_quality)
                confidence_progression.append(avg_confidence)
        
        return {
            "average_quality": quality_progression[-1] if quality_progression else 0.5,
            "average_confidence": confidence_progression[-1] if confidence_progression else 0.5,
            "quality_progression": quality_progression,
            "confidence_progression": confidence_progression,
            "improvement_achieved": len(quality_progression) > 1 and quality_progression[-1] > quality_progression[0]
        }
    
    def _print_human_output(self, results: Dict[str, Any]) -> None:
        """Format and print results in human-readable format with enhanced visual clarity."""
        workflow_results = results["results"]["workflow_results"]
        
        # Header with execution summary
        console.print("\n" + "="*80, style="bold blue")
        console.print("🔄 CREATOR-CRITIC ITERATION RESULTS", style="bold blue", justify="center")
        console.print("="*80, style="bold blue")
        console.print(f"📊 Total Iterations: {results['results']['total_iterations']}", style="cyan")
        console.print(f"🎯 Convergence Achieved: {'✅ Yes' if results['results']['convergence_achieved'] else '❌ No'}", style="green" if results['results']['convergence_achieved'] else "red")
        console.print(f"⚙️  Creator Model: {results['input']['creator_model']}", style="cyan")
        console.print(f"🔍 Critic Models: {', '.join(results['input']['critic_models'])}", style="cyan")
        
        # Show each iteration in detail
        for i, iteration in enumerate(workflow_results["iterations"], 1):
            console.print(f"\n{'='*20} ITERATION {i} {'='*20}", style="bold magenta")
            
            # Creator content in a clear box
            creator_content = iteration["creator"]["creator_response"]["content"]
            creator_confidence = iteration["creator"]["creator_response"].get("confidence", 0)
            
            console.print(f"\n🎨 CREATOR OUTPUT ({results['input']['creator_model']})", style="bold green")
            console.print(f"Confidence: {creator_confidence*100:.1f}%", style="dim green")
            
            # Create a panel for the creator content
            from rich.panel import Panel
            creator_panel = Panel(
                creator_content,
                title=f"📝 Creator Response - Iteration {i}",
                title_align="left",
                border_style="green",
                padding=(1, 2)
            )
            console.print(creator_panel)
            
            # Critics feedback section
            console.print(f"\n🔍 CRITICS FEEDBACK", style="bold yellow")
            
            for j, critic in enumerate(iteration["critics"], 1):
                critic_data = critic["critic_response"]
                model_name = critic_data.get("critic_model", f"Critic {j}")
                quality_score = critic_data.get("quality_score", 0)
                continue_flag = critic_data.get("continue_iteration", False)
                
                # Color code based on quality score
                score_color = "red" if quality_score < 0.6 else "yellow" if quality_score < 0.8 else "green"
                continue_color = "green" if continue_flag else "red"
                continue_text = "🔄 Continue" if continue_flag else "✅ Stop"
                
                console.print(f"\n  🤖 {model_name}", style="bold cyan")
                console.print(f"     📊 Quality Score: {quality_score*100:.1f}%", style=score_color)
                console.print(f"     💪 Strengths:", style="dim cyan")
                for strength in critic_data.get("strengths", []):
                    console.print(f"        • {strength}", style="green")
                
                console.print(f"     🔧 Improvements:", style="dim cyan")
                for improvement in critic_data.get("improvements", []):
                    console.print(f"        • {improvement}", style="yellow")
                
                console.print(f"     🎯 Decision: {continue_text}", style=continue_color)
                
                # Show specific feedback if available
                specific_feedback = critic_data.get("specific_feedback", "")
                if specific_feedback and len(specific_feedback) > 20:
                    feedback_panel = Panel(
                        specific_feedback,
                        title=f"💬 Detailed Feedback from {model_name}",
                        title_align="left",
                        border_style="cyan",
                        padding=(0, 1)
                    )
                    console.print(feedback_panel)
        
        # Final results section
        console.print(f"\n{'='*25} FINAL RESULTS {'='*25}", style="bold blue")
        
        final_panel = Panel(
            results["results"]["final_answer"],
            title="🏆 FINAL ANSWER",
            title_align="center",
            border_style="blue",
            padding=(1, 2)
        )
        console.print(final_panel)
        
        # Metrics summary
        console.print(f"\n📈 QUALITY METRICS", style="bold blue")
        console.print(f"  🎯 Final Confidence: {results['results']['confidence_score']*100:.1f}%", style="green")
        console.print(f"  ⭐ Final Quality: {results['results']['consensus_score']*100:.1f}%", style="green")
        
        # Performance summary  
        console.print(f"\n⚡ PERFORMANCE", style="bold blue")
        console.print(f"  ⏱️  Total Duration: {results['performance']['total_duration_ms']/1000:.1f}s", style="cyan")
        console.print(f"  💰 Estimated Cost: ${results['performance']['estimated_cost_usd']:.4f}", style="cyan")
        
        console.print("\n" + "="*80, style="bold blue") 