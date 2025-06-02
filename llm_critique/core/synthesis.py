from typing import Dict, List, Optional, Any
from datetime import datetime
import time
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import uuid

from .models import LLMClient
from .chains import IterativeWorkflowChain
from langchain_core.prompts import ChatPromptTemplate
from .personas import PersonaConfig

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
        total_cost = self._calculate_actual_cost(prompt, creator_model, critic_names, workflow_results)
        
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
            console.print(f"\n🔍 CRITICS FEEDBACK (Iteration {i})", style="bold yellow")
            
            for j, critic in enumerate(iteration["critics"], 1):
                critic_data = critic["critic_response"]
                model_name = critic_data.get("critic_model", f"Critic {j}")
                quality_score = critic_data.get("quality_score", 0)
                continue_flag = critic_data.get("continue_iteration", False)
                
                # Color code based on quality score
                score_color = "red" if quality_score < 0.6 else "yellow" if quality_score < 0.8 else "green"
                continue_color = "green" if continue_flag else "red"
                continue_text = "🔄 Continue" if continue_flag else "✅ Stop"
                
                console.print(f"\n  🤖 {model_name} (Iteration {i})", style="bold cyan")
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
                        title=f"💬 Detailed Feedback from {model_name} (Iteration {i})",
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
        
        # Iteration summary
        requested_iterations = results["input"]["max_iterations"]
        used_iterations = results["results"]["total_iterations"]
        convergence_achieved = results["results"]["convergence_achieved"]
        
        console.print(f"\n🔄 ITERATION SUMMARY", style="bold blue")
        console.print(f"  📝 Requested Iterations: {requested_iterations}", style="cyan")
        console.print(f"  ✅ Used Iterations: {used_iterations}", style="green")
        
        if convergence_achieved:
            console.print(f"  🎯 Status: Convergence achieved after {used_iterations} iteration{'s' if used_iterations != 1 else ''}", style="green")
            if used_iterations < requested_iterations:
                console.print(f"  💡 Early Stop: Stopped {requested_iterations - used_iterations} iteration{'s' if (requested_iterations - used_iterations) != 1 else ''} early due to quality convergence", style="dim green")
        else:
            console.print(f"  ⏸️  Status: Maximum iterations reached without convergence", style="yellow")
        
        # Metrics summary
        console.print(f"\n📈 QUALITY METRICS", style="bold blue")
        console.print(f"  🎯 Final Confidence: {results['results']['confidence_score']*100:.1f}%", style="green")
        console.print(f"  ⭐ Final Quality: {results['results']['consensus_score']*100:.1f}%", style="green")
        
        # Performance summary  
        console.print(f"\n⚡ PERFORMANCE", style="bold blue")
        console.print(f"  ⏱️  Total Duration: {results['performance']['total_duration_ms']/1000:.1f}s", style="cyan")
        console.print(f"  💰 Estimated Cost: ${results['performance']['estimated_cost_usd']:.4f}", style="cyan")
        
        console.print("\n" + "="*80, style="bold blue")

    def _calculate_actual_cost(self, prompt: str, creator_model: str, critic_names: List[str], workflow_results: Dict[str, Any]) -> float:
        """Calculate the actual cost of the workflow execution."""
        def estimate_tokens(text: str) -> int:
            """Estimate token count from text. Rough approximation: ~4 characters per token."""
            if not text:
                return 0
            return max(1, len(text) // 4)
        
        total_cost = 0.0
        input_tokens = estimate_tokens(prompt)
        
        # Estimate standard output tokens (these are estimates since we don't have exact token counts)
        creator_output_tokens_base = 500
        critic_output_tokens_base = 200
        
        for iteration_num, iteration in enumerate(workflow_results.get("iterations", []), 1):
            # Creator cost for this iteration
            creator_data = iteration.get("creator", {}).get("creator_response", {})
            creator_content = creator_data.get("content", "")
            
            # Estimate actual tokens used (context grows with iterations)
            context_multiplier = 1 + (iteration_num - 1) * 0.3
            iteration_input_tokens = int(input_tokens * context_multiplier)
            
            # Use actual creator response length for better accuracy
            creator_output_tokens = estimate_tokens(creator_content) if creator_content else creator_output_tokens_base
            
            # Calculate creator cost (input + output)
            creator_input_cost = self.llm_client.estimate_cost(creator_model, iteration_input_tokens)
            creator_output_cost = self.llm_client.estimate_cost(creator_model, creator_output_tokens)
            total_cost += creator_input_cost + creator_output_cost
            
            # Critics cost for this iteration
            for critic_name in critic_names:
                # Critics analyze creator output + original prompt
                critic_input_tokens = input_tokens + creator_output_tokens
                
                # Use standard critic output estimate
                critic_input_cost = self.llm_client.estimate_cost(critic_name, critic_input_tokens)
                critic_output_cost = self.llm_client.estimate_cost(critic_name, critic_output_tokens_base)
                total_cost += critic_input_cost + critic_output_cost
        
        return total_cost


class PersonaAwareSynthesizer:
    """Enhanced synthesizer that handles expert persona critiques with specialized workflows."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        persona_manager,
        max_iterations: int = 1,
        confidence_threshold: float = 0.8
    ):
        self.llm_client = llm_client
        self.persona_manager = persona_manager
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
    
    async def synthesize_with_personas(
        self,
        prompt: str,
        persona_configs: List,  # List of PersonaConfig objects
        creator_persona,  # PersonaConfig object (expert or vanilla)
        output_format: str = "human"
    ) -> Dict[str, Any]:
        """Execute persona-enhanced creator-critic iteration workflow."""
        start_time = time.time()
        
        from .personas import UnifiedCritic, CritiqueResult, PersonaType
        
        # Initialize creator with persona awareness
        creator_model = creator_persona.preferred_model
        creator_llm = self.llm_client.get_model(creator_model)
        
        # Initialize persona critics
        persona_critics = []
        for persona_config in persona_configs:
            try:
                critic = UnifiedCritic(persona_config, self.llm_client)
                persona_critics.append(critic)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not initialize critic for persona {persona_config.name}: {e}[/yellow]")
        
        if not persona_critics:
            raise ValueError("No valid persona critics could be initialized")
        
        # Execute persona-aware workflow
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Running persona-enhanced iterations...", total=None)
            
            workflow_results = await self._execute_persona_workflow(
                prompt, creator_llm, creator_persona, persona_critics, progress, task
            )
        
        # Calculate total duration and cost
        total_duration_ms = (time.time() - start_time) * 1000
        total_cost = self._calculate_persona_cost(prompt, creator_persona, persona_configs, workflow_results)
        
        # Extract final iteration data
        final_iteration = workflow_results["iterations"][-1] if workflow_results["iterations"] else {}
        final_content = workflow_results["final_content"]
        
        # Calculate quality metrics with persona awareness
        quality_metrics = self._calculate_persona_quality_metrics(workflow_results)
        
        # Prepare results with persona information
        results = {
            "execution_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "input": {
                "prompt": prompt,
                "creator_persona": creator_persona.name,
                "creator_type": creator_persona.persona_type.value,
                "creator_model": creator_model,
                "personas": [p.name for p in persona_configs],
                "max_iterations": self.max_iterations,
                "mode": "persona_enhanced"
            },
            "results": {
                "workflow_results": workflow_results,
                "final_answer": final_content,
                "confidence_score": quality_metrics["average_confidence"],
                "consensus_score": quality_metrics["persona_consensus"],
                "total_iterations": workflow_results["total_iterations"],
                "convergence_achieved": workflow_results["convergence_achieved"]
            },
            "persona_analysis": {
                "expert_insights": quality_metrics["expert_insights"],
                "consensus_areas": quality_metrics["consensus_areas"],
                "conflicting_perspectives": quality_metrics["conflicts"],
                "weighted_recommendations": quality_metrics["weighted_recommendations"]
            },
            "performance": {
                "total_duration_ms": total_duration_ms,
                "estimated_cost_usd": total_cost
            },
            "quality_metrics": quality_metrics
        }
        
        # Format output with persona-aware formatting
        if output_format == "human":
            self._print_persona_human_output(results)
        
        return results
    
    async def _execute_persona_workflow(
        self,
        prompt: str,
        creator_llm,
        creator_persona,
        persona_critics: List,
        progress,
        task_id
    ) -> Dict[str, Any]:
        """Execute the persona-enhanced workflow with specialized handling."""
        iterations = []
        current_content = ""
        convergence_achieved = False
        
        for iteration_num in range(1, self.max_iterations + 1):
            progress.update(task_id, description=f"Iteration {iteration_num}/{self.max_iterations}")
            
            # Creator phase - persona-aware prompting
            if iteration_num == 1:
                creator_prompt = self._generate_creator_prompt(prompt, creator_persona)
            else:
                # Incorporate previous feedback with persona awareness
                creator_prompt = self._generate_revision_prompt(
                    prompt, current_content, creator_persona, iterations[-1]["persona_critiques"]
                )
            
            # Generate creator response
            creator_response = await creator_llm.ainvoke(creator_prompt)
            current_content = creator_response.content
            
            # Persona critiques phase
            persona_critiques = []
            for critic in persona_critics:
                try:
                    critique_result = await critic.execute_critique(
                        content=current_content,
                        context=f"Original prompt: {prompt}\nIteration: {iteration_num}\nCreated by: {creator_persona.name}"
                    )
                    persona_critiques.append(critique_result)
                except Exception as e:
                    console.print(f"[yellow]Warning: Error executing critique from {critic.persona.name}: {e}[/yellow]")
            
            # Store iteration results
            iteration_data = {
                "iteration_num": iteration_num,
                "creator_response": {
                    "content": current_content,
                    "model": creator_persona.preferred_model,
                    "persona": creator_persona.name,
                    "persona_type": creator_persona.persona_type.value
                },
                "persona_critiques": persona_critiques,
                "consensus_score": self._calculate_iteration_consensus(persona_critiques),
                "should_continue": False
            }
            
            iterations.append(iteration_data)
            
            # Check convergence
            if iteration_data["consensus_score"] >= self.confidence_threshold:
                convergence_achieved = True
                break
            
            # Mark for continuation if not last iteration
            if iteration_num < self.max_iterations:
                iteration_data["should_continue"] = True
        
        return {
            "iterations": iterations,
            "final_content": current_content,
            "total_iterations": len(iterations),
            "convergence_achieved": convergence_achieved
        }
    
    def _generate_creator_prompt(self, prompt: str, creator_persona) -> str:
        """Generate persona-aware creator prompt."""
        from .personas import PersonaType
        
        if creator_persona.persona_type == PersonaType.EXPERT:
            # Rich, persona-specific prompt for expert creators
            language_patterns_text = ""
            if creator_persona.language_patterns:
                patterns = []
                for category, phrases in creator_persona.language_patterns.items():
                    if phrases:
                        patterns.append(f"  {category.title()}: {', '.join(phrases[:2])}")
                if patterns:
                    language_patterns_text = f"\n\n## Your Characteristic Language:\n" + "\n".join(patterns)

            creator_prompt = f"""# EXPERT PERSONA CREATION: {creator_persona.name}

## YOUR IDENTITY
{creator_persona.description}

## YOUR CORE PRINCIPLES TO APPLY
{chr(10).join([f"• {p}" for p in creator_persona.core_principles[:4]])}

## YOUR DECISION FRAMEWORKS
{chr(10).join([f"• {f}" for f in creator_persona.decision_frameworks[:3]])}
{language_patterns_text}

**CRITICAL INSTRUCTION**: Respond exactly as {creator_persona.name} would. Use your characteristic language patterns, apply your known principles, and create content that authentically reflects your perspective and expertise.

---

**Request to Address:**
{prompt}

---

**Create your response in character as {creator_persona.name}. Focus on:**
• Applying your core principles and expertise
• Using your authentic voice and communication style  
• Providing insights that reflect your unique perspective
• Creating well-structured, thoughtful content
"""
        else:
            # Simple prompt for vanilla model creators
            creator_prompt = f"""Please provide a comprehensive and thoughtful response to the following prompt:

{prompt}

Focus on creating content that is:
• Well-structured and clearly organized
• Thorough in addressing all key aspects
• Professional and engaging in tone
• Backed by solid reasoning and examples where appropriate
"""
        
        return creator_prompt
    
    def _generate_revision_prompt(self, original_prompt: str, previous_content: str, creator_persona, critiques: List) -> str:
        """Generate persona-aware revision prompt incorporating feedback."""
        from .personas import PersonaType
        
        feedback_summary = self._summarize_persona_feedback(critiques)
        
        if creator_persona.persona_type == PersonaType.EXPERT:
            # Expert persona revision with authentic voice preservation
            revision_prompt = f"""# EXPERT REVISION: {creator_persona.name}

## YOUR TASK
Revise and improve your previous response based on the expert feedback provided, while maintaining your authentic voice and perspective.

## ORIGINAL REQUEST
{original_prompt}

## YOUR PREVIOUS RESPONSE
{previous_content}

## EXPERT FEEDBACK TO ADDRESS
{feedback_summary}

## YOUR APPROACH
Stay true to your principles and communication style as {creator_persona.name}. Address the feedback constructively while:
• Maintaining your characteristic perspective and voice
• Applying your core principles to the improvements
• Using your preferred decision frameworks
• Ensuring the revision reflects your authentic expertise

**Provide your improved response as {creator_persona.name}:**
"""
        else:
            # Vanilla revision prompt
            revision_prompt = f"""Please revise and improve your response based on the expert feedback provided.

Original request: {original_prompt}

Previous response: {previous_content}

Expert feedback to address: {feedback_summary}

Provide an improved response that:
• Addresses the specific feedback points raised
• Maintains quality and clarity
• Builds upon the strengths of your previous response
• Incorporates new insights and improvements
"""
        
        return revision_prompt
    
    def _summarize_persona_feedback(self, persona_critiques: List) -> str:
        """Summarize feedback from all persona critiques."""
        if not persona_critiques:
            return "No feedback available."
        
        summary_parts = []
        for critique in persona_critiques:
            persona_name = critique.persona_name
            key_insights = "; ".join(critique.key_insights[:2])
            recommendations = "; ".join(critique.recommendations[:2])
            
            summary_parts.append(f"{persona_name}: {key_insights} | Recommends: {recommendations}")
        
        return "\n".join(summary_parts)
    
    def _calculate_iteration_consensus(self, persona_critiques: List) -> float:
        """Calculate consensus score for an iteration based on persona agreement."""
        if not persona_critiques:
            return 0.0
        
        # Weight by quality scores and confidence levels
        weighted_scores = []
        for critique in persona_critiques:
            # Combine quality and confidence with expertise match for expert personas
            if critique.persona_type.value == "expert":
                expertise_bonus = critique.expertise_match * 0.2
                score = (critique.quality_score + critique.confidence_level + expertise_bonus) / 3
            else:
                score = (critique.quality_score + critique.confidence_level) / 2
            
            weighted_scores.append(score)
        
        return sum(weighted_scores) / len(weighted_scores)
    
    def _calculate_persona_quality_metrics(self, workflow_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate quality metrics with persona-specific analysis."""
        if not workflow_results["iterations"]:
            return {
                "average_confidence": 0.5,
                "persona_consensus": 0.5,
                "expert_insights": [],
                "consensus_areas": [],
                "conflicts": [],
                "weighted_recommendations": []
            }
        
        # Collect all persona critiques
        all_critiques = []
        consensus_progression = []
        
        for iteration in workflow_results["iterations"]:
            iteration_critiques = iteration["persona_critiques"]
            all_critiques.extend(iteration_critiques)
            consensus_progression.append(iteration["consensus_score"])
        
        # Group by persona type for analysis
        expert_critiques = [c for c in all_critiques if c.persona_type.value == "expert"]
        vanilla_critiques = [c for c in all_critiques if c.persona_type.value == "vanilla"]
        
        # Calculate metrics
        final_consensus = consensus_progression[-1] if consensus_progression else 0.5
        avg_confidence = sum(c.confidence_level for c in all_critiques) / len(all_critiques) if all_critiques else 0.5
        
        # Extract expert insights
        expert_insights = []
        for critique in expert_critiques[-3:]:  # Last 3 expert critiques
            expert_insights.append({
                "persona": critique.persona_name,
                "insights": critique.key_insights,
                "expertise_match": critique.expertise_match,
                "red_flags": critique.red_flags_identified
            })
        
        # Find consensus areas (insights mentioned by multiple personas)
        consensus_areas = self._find_consensus_insights(all_critiques)
        
        # Identify conflicts
        conflicts = self._identify_persona_conflicts(all_critiques)
        
        # Weight recommendations by expertise
        weighted_recommendations = self._weight_recommendations_by_expertise(expert_critiques)
        
        return {
            "average_confidence": avg_confidence,
            "persona_consensus": final_consensus,
            "consensus_progression": consensus_progression,
            "expert_insights": expert_insights,
            "consensus_areas": consensus_areas,
            "conflicts": conflicts,
            "weighted_recommendations": weighted_recommendations
        }
    
    def _find_consensus_insights(self, critiques: List) -> List[str]:
        """Find insights that multiple personas agree on."""
        # Simple keyword-based consensus detection
        insight_words = {}
        
        for critique in critiques:
            for insight in critique.key_insights:
                words = insight.lower().split()
                for word in words:
                    if len(word) > 4:  # Only consider meaningful words
                        insight_words[word] = insight_words.get(word, 0) + 1
        
        # Find words mentioned by multiple personas
        consensus_words = [word for word, count in insight_words.items() if count >= 2]
        
        # Create consensus statements
        consensus_areas = []
        if "quality" in consensus_words or "improve" in consensus_words:
            consensus_areas.append("Quality improvements needed")
        if "clarity" in consensus_words or "clear" in consensus_words:
            consensus_areas.append("Clarity enhancements required")
        if "evidence" in consensus_words or "support" in consensus_words:
            consensus_areas.append("Better evidence/support needed")
        
        return consensus_areas[:3]  # Limit to top 3
    
    def _identify_persona_conflicts(self, critiques: List) -> List[Dict[str, Any]]:
        """Identify conflicting perspectives between personas."""
        conflicts = []
        
        # Compare quality scores - if they differ significantly, it's a conflict
        if len(critiques) >= 2:
            scores = [c.quality_score for c in critiques]
            score_range = max(scores) - min(scores)
            
            if score_range > 0.3:  # Significant disagreement
                high_scorers = [c for c in critiques if c.quality_score >= max(scores) - 0.1]
                low_scorers = [c for c in critiques if c.quality_score <= min(scores) + 0.1]
                
                conflicts.append({
                    "type": "quality_assessment",
                    "high_assessment": [c.persona_name for c in high_scorers],
                    "low_assessment": [c.persona_name for c in low_scorers],
                    "score_range": score_range
                })
        
        return conflicts
    
    def _weight_recommendations_by_expertise(self, expert_critiques: List) -> List[Dict[str, Any]]:
        """Weight recommendations by persona expertise and confidence."""
        weighted_recs = []
        
        for critique in expert_critiques:
            if critique.recommendations:
                weight = (critique.expertise_match + critique.confidence_level) / 2
                
                for rec in critique.recommendations[:2]:  # Top 2 recommendations
                    weighted_recs.append({
                        "recommendation": rec,
                        "persona": critique.persona_name,
                        "weight": weight,
                        "expertise_match": critique.expertise_match
                    })
        
        # Sort by weight descending
        weighted_recs.sort(key=lambda x: x["weight"], reverse=True)
        
        return weighted_recs[:5]  # Top 5 weighted recommendations
    
    def _calculate_persona_cost(self, prompt: str, creator_persona: PersonaConfig, persona_configs: List, workflow_results: Dict[str, Any]) -> float:
        """Calculate cost with persona-specific context overhead."""
        def estimate_tokens(text: str) -> int:
            return max(1, len(text) // 4)
        
        cost = 0.0
        
        for iteration in workflow_results["iterations"]:
            # Creator cost
            creator_content = iteration["creator_response"]["content"]
            creator_input_tokens = estimate_tokens(prompt)
            creator_output_tokens = estimate_tokens(creator_content)
            
            cost += self.llm_client.estimate_cost(creator_persona.preferred_model, creator_input_tokens)
            cost += self.llm_client.estimate_cost(creator_persona.preferred_model, creator_output_tokens)
            
            # Persona critics cost (includes context overhead)
            for i, critique in enumerate(iteration["persona_critiques"]):
                persona_config = persona_configs[i] if i < len(persona_configs) else None
                
                if persona_config:
                    # Include persona context in input token calculation
                    context_tokens = persona_config.get_prompt_context_size_estimate()
                    content_tokens = estimate_tokens(prompt + creator_content)
                    total_input_tokens = context_tokens + content_tokens
                    
                    output_tokens = estimate_tokens(critique.critique_text)
                    
                    model_name = critique.model_used
                    cost += self.llm_client.estimate_cost(model_name, total_input_tokens)
                    cost += self.llm_client.estimate_cost(model_name, output_tokens)
        
        return cost
    
    def _print_persona_human_output(self, results: Dict[str, Any]) -> None:
        """Print human-readable results with persona-aware formatting."""
        from rich.table import Table
        from rich.panel import Panel
        
        console.print()
        console.print(Panel.fit(
            "[bold blue]🎭 Persona-Enhanced LLM Critique Results[/bold blue]",
            border_style="blue"
        ))
        
        # Input information with creator persona details
        input_info = results["input"]
        console.print(f"📝 Prompt: {input_info['prompt'][:100]}{'...' if len(input_info['prompt']) > 100 else ''}", style="dim")
        
        # Enhanced creator display
        creator_icon = "🎭" if input_info["creator_type"] == "expert" else "🤖"
        console.print(f"⚙️  Creator: {creator_icon} {input_info['creator_persona']} ({input_info['creator_type']}) → {input_info['creator_model']}", style="cyan")
        
        personas_display = ", ".join(input_info["personas"])
        console.print(f"🧠 Critics: {personas_display}", style="cyan")
        
        # Performance metrics
        perf = results["performance"]
        console.print(f"⏱️  Duration: {perf['total_duration_ms']:.0f}ms | 💰 Cost: ${perf['estimated_cost_usd']:.4f}", style="dim")
        console.print()
        
        # Quality metrics
        quality = results["quality_metrics"]
        console.print(f"📊 Confidence: {quality['average_confidence']:.1%} | 🤝 Consensus: {quality['persona_consensus']:.1%}", style="green")
        console.print()
        
        # Creator output with persona context
        creator_type_label = "🎭 EXPERT CREATOR" if input_info["creator_type"] == "expert" else "🤖 CREATOR"
        console.print(f"\n{creator_type_label} OUTPUT ({input_info['creator_persona']})", style="bold green")
        console.print(Panel(results["results"]["final_answer"], border_style="green"))
        
        # Persona analysis summary
        persona_analysis = results["persona_analysis"]
        
        if persona_analysis["consensus_areas"]:
            console.print("\n🤝 [bold]CONSENSUS AREAS[/bold]", style="green")
            for area in persona_analysis["consensus_areas"][:3]:
                console.print(f"  • {area}")
        
        if persona_analysis["conflicting_perspectives"]:
            console.print("\n⚡ [bold]CONFLICTING PERSPECTIVES[/bold]", style="yellow")
            for conflict in persona_analysis["conflicting_perspectives"][:2]:
                persona1 = conflict["persona1"]
                persona2 = conflict["persona2"]
                issue = conflict["issue"]
                console.print(f"  • {persona1} vs {persona2}: {issue}")
        
        if persona_analysis["weighted_recommendations"]:
            console.print("\n💡 [bold]TOP RECOMMENDATIONS[/bold]", style="blue")
            for rec in persona_analysis["weighted_recommendations"][:5]:
                console.print(f"  • {rec['recommendation']} (from {rec['persona']})")
        
        # Individual persona critiques
        workflow_results = results["results"]["workflow_results"]
        if workflow_results["iterations"]:
            final_iteration = workflow_results["iterations"][-1]
            
            console.print(f"\n🎭 [bold]EXPERT PERSONA CRITIQUES[/bold]")
            console.print()
            
            for critique in final_iteration["persona_critiques"]:
                persona_name = critique.persona_name
                persona_type = critique.persona_type.value
                quality_score = critique.quality_score
                confidence = critique.confidence_level
                
                # Persona header with type indicator
                persona_icon = "🧠" if persona_type == "expert" else "🤖"
                console.print(f"{persona_icon} [bold]{persona_name}[/bold] ({persona_type}) - Quality: {quality_score:.1%}, Confidence: {confidence:.1%}", style="cyan")
                
                # Key insights
                if critique.key_insights:
                    console.print("  [bold]Key Insights:[/bold]")
                    for insight in critique.key_insights[:2]:
                        console.print(f"    • {insight}")
                
                # Recommendations
                if critique.recommendations:
                    console.print("  [bold]Recommendations:[/bold]")
                    for rec in critique.recommendations[:2]:
                        console.print(f"    • {rec}")
                
                # Red flags for expert personas
                if persona_type == "expert" and critique.red_flags_identified:
                    console.print("  [bold red]Red Flags:[/bold red]")
                    for flag in critique.red_flags_identified[:2]:
                        console.print(f"    ⚠️  {flag}")
                
                console.print()
        
        # Iteration summary
        iterations_count = results["results"]["total_iterations"]
        convergence = results["results"]["convergence_achieved"]
        convergence_text = "✅ Achieved" if convergence else "❌ Not reached"
        
        console.print(f"🔄 [bold]ITERATIONS:[/bold] {iterations_count} | [bold]CONVERGENCE:[/bold] {convergence_text}")
        console.print()