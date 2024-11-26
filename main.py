import json
import os
import uuid
from typing import Any, Dict, List

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from termcolor import colored


class AdvancedTeamCollaboration:
    def __init__(self, model="llama3.1:8b", logging_dir="team_logs"):
        """
        Initialize the collaborative AI team framework with extended roles
        """
        self.model = ChatOllama(model=model)
        self.logging_dir = logging_dir
        os.makedirs(logging_dir, exist_ok=True)

        # Enhanced role definitions with more nuanced responsibilities and a new Intern role
        self.ROLES = {
            "Systems Architect": {
                "prompt": """You are a Systems Architect responsible for holistic system design:
                - Analyze architectural feasibility and long-term scalability
                - Identify potential system-wide integration challenges
                - Design modular, extensible architectural patterns
                - Evaluate solution against enterprise-level technical standards""",
                "color": "cyan",
            },
            "Technical Product Manager": {
                "prompt": """You are a Technical Product Manager bridging business and technology:
                - Translate business requirements into technical specifications
                - Assess solution's alignment with user needs and business goals
                - Prioritize features based on strategic value
                - Validate solution's market and user experience potential""",
                "color": "green",
            },
            "Security Engineer": {
                "prompt": """You are a Security Engineer focused on comprehensive threat modeling:
                - Conduct rigorous security vulnerability assessment
                - Identify potential attack vectors and mitigation strategies
                - Ensure compliance with industry security standards
                - Propose robust authentication and data protection mechanisms""",
                "color": "red",
            },
            "Performance Engineer": {
                "prompt": """You are a Performance Engineer dedicated to system efficiency:
                - Analyze performance characteristics and potential bottlenecks
                - Develop performance benchmarks and optimization strategies
                - Evaluate scalability and resource utilization
                - Propose caching, query optimization, and efficient architectures""",
                "color": "yellow",
            },
            "100x Intern": {
                "prompt": """You are an unconventional 100x Intern known for disruptive thinking:
                - Challenge existing solutions with creative, often radical approaches
                - Ask provocative questions that expose hidden assumptions
                - Propose completely unexpected solution strategies
                - Use lateral thinking to reframe the problem in unique ways
                - Don't be afraid to suggest seemingly impossible or absurd solutions""",
                "color": "magenta",
            },
        }

    def generate_initial_perspectives(self, main_problem: str) -> Dict[str, str]:
        """
        Generate initial perspectives from each role on the problem statement

        Returns:
            Dictionary of role perspectives
        """
        initial_perspectives = {}

        for role, role_config in self.ROLES.items():
            perspective_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=role_config["prompt"]),
                    HumanMessage(
                        content=f"""Analyze this problem from your unique {role} perspective:
                Problem Statement: {main_problem}
                
                Provide a detailed initial analysis focusing on:
                - Your primary concerns and observations
                - Potential challenges specific to your domain
                - Initial high-level approach or strategy
                - Critical questions this problem raises
                """
                    ),
                ]
            )

            chain = perspective_prompt | self.model
            response = chain.invoke({})

            # Print colored perspective
            print(colored(f"{role}'s Initial Perspective:", role_config["color"]))
            print(colored(response.content, role_config["color"]))
            print("-" * 50)

            initial_perspectives[role] = response.content

        return initial_perspectives

    def collaborative_solution_generation(self, main_problem: str) -> Dict[str, Any]:
        """
        Generate collaborative solutions using multiple role perspectives

        Returns:
            Comprehensive solution dictionary
        """
        solution = {
            "problem_id": str(uuid.uuid4()),
            "solutions": {},
            "consensus_details": {},
        }

        # Generate solution from each role's perspective
        for role, role_config in self.ROLES.items():
            solution_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=role_config["prompt"]),
                    HumanMessage(
                        content=f"""Develop a solution for this problem from a {role} perspective:
                Problem Statement: {main_problem}
                
                Provide a comprehensive solution addressing:
                - Specific strategies and approaches
                - Technical implementation details
                - Potential challenges and mitigations
                - Unique insights from your domain expertise
                """
                    ),
                ]
            )

            chain = solution_prompt | self.model
            response = chain.invoke({})

            # Print colored solution
            print(colored(f"{role}'s Solution:", role_config["color"]))
            print(colored(response.content, role_config["color"]))
            print("-" * 50)

            solution["solutions"][role] = response.content

        return solution

    def cross_validation(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform cross-validation of the solution by having each role
        critically evaluate the entire solution

        Returns:
            Detailed consensus and validation report
        """
        consensus_details = {}

        for validator_role, validator_config in self.ROLES.items():
            validation_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=validator_config["prompt"]),
                    HumanMessage(
                        content=f"""Critically evaluate the proposed solution from a comprehensive perspective:
                
                Solution Overview:
                {json.dumps(solution['solutions'], indent=2)}
                
                Your Task:
                - Thoroughly assess the solution's strengths and weaknesses
                - Identify potential blind spots or risks
                - Rate the solution's effectiveness (1-10 scale)
                - Suggest any critical improvements or modifications
                
                Provide a structured, detailed evaluation that considers 
                technical feasibility, business value, and potential challenges.
                """
                    ),
                ]
            )

            chain = validation_prompt | self.model
            response = chain.invoke({})

            # Print colored validation
            print(colored(f"{validator_role}'s Validation:", validator_config["color"]))
            print(colored(response.content, validator_config["color"]))
            print("-" * 50)

            consensus_details[validator_role] = {
                "validation": response.content,
            }

        return consensus_details

    def run_collaborative_problem_solving(self, main_problem: str):
        """
        Orchestrate the entire collaborative problem-solving process
        """
        print(
            colored(
                "🚀 Initiating Advanced Collaborative Problem Solving",
                "white",
                attrs=["bold"],
            )
        )
        print(colored(f"Problem: {main_problem}\n", "cyan"))

        # Step 1: Initial Perspectives
        print(colored("🔍 STAGE 1: Initial Perspectives", "blue", attrs=["bold"]))
        initial_perspectives = self.generate_initial_perspectives(main_problem)

        # Step 2: Solution Generation
        print(colored("\n🧠 STAGE 2: Solution Generation", "blue", attrs=["bold"]))
        solution = self.collaborative_solution_generation(main_problem)

        # Step 3: Cross-Validation
        print(
            colored(
                "\n🤝 STAGE 3: Cross-Validation and Consensus", "blue", attrs=["bold"]
            )
        )
        consensus_details = self.cross_validation(solution)

        # Final Recommendation Synthesis
        print(
            colored("\n🏁 FINAL COLLABORATIVE RECOMMENDATION", "green", attrs=["bold"])
        )
        final_recommendation = self._synthesize_final_recommendation(
            solution, consensus_details
        )
        print(colored(final_recommendation, "green"))

        return solution, consensus_details

    def _synthesize_final_recommendation(
        self, solution: Dict[str, Any], consensus: Dict[str, Any]
    ) -> str:
        """
        Synthesize a comprehensive final recommendation
        """
        synthesis_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""
            As a master synthesizer, integrate multiple perspectives into a cohesive, 
            actionable final recommendation. Consider:
            - Strengths from different role perspectives
            - Critical considerations and potential risks
            - Balanced approach addressing multiple concerns
            - Clear, implementable strategy
            """
                ),
                HumanMessage(
                    content=f"""
            Synthesize a final recommendation based on:
            Solutions: {json.dumps(solution['solutions'], indent=2)}
            Consensus Details: {json.dumps(consensus, indent=2)}
            
            Provide a comprehensive, nuanced recommendation that:
            1. Highlights key strategic insights
            2. Addresses potential challenges
            3. Offers clear next steps
            4. Integrates diverse perspectives
            """
                ),
            ]
        )

        chain = synthesis_prompt | self.model
        response = chain.invoke({})

        return response.content


# Example Usage
if __name__ == "__main__":
    team = AdvancedTeamCollaboration()
    problem_statement = (
        "Design a scalable, secure configuration management system "
        "that supports feature flagging for applications with over a million users, "
        "ensuring performance, flexibility, and minimal overhead."
    )
    team.run_collaborative_problem_solving(problem_statement)
