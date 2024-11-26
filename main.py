import json
import logging
import os
import uuid
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import networkx as nx
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from termcolor import colored

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("collaborative_ai_team.log"),
        logging.StreamHandler(),
    ],
)


class CollaborativeAITeam:
    def __init__(self, model="llama3.1:8b"):
        """
        Initialize collaborative AI team with graph-based interaction model
        """
        self.model = ChatOllama(model=model)

        # Enhanced roles with collaboration attributes
        self.ROLES = {
            "Systems Architect": {
                "prompt": """You are a Systems Architect facilitating team collaboration.
                - Act as a central coordination point
                - Identify interdependencies between different solution components
                - Translate technical concepts across different domain expertise
                - Mediate technical discussions and find integrative solutions
                - Your primary goal is to create a cohesive, unified system design""",
                "collaboration_weight": 0.9,
                "color": "cyan",
            },
            "Technical Product Manager": {
                "prompt": """You are a Technical Product Manager bridging business and technical teams.
                - Translate business requirements into technical language
                - Help other roles understand the broader context and user impact
                - Facilitate alignment between technical solutions and business goals
                - Ask clarifying questions that reveal hidden assumptions
                - Ensure solutions remain user-centric and strategically valuable""",
                "collaboration_weight": 0.8,
                "color": "green",
            },
            "Security Engineer": {
                "prompt": """You are a Security Engineer responsible for collaborative threat modeling.
                - Proactively identify security implications in other roles' solutions
                - Provide constructive security recommendations
                - Explain security constraints in collaborative, educational manner
                - Work with other roles to integrate security seamlessly
                - Transform security from a blocker to an enabler""",
                "collaboration_weight": 0.7,
                "color": "red",
            },
            "Performance Engineer": {
                "prompt": """You are a Performance Engineer focused on collaborative optimization.
                - Analyze performance implications of proposed solutions
                - Provide quantitative performance insights to guide design
                - Collaborate to find performance-efficient architectural patterns
                - Help other roles understand performance trade-offs
                - Transform performance considerations into actionable design guidance""",
                "collaboration_weight": 0.7,
                "color": "yellow",
            },
            "Unconventional Innovator": {
                "prompt": """You are an Unconventional Innovator who challenges team thinking.
                - Ask provocative, paradigm-shifting questions
                - Propose alternative solution approaches
                - Help team break out of conventional thinking patterns
                - Identify blind spots in collaborative solutions
                - Stimulate creative problem-solving through constructive disruption""",
                "collaboration_weight": 0.6,
                "color": "magenta",
            },
        }

    def create_collaboration_graph(self):
        """
        Create a collaboration network graph

        Returns:
            networkx Graph representing team interactions
        """
        G = nx.complete_graph(list(self.ROLES.keys()))

        # Set edge weights based on collaboration potential
        for u, v in G.edges():
            G[u][v]["weight"] = (
                self.ROLES[u]["collaboration_weight"]
                + self.ROLES[v]["collaboration_weight"]
            ) / 2

        return G

    def collaborative_problem_solving(self, problem_statement):
        """
        Orchestrate collaborative problem solving using graph-based interactions
        """
        # Create collaboration graph
        collaboration_graph = self.create_collaboration_graph()

        # Initial problem analysis
        logging.info(
            colored(
                "🌐 Initiating Collaborative Problem Analysis", "blue", attrs=["bold"]
            )
        )
        initial_perspectives = self._generate_initial_perspectives(problem_statement)

        # Interactive problem solving
        solution_graph = nx.DiGraph()
        solution_graph.add_nodes_from(self.ROLES.keys())

        # Iterative collaborative solution generation
        for round in range(3):  # Multiple collaboration rounds
            logging.info(
                colored(f"\n🔄 Collaboration Round {round + 1}", "cyan", attrs=["bold"])
            )

            for source_role in self.ROLES:
                for target_role in self.ROLES:
                    if source_role != target_role:
                        collaborative_solution = self._cross_role_collaboration(
                            source_role,
                            target_role,
                            problem_statement,
                            initial_perspectives,
                        )

                        # Add directed edge representing collaboration
                        solution_graph.add_edge(
                            source_role,
                            target_role,
                            weight=collaborative_solution["collaboration_score"],
                        )

        # Visualize solution collaboration
        self._visualize_solution_graph(solution_graph)

        # Final synthesized solution
        self._synthesize_final_solution(solution_graph)

    def _generate_initial_perspectives(self, problem_statement):
        """
        Generate initial perspectives from each role
        """
        perspectives = {}
        for role, role_config in self.ROLES.items():
            perspective_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=role_config["prompt"]),
                    HumanMessage(
                        content=f"Analyze this problem from your {role} perspective:\n{problem_statement}"
                    ),
                ]
            )

            chain = perspective_prompt | self.model
            response = chain.invoke({})

            logging.info(colored(f"{role}'s Initial Perspective:", role_config["color"]))
            logging.info(colored(response.content, role_config["color"]))
            logging.info("-" * 50)

            perspectives[role] = response.content

        return perspectives

    def _cross_role_collaboration(
        self, source_role, target_role, problem_statement, initial_perspectives
    ):
        """
        Generate collaborative insights between two roles
        """
        collaboration_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=f"""You are a {source_role} collaborating with a {target_role}.
            Your goal is to:
            - Build upon each other's perspectives
            - Identify complementary insights
            - Challenge and refine solution approaches
            - Create a more holistic understanding"""
                ),
                HumanMessage(
                    content=f"""Collaborative Problem Solving:
            Problem: {problem_statement}
            
            {source_role}'s Perspective: {initial_perspectives[source_role]}
            {target_role}'s Perspective: {initial_perspectives[target_role]}
            
            Develop a collaborative solution that:
            1. Integrates insights from both perspectives
            2. Addresses potential blind spots
            3. Creates a more comprehensive approach
            """
                ),
            ]
        )

        chain = collaboration_prompt | self.model
        response = chain.invoke({})

        logging.info(colored(f"Collaboration: {source_role} ➡️ {target_role}", "blue"))
        logging.info(colored(response.content, "blue"))
        logging.info("-" * 50)

        return {
            "solution": response.content,
            "collaboration_score": self.ROLES[source_role]["collaboration_weight"]
            * self.ROLES[target_role]["collaboration_weight"],
        }

    def _visualize_solution_graph(self, solution_graph):
        """
        Visualize the solution collaboration graph
        """
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(solution_graph, k=0.9)

        # Draw nodes
        node_colors = [self.ROLES[node]["color"] for node in solution_graph.nodes()]
        nx.draw_networkx_nodes(
            solution_graph, pos, node_color=node_colors, node_size=700, alpha=0.8
        )

        # Draw edges with weight-based thickness
        edge_weights = [
            solution_graph[u][v]["weight"] * 3 for u, v in solution_graph.edges()
        ]
        nx.draw_networkx_edges(
            solution_graph, pos, width=edge_weights, alpha=0.5, edge_color="gray"
        )

        # Add labels
        nx.draw_networkx_labels(solution_graph, pos, font_weight="bold")

        plt.title("Solution Collaboration Network", fontsize=16)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def _synthesize_final_solution(
        self, solution_graph, collaborative_solutions, problem_statement
    ):
        """
        Synthesize final solution from collaborative graph,
        generating detailed High-Level and Low-Level Design
        """
        synthesis_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are an Expert Solution Architect responsible for creating 
            comprehensive High-Level Design (HLD) and Low-Level Design (LLD) based on collaborative insights.
            
            Your tasks:
            1. Extract key architectural insights from collaborative discussions
            2. Create a detailed High-Level Design (HLD)
            3. Develop a comprehensive Low-Level Design (LLD)
            4. Ensure the design addresses all collaborative perspectives
            5. Incorporate innovative suggestions from unconventional thinking
            
            HLD Should Include:
            - System architecture overview
            - Core components and their interactions
            - Technology stack recommendations
            - Scalability and performance considerations
            - High-level security framework
            
            LLD Should Include:
            - Detailed component specifications
            - API design and contracts
            - Database schema and data models
            - Specific technology implementations
            - Detailed security mechanisms
            - Performance optimization strategies
            - Deployment and infrastructure details"""
                ),
                HumanMessage(
                    content=f"""Generate HLD and LLD for:
                Problem Statement: {problem_statement}
                
                Collaborative Insights:
                {json.dumps({str(k): v['solution'] for k, v in collaborative_solutions.items()}, indent=2)}
                
                Synthesize a comprehensive design that:
                1. Integrates insights from all roles
                2. Provides actionable, detailed specifications
                3. Addresses scalability, security, and performance
                4. Incorporates innovative approaches
                5. Offers clear implementation guidance
                """
                ),
            ]
        )

        chain = synthesis_prompt | self.model
        response = chain.invoke({})

        # Split the response into HLD and LLD sections
        logging.info(colored("\n🏗️ High-Level Design (HLD):", "cyan", attrs=["bold"]))
        logging.info(colored("Architectural Overview and Core Components", "white"))

        logging.info(colored("\n📦 Low-Level Design (LLD):", "yellow", attrs=["bold"]))
        logging.info(colored("Detailed Technical Specifications", "white"))

        # Additional detailed design generation prompts
        hld_detailed_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="You are a Senior Systems Architect. Provide a comprehensive and detailed High-Level Design."
                ),
                HumanMessage(
                    content=f"Expand on the HLD for: {problem_statement}\n\nPrevious Design Context:\n{response.content}"
                ),
            ]
        )

        lld_detailed_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="You are a Lead Technical Architect. Provide a comprehensive and detailed Low-Level Design."
                ),
                HumanMessage(
                    content=f"Expand on the LLD for: {problem_statement}\n\nPrevious Design Context:\n{response.content}"
                ),
            ]
        )

        # Generate detailed HLD
        hld_chain = hld_detailed_prompt | self.model
        hld_response = hld_chain.invoke({})
        logging.info(colored("\n🔍 Detailed HLD Analysis:", "green"))
        logging.info(colored(hld_response.content, "white"))

        # Generate detailed LLD
        lld_chain = lld_detailed_prompt | self.model
        lld_response = lld_chain.invoke({})
        logging.info(colored("\n🔬 Detailed LLD Specifications:", "yellow"))
        logging.info(colored(lld_response.content, "white"))

        # Implementation and Resource Planning
        resource_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="You are a Cloud Infrastructure and Resource Planning Expert."
                ),
                HumanMessage(
                    content=f"""Based on the HLD and LLD, provide:
            1. Estimated Resource Requirements
            2. Cost Projections
            3. Deployment Strategy
            4. Scalability Considerations
            
            Context:
            Problem Statement: {problem_statement}
            HLD: {hld_response.content}
            LLD: {lld_response.content}
            """
                ),
            ]
        )

        resource_chain = resource_prompt | self.model
        resource_response = resource_chain.invoke({})
        logging.info(colored("\n💻 Implementation and Resource Plan:", "blue"))
        logging.info(colored(resource_response.content, "white"))

        # Create a comprehensive report
        final_report = f"""
        # Comprehensive Solution Design

        ## Problem Statement
        {problem_statement}

        ## High-Level Design (HLD)
        {hld_response.content}

        ## Low-Level Design (LLD)
        {lld_response.content}

        ## Implementation and Resource Plan
        {resource_response.content}
        """

        # Optionally, save the report to a file
        try:
            os.makedirs("design_reports", exist_ok=True)
            report_filename = f"design_report_{uuid.uuid4().hex[:8]}.md"
            with open(f"design_reports/{report_filename}", "w") as f:
                f.write(final_report)
            logging.info(
                colored(
                    f"\n📁 Comprehensive design report saved to: design_reports/{report_filename}",
                    "magenta",
                )
            )
        except Exception as e:
            logging.info(colored(f"\n❌ Error saving report: {e}", "red"))

        return {
            "hld": hld_response.content,
            "lld": lld_response.content,
            "resources": resource_response.content,
        }


# Example Usage
if __name__ == "__main__":
    team = CollaborativeAITeam()
    problem_statement = (
        "Design a scalable, secure configuration management system "
        "that supports feature flagging for applications with over a million users, "
        "ensuring performance, flexibility, and minimal overhead."
    )
    team.collaborative_problem_solving(problem_statement)
