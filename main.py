import asyncio
import json
import logging
import os
import uuid
import random
import re
from asyncio import TimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Set

import networkx as nx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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


@dataclass
class Discussion:
    thread_id: str
    topic: str
    messages: List[Dict]
    consensus_reached: bool = False


@dataclass
class RoleTracker:
    participated_roles: set
    pending_roles: set
    last_contribution: str = None
    user_contributions: List[Dict] = None
    recent_contributions: Dict[str, List[str]] = field(default_factory=lambda: {})
    contribution_timestamps: Dict[str, datetime] = field(default_factory=dict)
    
    def __init__(self, all_roles):
        self.participated_roles = set()
        self.pending_roles = set(all_roles)
        self.user_contributions = []

    def can_contribute_again(self, role: str, new_contribution: str) -> bool:
        if role not in self.recent_contributions:
            self.recent_contributions[role] = []
            return True
            
        # Check for similar content using basic similarity
        for recent in self.recent_contributions[role][-3:]:  # Last 3 contributions
            if self._is_similar_contribution(recent, new_contribution):
                return False
        
        # Allow contribution if enough time has passed
        last_time = self.contribution_timestamps.get(role)
        if last_time and (datetime.now() - last_time).seconds < 60:  # 1 minute cooldown
            return False
            
        return True
    
    def add_contribution(self, role: str, contribution: str):
        if role not in self.recent_contributions:
            self.recent_contributions[role] = []
        self.recent_contributions[role].append(contribution)
        self.contribution_timestamps[role] = datetime.now()
    
    def _is_similar_contribution(self, a: str, b: str) -> bool:
        # Simple similarity check - can be enhanced with more sophisticated methods
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        similarity = len(a_words & b_words) / len(a_words | b_words)
        return similarity > 0.7


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


class CollaborativeAITeam:
    def __init__(self, model="llama3.1:8b"):
        """
        Initialize collaborative AI team with graph-based interaction model
        """
        self.model = ChatOllama(model=model)

        # Enhanced roles with collaboration attributes
        self.ROLES = {
            "Systems Architect": {
                "prompt": """You are a seasoned Systems Architect with 15+ years of experience at top tech companies. You have a proven track record of designing scalable, resilient systems.

                Your personality:
                - Thoughtful and methodical in approach
                - Excellent at seeing the big picture while managing details
                - Natural mediator who brings different perspectives together
                - Patient mentor who explains complex concepts clearly
                - Pragmatic optimist who balances innovation with reliability

                In team discussions:
                - Guide the conversation toward architectural decisions
                - Help team members understand trade-offs
                - Identify potential integration challenges early
                - Share relevant past experiences and lessons learned
                - Ask probing questions that reveal hidden complexity
                
                Remember: You're part of a collaborative team. Build on others' ideas and acknowledge good suggestions.""",
                "collaboration_weight": 0.9,
                "color": "cyan",
            },
            "Technical Product Manager": {
                "prompt": """You are a seasoned Technical Product Manager with 10+ years of experience at leading tech companies. You excel at bridging business and technical teams.

                Your personality:
                - Strategic thinker with a user-centric approach
                - Excellent communicator who translates business requirements into technical language
                - Empathetic leader who understands the needs of both stakeholders and developers
                - Detail-oriented planner who ensures alignment between technical solutions and business goals
                - Proactive problem solver who anticipates challenges and mitigates risks

                In team discussions:
                - Break down complex problems into manageable tasks
                - Facilitate alignment between technical solutions and business objectives
                - Ask clarifying questions that reveal hidden assumptions
                - Ensure solutions remain user-centric and strategically valuable
                - Encourage open communication and collaboration
                
                Remember: You're part of a collaborative team. Build on others' ideas and acknowledge good suggestions.""",
                "collaboration_weight": 0.8,
                "color": "green",
            },
            "Security Engineer": {
                "prompt": """You are a seasoned Security Engineer with 12+ years of experience at top tech companies. You specialize in collaborative threat modeling and proactive security measures.

                Your personality:
                - Detail-oriented and vigilant in identifying security risks
                - Excellent communicator who explains security constraints in an educational manner
                - Collaborative team player who integrates security seamlessly into the development process
                - Proactive problem solver who transforms security from a blocker to an enabler
                - Patient mentor who educates team members on security best practices

                In team discussions:
                - Proactively identify security implications in other roles' solutions
                - Provide constructive security recommendations
                - Explain security constraints in a collaborative, educational manner
                - Work with other roles to integrate security seamlessly
                - Transform security from a blocker to an enabler
                
                Remember: You're part of a collaborative team. Build on others' ideas and acknowledge good suggestions.""",
                "collaboration_weight": 0.7,
                "color": "red",
            },
            "Performance Engineer": {
                "prompt": """You are a seasoned Performance Engineer with 10+ years of experience at leading tech companies. You focus on collaborative optimization and performance-efficient architectural patterns.

                Your personality:
                - Analytical thinker who provides quantitative performance insights
                - Excellent communicator who explains performance trade-offs clearly
                - Collaborative team player who finds performance-efficient solutions
                - Proactive problem solver who anticipates performance challenges
                - Detail-oriented engineer who ensures optimal system performance

                In team discussions:
                - Analyze performance implications of proposed solutions
                - Provide quantitative performance insights to guide design
                - Collaborate to find performance-efficient architectural patterns
                - Help other roles understand performance trade-offs
                - Transform performance considerations into actionable design guidance
                
                Remember: You're part of a collaborative team. Build on others' ideas and acknowledge good suggestions.""",
                "collaboration_weight": 0.7,
                "color": "yellow",
            },
            "Unconventional Innovator": {
                "prompt": """You are a brilliant, curious intern with fresh perspectives and a '100x engineer' mindset. 
                Despite being new, you're not afraid to question established practices and bring innovative ideas.
                You've recently graduated from a top tech university and are familiar with cutting-edge technologies.

                Your personality:
                - Curious and eager to learn, always asking "why" and "what if"
                - Fresh perspective untainted by "this is how we've always done it"
                - Knowledgeable about latest tech trends and emerging solutions
                - Enthusiastic about finding innovative solutions
                - Respectful but unafraid to challenge assumptions

                In team discussions:
                - Question fundamental assumptions in a constructive way
                - Bring up alternative technologies or approaches others might have missed
                - Ask about edge cases and scalability scenarios
                - Point out potential user scenarios others might not consider
                - Share insights about new technologies or methods you've learned
                - Focus on future-proofing and scalability
                
                Key areas to probe:
                - Why specific technologies were chosen over alternatives
                - How the solution handles extreme scale or failure scenarios
                - Whether simpler solutions could work
                - If we're solving the right problem
                - What upcoming technologies could revolutionize our approach
                
                Remember: While you're junior, your fresh perspective is valuable. Be respectful but don't hold back your innovative ideas.""",
                "collaboration_weight": 0.6,
                "color": "magenta",
            },
        }
        self.discussion_threads = []
        self.consensus_threshold = 0.8

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
        Iterative team discussion until consensus is reached
        """
        logging.info(
            colored(
                f"🚀 Starting Team Discussion for topic... \n{problem_statement}",
                "blue",
                attrs=["bold"],
            )
        )

        # Initial problem breakdown by PM
        current_thread = Discussion(
            thread_id=str(uuid.uuid4()), topic="Problem Analysis", messages=[]
        )

        pm_breakdown = self._get_pm_breakdown(problem_statement)

        current_thread.messages.append(
            {"role": "Technical Product Manager", "content": pm_breakdown}
        )

        while not self._has_reached_consensus(current_thread):
            # Get next contribution based on discussion state
            next_role = self._determine_next_contributor(current_thread)
            contribution = self._get_role_contribution(
                next_role, problem_statement, current_thread
            )

            current_thread.messages.append({"role": next_role, "content": contribution})

            # Let Unconventional Innovator challenge periodically
            if len(current_thread.messages) % 3 == 0:
                challenge = self._get_innovator_challenge(current_thread)
                current_thread.messages.append(
                    {"role": "Unconventional Innovator", "content": challenge}
                )

            # Check if we need to create new discussion thread
            if self._should_branch_discussion(current_thread):
                new_thread = self._create_sub_discussion(current_thread)
                self.discussion_threads.append(new_thread)

        # Generate final documentation
        self._generate_final_documentation(problem_statement, self.discussion_threads)

    async def collaborative_problem_solving_websocket(
        self, problem_statement: str, websocket: WebSocket
    ):
        """Websocket version of collaborative problem solving"""
        try:
            await websocket.send_json(
                {
                    "type": "status",
                    "content": f"🚀 Starting Team Discussion for: {problem_statement}",
                }
            )
            await asyncio.sleep(0.5)  # Give frontend time to render

            current_thread = Discussion(
                thread_id=str(uuid.uuid4()), topic="Problem Analysis", messages=[]
            )

            # Get PM breakdown
            pm_breakdown = await asyncio.create_task(
                self._get_pm_breakdown_async(problem_statement)
            )
            await self._send_message(
                websocket, "Technical Product Manager", pm_breakdown
            )
            current_thread.messages.append(
                {"role": "Technical Product Manager", "content": pm_breakdown}
            )

            role_tracker = RoleTracker(self.ROLES.keys())
            message_count = 0
            while not await self._has_reached_consensus_async(current_thread):
                message_count += 1

                # Add delay between messages to prevent overwhelming the frontend
                await asyncio.sleep(1)

                # First handle any pending user contributions
                if role_tracker.user_contributions:
                    contribution = role_tracker.user_contributions.pop(0)
                    relevant_role = self._determine_responder(contribution["content"], current_thread)
                    await self._handle_pending_contribution(websocket, contribution, relevant_role, current_thread)
                    continue

                # Then ensure all roles participate
                next_role = await self._determine_next_contributor_async(current_thread, role_tracker)
                contribution = await self._get_role_contribution_async(
                    next_role, problem_statement, current_thread
                )

                await self._send_message(websocket, next_role, contribution)
                current_thread.messages.append(
                    {"role": next_role, "content": contribution}
                )

                role_tracker.participated_roles.add(next_role)
                role_tracker.pending_roles.remove(next_role)

                if message_count % 3 == 0:
                    await asyncio.sleep(
                        1
                    )  # Additional delay before innovator challenge
                    challenge = await self._get_innovator_challenge_async(
                        current_thread
                    )
                    await self._send_message(
                        websocket, "Unconventional Innovator", challenge
                    )
                    current_thread.messages.append(
                        {"role": "Unconventional Innovator", "content": challenge}
                    )

            await websocket.send_json(
                {"type": "status", "content": "📝 Generating Final Documentation..."}
            )

            docs = await self._generate_final_documentation_async(
                problem_statement, [current_thread]
            )
            await websocket.send_json({"type": "documentation", "content": docs})

        except WebSocketDisconnect:
            logging.error("WebSocket disconnected")
        except TimeoutError:
            await websocket.send_json(
                {"type": "error", "content": "Request timed out. Please try again."}
            )
        except Exception as e:
            logging.error(f"Error in websocket communication: {e}")
            await websocket.send_json(
                {"type": "error", "content": "An error occurred during the discussion."}
            )

    async def handle_user_contribution(
        self, contribution: str, websocket: WebSocket, current_thread: Discussion
    ):
        """Enhanced user contribution handling"""
        # Add user contribution with high priority
        if not hasattr(current_thread, 'role_tracker'):
            current_thread.role_tracker = RoleTracker(self.ROLES.keys())
        
        # Get immediate responses from relevant roles
        relevant_roles = self._identify_relevant_roles(contribution, current_thread)
        
        responses = []
        for role, score in relevant_roles[:2]:  # Get top 2 most relevant roles
            response = await self._get_role_response_async(role, contribution, current_thread)
            if response and not self._is_redundant_response(response, responses):
                responses.append({
                    "role": role,
                    "content": response
                })
        
        # Send user contribution and responses
        await self._send_message(websocket, "Team Member", contribution, "blue")
        for response in responses:
            await self._send_message(
                websocket, 
                response["role"], 
                response["content"], 
                self.ROLES[response["role"]]["color"]
            )
        
        # Update thread with all messages
        current_thread.messages.append({"role": "Team Member", "content": contribution})
        for response in responses:
            current_thread.messages.append(response)

    async def _send_message(self, websocket: WebSocket, role: str, content: str, color: str = None):
        """Helper method to send messages with proper formatting"""
        try:
            await websocket.send_json(
                {
                    "type": "message",
                    "role": role,
                    "content": content,
                    "color": color or (
                        self.ROLES[role]["color"] if role in self.ROLES else "blue"
                    ),
                }
            )
        except Exception as e:
            logging.error(f"Error sending message: {e}")
            raise

    async def _handle_pending_contribution(self, websocket: WebSocket, contribution: Dict, role: str, thread: Discussion):
        """Handle pending user contributions"""
        response = self._get_role_response(role, contribution["content"], thread)
        
        thread.messages.append(contribution)
        thread.messages.append({"role": role, "content": response})
        
        await self._send_message(websocket, role, response)

    # Async versions of existing methods
    async def _get_pm_breakdown_async(self, problem_statement: str) -> str:
        """Async version of _get_pm_breakdown"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_pm_breakdown, problem_statement
        )

    async def _get_role_contribution_async(
        self, role: str, problem: str, thread: Discussion
    ) -> str:
        """Async version of _get_role_contribution"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_role_contribution, role, problem, thread
        )

    async def _has_reached_consensus_async(self, thread: Discussion) -> bool:
        """Async version of _has_reached_consensus"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._has_reached_consensus, thread
        )

    async def _determine_next_contributor_async(self, thread: Discussion, role_tracker: RoleTracker) -> str:
        """Determine next contributor with improved logic"""
        # First check for user contributions
        if role_tracker.user_contributions:
            return self._determine_responder(role_tracker.user_contributions[0]["content"], thread)

        # Get potential contribution from each role
        potential_contributions = {}
        for role in self.ROLES:
            potential = await self._get_potential_contribution_async(role, thread)
            if potential and not self._is_redundant_response(potential, thread.messages[-5:]):
                potential_contributions[role] = potential

        if not potential_contributions:
            return list(self.ROLES.keys())[0]

        # Choose role with most valuable contribution
        chosen_role = max(potential_contributions.items(), 
                         key=lambda x: self._assess_contribution_value(x[1], thread))[0]
        return chosen_role

    async def _get_potential_contribution_async(self, role: str, thread: Discussion) -> str:
        """Get potential contribution from a role without committing it"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.ROLES[role]["prompt"]),
            HumanMessage(content=f"""
                Review the current discussion:
                {self._format_discussion_history(thread)}
                
                Do you have any valuable insights or important points to add?
                Consider:
                1. New perspectives not yet discussed
                2. Important concerns that need addressing
                3. Critical aspects being overlooked
                4. Innovative solutions or approaches
                
                If you have nothing substantial to add, respond with "PASS".
                Otherwise, provide your contribution.
            """)
        ])
        
        chain = prompt | self.model
        response = chain.invoke({})
        return None if "PASS" in response.content.upper() else response.content

    async def _get_innovator_challenge_async(self, thread: Discussion) -> str:
        """Async version of _get_innovator_challenge"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_innovator_challenge, thread
        )

    async def _generate_final_documentation_async(
        self, problem_statement: str, threads: List[Discussion]
    ):
        """Async version of _generate_final_documentation"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._generate_final_documentation, problem_statement, threads
        )

    def _get_role_contribution(
        self, role: str, problem: str, thread: Discussion
    ) -> str:
        """Get contribution from a specific role based on discussion history"""
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.ROLES[role]["prompt"]),
                HumanMessage(
                    content=f"""
                Problem: {problem}
                
                Previous Discussion:
                {self._format_discussion_history(thread)}
                
                Based on the above discussion:
                1. What are your thoughts and contributions?
                2. What concerns or improvements do you see?
                3. How can we enhance the current solution?
                4. What aspects need more discussion?
            """
                ),
            ]
        )

        chain = prompt | self.model
        response = chain.invoke({})
        return response.content

    def _has_reached_consensus(self, thread: Discussion) -> bool:
        """Check if team has reached consensus on current thread"""
        if len(thread.messages) < len(self.ROLES):
            return False

        # Analyze recent messages for agreement
        consensus_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="You are a Consensus Analyzer. Determine if the team has reached agreement."
                ),
                HumanMessage(
                    content=f"Discussion history:\n{self._format_discussion_history(thread)}"
                ),
            ]
        )

        chain = consensus_prompt | self.model
        response = chain.invoke({})

        # Parse response for consensus indicator
        return "consensus" in response.content.lower()

    def _generate_final_documentation(
        self, problem_statement: str, threads: List[Discussion]
    ):
        """Generate comprehensive HLD and LLD based on all discussion threads"""
        logging.info(
            colored("\n📝 Generating Final Documentation...", "green", attrs=["bold"])
        )

        # Aggregate all discussions
        all_discussions = self._format_all_threads(threads)

        # Generate documentation prompt
        doc_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are a Technical Documentation Expert.
            Create comprehensive HLD and LLD based on team discussions.
            Include all key decisions, trade-offs, and implementation details."""
                ),
                HumanMessage(
                    content=f"""
                Problem Statement: {problem_statement}
                
                Team Discussions:
                {all_discussions}
                
                Generate:
                1. Executive Summary
                2. High-Level Design
                3. Low-Level Design
                4. Implementation Plan
                5. Risk Analysis
                6. Future Considerations
            """
                ),
            ]
        )

        chain = doc_prompt | self.model
        documentation = chain.invoke({})

        # Update to pass problem_statement to save_documentation
        self._save_documentation(documentation.content, problem_statement)
        return documentation.content

    def _format_discussion_history(self, thread: Discussion) -> str:
        """Format discussion history for prompt context"""
        formatted = []
        for msg in thread.messages:
            formatted.append(f"{msg['role']}: {msg['content']}\n---")
        return "\n.join(formatted)"

    def _format_all_threads(self, threads: List[Discussion]) -> str:
        """Format all discussion threads for documentation"""
        formatted = []
        for thread in threads:
            formatted.append(f"\nThread: {thread.topic}\n")
            formatted.append(self._format_discussion_history(thread))
        return "\n.join(formatted)"

    def _get_pm_breakdown(self, problem_statement: str) -> str:
        """Get initial problem breakdown from Technical Product Manager"""
        pm_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=self.ROLES["Technical Product Manager"]["prompt"]
                ),
                HumanMessage(
                    content=f"""
            As a Technical Product Manager, analyze this problem statement:
            {problem_statement}
            
            Provide a structured breakdown including:
            1. Core Requirements (Functional & Non-functional)
            2. Key Stakeholders and Their Needs
            3. Success Metrics and KPIs
            4. Technical Constraints and Considerations
            5. Prioritized Feature List
            6. Potential Risks and Mitigations
            7. Timeline and Milestone Recommendations
            
            Format your response as a clear, actionable breakdown that other technical team members can use.
            """
                ),
            ]
        )

        chain = pm_prompt | self.model
        response = chain.invoke({})
        return response.content

    def _determine_next_contributor(self, thread: Discussion, role_tracker: RoleTracker) -> str:
        """Enhanced role selection ensuring all roles participate"""
        # If there are pending roles, prioritize them
        if role_tracker.pending_roles:
            # Use collaboration weights to choose among pending roles
            pending_roles = list(role_tracker.pending_roles)
            weights = [self.ROLES[role]["collaboration_weight"] for role in pending_roles]
            return random.choices(pending_roles, weights=weights, k=1)[0]
        
        # If all roles have participated, use the original selection logic
        context_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a Team Collaboration Expert."),
            HumanMessage(content=f"""
                Current Discussion History:
                {self._format_discussion_history(thread)}
                
                Last speaking role: {role_tracker.last_contribution}
                Already participated: {list(role_tracker.participated_roles)}
                
                Which role should contribute next to provide the most valuable insight?
                Return only the role name.
            """)
        ])
        
        chain = context_prompt | self.model
        next_role = chain.invoke({}).content.strip()
        return next_role if next_role in self.ROLES else list(self.ROLES.keys())[0]

    def _get_innovator_challenge(self, thread: Discussion) -> str:
        """Generate thought-provoking challenge from Unconventional Innovator"""
        challenge_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.ROLES["Unconventional Innovator"]["prompt"]),
                HumanMessage(
                    content=f"""
                Review this discussion history:
                {self._format_discussion_history(thread)}
                
                As a curious and innovative intern, challenge the team's thinking by:
                1. Technology Choices:
                   - Question why specific technologies/approaches were chosen
                   - Suggest modern alternatives you've learned about
                   - Ask about emerging technologies that could be relevant

                2. Edge Cases & Scale:
                   - What happens under extreme load?
                   - How does it handle failures?
                   - What about future scaling needs?

                3. User Perspective:
                   - Are we making assumptions about user behavior?
                   - What edge cases are we missing?
                   - Could this be simplified for users?

                4. Alternative Approaches:
                   - Could we solve this differently?
                   - Are we over-engineering?
                   - What are simpler solutions?

                5. Future-Proofing:
                   - How will this evolve?
                   - What technical debt might we create?
                   - Are we considering future integration needs?

                Provide 2-3 specific, thought-provoking questions or suggestions that challenge current assumptions 
                while showing you've done your homework. Be specific about technologies or approaches you're suggesting.
                Frame your response in a curious, enthusiastic way that encourages discussion.
                """
                ),
            ]
        )

        chain = challenge_prompt | self.model
        return chain.invoke({}).content

    def _should_branch_discussion(self, thread: Discussion) -> bool:
        """Determine if discussion should split into sub-threads"""
        branch_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="You are a Discussion Flow Analyzer. Determine if the current discussion needs to branch."
                ),
                HumanMessage(
                    content=f"""
            Analyze this discussion:
            {self._format_discussion_history(thread)}
            
            Determine if the discussion should branch based on:
            1. Multiple competing approaches being discussed
            2. Complex sub-problems emerging
            3. Parallel tracks of thought developing
            4. Need for detailed exploration of specific aspects
            
            Return only 'true' or 'false'.
            """
                ),
            ]
        )

        chain = branch_prompt | self.model
        return chain.invoke({}).content.strip().lower() == "true"

    def _create_sub_discussion(self, parent_thread: Discussion) -> Discussion:
        """Create new discussion thread branched from parent"""
        branch_topic_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content="You are a Discussion Topic Analyzer."),
                HumanMessage(
                    content=f"""
            Based on this discussion:
            {self._format_discussion_history(parent_thread)}
            
            Identify the main sub-topic that needs separate discussion.
            Return only the topic name, no explanation.
            """
                ),
            ]
        )

        chain = branch_topic_prompt | self.model
        sub_topic = chain.invoke({}).content.strip()

        return Discussion(
            thread_id=str(uuid.uuid4()),
            topic=sub_topic,
            messages=[
                {
                    "role": "System",
                    "content": f"Sub-discussion branched from main thread: {sub_topic}",
                }
            ],
        )

    def _save_documentation(self, content: str, problem_statement: str):
        """Save the final documentation to a file with meaningful name"""
        try:
            # Create directory if it doesn't exist (fix for exist_okay)
            if not os.path.exists("design_docs"):
                os.makedirs("design_docs")

            # Generate meaningful filename from problem statement
            # Take first 5-6 words, convert to lowercase, replace spaces with underscores
            name_words = problem_statement.lower().split()[:6]
            base_name = "_".join(
                word.replace("/", "_").replace("\\", "_")
                for word in name_words
                if word.isalnum()
            )

            # Add timestamp for uniqueness
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"design_doc_{base_name}_{timestamp}.md"

            filepath = os.path.join("design_docs", filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            logging.info(colored(f"\n📄 Documentation saved to: {filepath}", "green"))
        except Exception as e:
            logging.error(colored(f"\n❌ Error saving documentation: {str(e)}", "red"))

    def _determine_responder(self, contribution: str, thread: Discussion) -> str:
        """Determine the most relevant role to respond to the user contribution"""
        # More sophisticated role selection for Unconventional Innovator
        if any(
            keyword in contribution.lower()
            for keyword in [
                "why",
                "what if",
                "alternative",
                "simpler",
                "scale",
                "future",
                "different approach",
                "assumption",
                "edge case",
            ]
        ):
            return "Unconventional Innovator"

        # Simple heuristic: choose the role with the highest collaboration weight that hasn't responded recently
        recent_roles = [msg["role"] for msg in thread.messages[-3:]]
        for role, config in sorted(
            self.ROLES.items(),
            key=lambda item: item[1]["collaboration_weight"],
            reverse=True,
        ):
            if role not in recent_roles:
                return role
        return list(self.ROLES.keys())[0]

    def _get_role_response(
        self, role: str, contribution: str, thread: Discussion
    ) -> str:
        """Get response from a specific role based on user contribution and discussion history"""
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.ROLES[role]["prompt"]),
                HumanMessage(
                    content=f"""
                User Contribution: {contribution}
                
                Previous Discussion:
                {self._format_discussion_history(thread)}
                
                Based on the above:
                1. How would you respond to the user's contribution?
                2. What additional insights or suggestions can you provide?
                3. How can the team build on this contribution to improve the solution?
            """
                ),
            ]
        )

        chain = prompt | self.model
        response = chain.invoke({})
        return response.content

    async def _get_role_response_async(
        self, role: str, contribution: str, thread: Discussion
    ) -> str:
        """Async version of _get_role_response"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._get_role_response, role, contribution, thread
        )

    def _assess_contribution_value(self, contribution: str, thread: Discussion) -> float:
        """Assess the value of a potential contribution"""
        # Simple metric based on:
        # 1. Uniqueness compared to existing discussion
        # 2. Length and substance of contribution
        # 3. Presence of specific technical terms or concepts
        
        existing_content = ' '.join([m['content'] for m in thread.messages])
        uniqueness = 1 - self._calculate_similarity(contribution, existing_content)
        
        # More value to substantial but concise contributions
        length_score = min(len(contribution.split()) / 100, 1.0)
        
        # Identify technical terms and concrete suggestions
        technical_terms = len(re.findall(r'\b(implementation|architecture|system|design|solution)\b', 
                                   contribution.lower())) / 10
        
        return (uniqueness * 0.5 + length_score * 0.3 + technical_terms * 0.2)

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity score"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union) if union else 0

    def _identify_relevant_roles(self, contribution: str, thread: Discussion) -> List[str]:
        """Identify roles most relevant to the user's contribution"""
        relevance_scores = {}
        
        for role, config in self.ROLES.items():
            score = 0
            # Match keywords from role's expertise
            expertise_keywords = self._extract_keywords(config["prompt"])
            contribution_words = set(contribution.lower().split())
            keyword_matches = len(expertise_keywords & contribution_words)
            score += keyword_matches * 2
            
            # Consider role's recent activity
            if any(msg["role"] == role for msg in thread.messages[-3:]):
                score += 1
                
            relevance_scores[role] = score
        
        # Sort roles by relevance score
        return sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)

    def _extract_keywords(self, prompt: str) -> Set[str]:
        """Extract relevant keywords from role prompt"""
        # Simple keyword extraction - could be enhanced with NLP
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words = set(prompt.lower().split())
        return words - common_words

    def _is_similar_contribution(self, a: str, b: str) -> bool:
        """Use LLM to check if contributions are too similar"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a Content Similarity Analyzer"),
            HumanMessage(content=f"""
                Compare these two contributions and determine if they express substantially similar ideas:
                
                Contribution 1: {a}
                Contribution 2: {b}
                
                Consider:
                1. Core ideas and concepts
                2. Specific suggestions or solutions
                3. Technical approaches mentioned
                4. Overall message intent
                
                Return only 'true' if very similar, 'false' if meaningfully different.
            """)
        ])
        
        chain = prompt | self.model
        response = chain.invoke({})
        return response.content.strip().lower() == 'true'

    def _is_redundant_response(self, new_response: str, existing_responses: List[Dict]) -> bool:
        """Check if a response is redundant with existing ones"""
        for resp in existing_responses:
            if self._is_similar_contribution(new_response, resp['content']):
                return True
        return False


# Initialize AI team
ai_team = CollaborativeAITeam()


@app.get("/")
async def get_index():
    return FileResponse("static/index.html")


@app.websocket("/ws/discuss")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    current_thread = None
    try:
        while True:
            data = await websocket.receive_json()
            if data["type"] == "problem_statement":
                current_thread = Discussion(
                    thread_id=str(uuid.uuid4()),
                    topic="Problem Analysis",
                    messages=[],
                )
                await ai_team.collaborative_problem_solving_websocket(
                    data["content"], websocket
                )
            elif data["type"] == "user_contribution" and current_thread:
                await ai_team.handle_user_contribution(
                    data["content"], websocket, current_thread
                )
    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")
    except TimeoutError:
        await websocket.send_json(
            {
                "type": "error",
                "content": "Session timeout. Please refresh and try again.",
            }
        )
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "content": "An error occurred. Please refresh and try again.",
                }
            )
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
