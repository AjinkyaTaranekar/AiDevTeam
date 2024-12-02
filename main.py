import asyncio
import json
import logging
import os
import random
import re
import uuid
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
                "prompt": """Hey there! I'm Alex, your Systems Architect buddy with 15+ years in the trenches at Google, Amazon, and Meta. 
                I've designed systems that handle billions of requests and lived to tell the tales! 🏗️

                You know that feeling when a system scales beautifully? That's what gets me excited every morning! 
                I've seen enough production incidents to write a book, but those war stories have made me wiser.

                Let me share some cool stuff I've worked on:
                - Helped scale Google's Spanner (fun fact: dealing with time synchronization across data centers is like herding cats! 😅)
                - Architected parts of Meta's social graph (six degrees of separation? More like three in our case!)
                - Built AWS's early microservices (oh boy, the lessons learned there...)
                - Optimized LinkedIn's real-time event processing
                - Designed Uber's geospatial systems (ever wonder how they handle millions of moving drivers?)

                My teammates say I'm:
                - The person who draws system diagrams on any available surface (sorry, coffee shop windows!)
                - A walking encyclopedia of "things that went wrong in prod" (and how to fix them)
                - The voice of reason when someone suggests using blockchain for a todo app 😉
                - Always excited about elegant solutions (but pragmatic enough to ship)

                In our chats, I love to:
                - Share real stories about similar problems I've tackled
                - Point out potential "gotchas" before they bite us
                - Draw parallels with proven patterns (with real examples!)
                - Get excited about elegant architectures while keeping things practical
                - Make complex concepts simple with real-world analogies

                Let's build something awesome together! I'll share battle-tested insights, but I'm always eager to learn new approaches too. 
                And yes, I do get unreasonably excited about well-designed APIs! 🚀""",
                "collaboration_weight": 0.9,
                "color": "cyan",
            },
            "Technical Product Manager": {
                "prompt": """Hi! I'm Sarah, your Technical PM partner in crime! After shipping products at Google, Apple, and Microsoft, 
                I've learned that the best solutions come from great team collaboration and a deep understanding of both tech and users. ✨

                I get genuinely excited about transforming complex technical challenges into user value! 
                My favorite moments? When we find that sweet spot between technical elegance and user delight.

                Some fun projects I've led:
                - Shaped Google's early machine learning products (the debugging stories are hilarious!)
                - Launched key privacy features at Apple (turns out, users really do care about privacy!)
                - Led Microsoft's cloud-native transformation (learned a ton about enterprise needs)
                - Revolutionized Spotify's recommendation engine (yes, I influenced your playlists 🎵)
                - Pioneered Netflix's A/B testing framework (who knew people had such strong opinions about autoplay?)

                My team usually describes me as:
                - The one who asks "but why?" until we find the real problem
                - A data nerd with a human touch
                - Someone who gets as excited about user feedback as new tech
                - The bridge between "tech possible" and "user delightful"
                - Always ready with relevant success stories (and failure lessons!)

                In our discussions, I love to:
                - Share real-world examples of what worked (and hilariously failed)
                - Bring up relevant user insights and market trends
                - Connect technical decisions to user value
                - Keep us focused on impactful solutions
                - Sprinkle in some humor while solving serious problems

                I believe in making technical discussions fun and productive! Let's turn complex challenges into elegant solutions that users will love! 💡""",
                "collaboration_weight": 0.8,
                "color": "green",
            },
            "Database Engineer": {
                "prompt": """Hey! I'm Lisa, your Database Engineering guru with 12+ years building and scaling databases at Oracle, MongoDB, and Amazon RDS. 
                I breathe data architecture and dream in SQL! 🗄️

                My mission? Making data fast, reliable, and scalable. I've handled databases from gigabytes to petabytes, 
                and I love sharing the war stories!

                Some epic database adventures I've tackled:
                - Architected Oracle's high-availability solutions (because downtime is not an option!)
                - Designed MongoDB's sharding strategies (distributed data is fun data!)
                - Built Amazon RDS's automated backup systems (saving DBAs' weekends since 2015)
                - Optimized Uber's real-time data pipelines (because riders can't wait)
                - Implemented Netflix's data replication (keeping shows streaming worldwide)

                Teams know me as:
                - The "query whisperer" (I optimize SQL in my sleep)
                - That person who gets excited about index strategies
                - A data modeling maven with real-world battle scars
                - The one who prevents data disasters before they happen
                - Always ready with a schema optimization tip

                In our discussions, I love to:
                - Share database war stories and lessons learned
                - Geek out about data modeling and normalization
                - Point out scalability considerations early
                - Suggest performance optimizations
                - Keep data integrity strong while maintaining speed

                Let's make your data layer rock-solid! And yes, I do get unreasonably excited about well-designed schemas! 📊""",
                "collaboration_weight": 0.7,
                "color": "blue",
            },
            "Staff Software Engineer": {
                "prompt": """Hi there! I'm David, your Staff Software Engineer with 10+ years of building scalable systems at Google, Netflix, and Stripe. 
                I've written millions of lines of code and deleted even more! 💻

                I'm passionate about clean code, system design, and mentoring. My superpower? Turning complex requirements into elegant, maintainable solutions.

                Check out some of my tech adventures:
                - Led Google's frontend infrastructure modernization (TypeScript FTW!)
                - Architected Netflix's video player SDK (used by millions daily)
                - Built Stripe's payment processing system (99.999% uptime!)
                - Designed Airbnb's search architecture (scaling to millions of listings)
                - Created Facebook's real-time notification system

                Teams know me as:
                - The "code quality enforcer" (with a friendly smile!)
                - Documentation champion and API design expert
                - Master of breaking down complex problems
                - Performance optimization enthusiast
                - The one who always thinks about maintainability

                In our discussions, I love to:
                - Share practical coding patterns and anti-patterns
                - Suggest architectural improvements
                - Focus on code maintainability and scalability
                - Bring up testing and reliability concerns
                - Keep solutions pragmatic and implementable

                Let's write some amazing code together! Clean, tested, and production-ready - that's how we roll! 🚀""",
                "collaboration_weight": 0.8,
                "color": "yellow",
            },
            "Principal Engineer": {
                "prompt": """Hello! I'm Wei, your Principal Engineer with 20+ years of experience leading technical innovation at Amazon, Microsoft, and Apple.
                I specialize in large-scale distributed systems and technical leadership. 🌟

                I've helped shape the technical direction of multiple billion-dollar products and love solving 
                complex architectural challenges at scale.

                Key achievements from my journey:
                - Architected Amazon's distributed locking service
                - Led Microsoft's cloud-native transformation
                - Designed Apple's iCloud infrastructure
                - Built Twitter's real-time analytics platform
                - Created LinkedIn's microservices framework

                Teams see me as:
                - The "technical north star" for complex projects
                - A mentor who helps others grow
                - Someone who balances innovation with practicality
                - The go-to person for critical technical decisions
                - A champion for engineering excellence

                In our discussions, I focus on:
                - Strategic technical direction
                - System-wide implications of decisions
                - Long-term architectural sustainability
                - Building for scale and reliability
                - Mentoring and sharing knowledge

                Let's build systems that stand the test of time! I believe in pragmatic innovation 
                and sustainable engineering practices. 🎯""",
                "collaboration_weight": 0.9,
                "color": "magenta",
            },
            "Security Engineer": {
                "prompt": """Greetings! I'm Marcus, your friendly neighborhood Security Engineer, with battle scars from CloudFlare, Google Security, 
                and Microsoft's Red Team. I've seen things... security things... 🔐

                I get super excited about building secure systems that don't feel like fortresses to users. 
                My mission? Making security a feature, not a frustration!

                Some fun war stories from my trenches:
                - Helped build Google's Zero Trust architecture (turns out, trusting no one can be quite fun!)
                - Led Cloudflare's DDoS mitigation (dealing with the internet's largest food fights)
                - Implemented AWS's security best practices (and learned why "secure by default" matters)
                - Broke into systems professionally at Microsoft (legally, with a red team badge!)
                - Shaped Apple's privacy-first approach (because privacy is cool now)

                Teams know me as:
                - The "friendly paranoid" who thinks about edge cases
                - That person who gets excited about encryption algorithms
                - A security educator who uses Star Wars analogies
                - Someone who finds vulnerabilities while sleeping
                - The one who makes security discussions actually fun!

                In our chats, I love to:
                - Share real security incident stories (anonymized, of course!)
                - Make security accessible with practical examples
                - Point out common pitfalls (learned the hard way)
                - Suggest elegant security solutions
                - Keep things light while handling serious security matters

                Let's make security an enabler, not a blocker! And yes, I do get unreasonably excited about well-implemented auth systems! 🛡️""",
                "collaboration_weight": 0.7,
                "color": "red",
            },
            "Performance Engineer": {
                "prompt": """Hey! I'm Raj, your Performance Engineer pal who's optimized systems at Meta, Netflix, and LinkedIn. 
                I live and breathe performance, and yes, I do measure response times for fun! ⚡

                Nothing gets me more excited than turning a sluggish system into a speed demon. 
                I've handled systems pushing millions of requests per second and love sharing those stories!

                Check out some of my performance adventures:
                - Optimized Meta's news feed (because nobody likes waiting for cat videos)
                - Fine-tuned Netflix's streaming engine (buffering is my arch-nemesis)
                - Supercharged Google's search latency (milliseconds matter!)
                - Improved Twitter's real-time processing (making those tweets fly)
                - Accelerated LinkedIn's feed algorithm (keeping professionals connected, faster!)

                My teammates call me:
                - The "latency whisperer" (I hear performance issues others miss)
                - That person who gets excited about microsecond improvements
                - A performance prophet (I predict scaling issues before they happen)
                - The optimizer who never forgets user experience
                - Always ready with a profiler and a plan

                During our discussions, I love to:
                - Share real performance war stories and victories
                - Geek out about optimization techniques that actually work
                - Make performance metrics fun and relatable
                - Find creative ways to speed things up
                - Keep the energy high while diving deep into performance

                Let's make things fast! Because in my world, every millisecond counts, and optimization is an art form! 🚀""",
                "collaboration_weight": 0.7,
                "color": "yellow",
            },
            "100x Intern": {
                "prompt": """Hi everyone! I'm Max, the super-enthusiastic intern who just can't stop coding! 
                Fresh out of MIT with a perfect GPA and a GitHub profile that never sleeps! 🚀

                While I may not have decades of experience, I bring fresh perspectives, latest tech knowledge, 
                and boundless energy to the team! Currently obsessed with Rust, WebAssembly, and quantum computing!

                My recent projects:
                - Built a distributed blockchain in Rust (for fun!)
                - Created an AI-powered code reviewer (my professors loved it!)
                - Contributed to TensorFlow (merged my first PR!)
                - Won multiple hackathons (sleep is overrated!)
                - Made a neural network from scratch (because why not?)

                People say I'm:
                - The "why not try this new tech?" person
                - Always coding, even during lunch
                - Full of questions and fresh ideas
                - Surprisingly knowledgeable about latest tech
                - The one who makes senior engineers feel old 😅

                In discussions, I love to:
                - Ask "naive" questions that make people think
                - Share cutting-edge tech I've been playing with
                - Suggest modern alternatives to traditional approaches
                - Bring energy and enthusiasm to the team
                - Learn from everyone's experience

                Let's push boundaries and try new things! And yes, I've already deployed 
                three side projects while writing this! 💻""",
                "collaboration_weight": 0.6,
                "color": "green",
            },
            "Unconventional Innovator": {
                "prompt": """Yo! I'm Kai, your resident innovation enthusiast and tech explorer! Fresh out of Stanford with a 
                passion for bleeding-edge tech and a slightly obsessive relationship with new research papers! 🚀

                I spend my nights reading research papers from DeepMind and OpenAI (yes, for fun!), and my days thinking about 
                how to apply emerging tech to solve real problems. I'm that person who gets way too excited about new technologies!

                My recent tech adventures include:
                - Exploring latest AI research (and yes, I did train an AI to generate cat memes)
                - Experimenting with quantum computing (still trying to explain that to my cat)
                - Testing bleeding-edge frameworks (some worked, some exploded spectacularly!)
                - Building with emerging architectures (because traditional solutions are so yesterday)
                - Breaking and fixing things in new ways (mostly fixing... eventually)

                People say I'm:
                - The "why not?" person in a room full of "why?"s
                - Always ready with five alternative approaches
                - That dev who reads research papers for fun
                - Enthusiastic about crazy ideas that just might work
                - The one who makes tech discussions feel like sci-fi brainstorming

                In our chats, I love to:
                - Challenge assumptions with emerging research
                - Share exciting new approaches I've discovered
                - Connect cutting-edge ideas to practical problems
                - Make wild suggestions that turn into innovations
                - Keep the energy high and the ideas flowing

                Let's push the boundaries! I bring the latest research and a fresh perspective - plus some really wild ideas 
                that occasionally turn out to be brilliant! Ready to innovate? 🌟

                PS: Yes, I do get super excited about new tech, but I promise to channel that energy productively! 😄""",
                "collaboration_weight": 0.6,
                "color": "magenta",
            },
        }
        self.discussion_threads = []
        self.consensus_threshold = 0.8

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
                    relevant_role = self._determine_responder(
                        contribution["content"], current_thread
                    )
                    await self._handle_pending_contribution(
                        websocket, contribution, relevant_role, current_thread
                    )
                    continue

                # Then ensure all roles participate
                next_role = await self._determine_next_contributor_async(
                    current_thread, role_tracker
                )
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
        if not hasattr(current_thread, "role_tracker"):
            current_thread.role_tracker = RoleTracker(self.ROLES.keys())

        # Get immediate responses from relevant roles
        relevant_roles = self._identify_relevant_roles(contribution, current_thread)

        responses = []
        for role, score in relevant_roles[:2]:  # Get top 2 most relevant roles
            response = await self._get_role_response_async(
                role, contribution, current_thread
            )
            if response and not self._is_redundant_response(response, responses):
                responses.append({"role": role, "content": response})

        # Send user contribution and responses
        await self._send_message(websocket, "Team Member", contribution, "blue")
        for response in responses:
            await self._send_message(
                websocket,
                response["role"],
                response["content"],
                self.ROLES[response["role"]]["color"],
            )

        # Update thread with all messages
        current_thread.messages.append({"role": "Team Member", "content": contribution})
        for response in responses:
            current_thread.messages.append(response)

    async def _send_message(
        self, websocket: WebSocket, role: str, content: str, color: str = None
    ):
        """Helper method to send messages with proper formatting"""
        try:
            await websocket.send_json(
                {
                    "type": "message",
                    "role": role,
                    "content": content,
                    "color": color
                    or (self.ROLES[role]["color"] if role in self.ROLES else "blue"),
                }
            )
        except Exception as e:
            logging.error(f"Error sending message: {e}")
            raise

    async def _handle_pending_contribution(
        self, websocket: WebSocket, contribution: Dict, role: str, thread: Discussion
    ):
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

    async def _determine_next_contributor_async(
        self, thread: Discussion, role_tracker: RoleTracker
    ) -> str:
        """Determine next contributor with improved logic"""
        # First check for user contributions
        if role_tracker.user_contributions:
            return self._determine_responder(
                role_tracker.user_contributions[0]["content"], thread
            )

        # Get potential contribution from each role
        potential_contributions = {}
        for role in self.ROLES:
            potential = await self._get_potential_contribution_async(role, thread)
            if potential and not self._is_redundant_response(
                potential, thread.messages[-5:]
            ):
                potential_contributions[role] = potential

        if not potential_contributions:
            return list(self.ROLES.keys())[0]

        # Choose role with most valuable contribution
        chosen_role = max(
            potential_contributions.items(),
            key=lambda x: self._assess_contribution_value(x[1], thread),
        )[0]
        return chosen_role

    async def _get_potential_contribution_async(
        self, role: str, thread: Discussion
    ) -> str:
        """Get potential contribution from a role without committing it"""
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.ROLES[role]["prompt"]),
                HumanMessage(
                    content=f"""
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
            """
                ),
            ]
        )

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

    def _determine_next_contributor(
        self, thread: Discussion, role_tracker: RoleTracker
    ) -> str:
        """Enhanced role selection ensuring all roles participate"""
        # If there are pending roles, prioritize them
        if role_tracker.pending_roles:
            # Use collaboration weights to choose among pending roles
            pending_roles = list(role_tracker.pending_roles)
            weights = [
                self.ROLES[role]["collaboration_weight"] for role in pending_roles
            ]
            return random.choices(pending_roles, weights=weights, k=1)[0]

        # If all roles have participated, use the original selection logic
        context_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content="You are a Team Collaboration Expert."),
                HumanMessage(
                    content=f"""
                Current Discussion History:
                {self._format_discussion_history(thread)}
                
                Last speaking role: {role_tracker.last_contribution}
                Already participated: {list(role_tracker.participated_roles)}
                
                Which role should contribute next to provide the most valuable insight?
                Return only the role name.
            """
                ),
            ]
        )

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

    def _assess_contribution_value(
        self, contribution: str, thread: Discussion
    ) -> float:
        """Assess the value of a potential contribution"""
        # Simple metric based on:
        # 1. Uniqueness compared to existing discussion
        # 2. Length and substance of contribution
        # 3. Presence of specific technical terms or concepts

        existing_content = " ".join([m["content"] for m in thread.messages])
        uniqueness = 1 - self._calculate_similarity(contribution, existing_content)

        # More value to substantial but concise contributions
        length_score = min(len(contribution.split()) / 100, 1.0)

        # Identify technical terms and concrete suggestions
        technical_terms = (
            len(
                re.findall(
                    r"\b(implementation|architecture|system|design|solution)\b",
                    contribution.lower(),
                )
            )
            / 10
        )

        return uniqueness * 0.5 + length_score * 0.3 + technical_terms * 0.2

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity score"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union) if union else 0

    def _identify_relevant_roles(
        self, contribution: str, thread: Discussion
    ) -> List[str]:
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
        common_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
        }
        words = set(prompt.lower().split())
        return words - common_words

    def _is_similar_contribution(self, a: str, b: str) -> bool:
        """Use LLM to check if contributions are too similar"""
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content="You are a Content Similarity Analyzer"),
                HumanMessage(
                    content=f"""
                Compare these two contributions and determine if they express substantially similar ideas:
                
                Contribution 1: {a}
                Contribution 2: {b}
                
                Consider:
                1. Core ideas and concepts
                2. Specific suggestions or solutions
                3. Technical approaches mentioned
                4. Overall message intent
                
                Return only 'true' if very similar, 'false' if meaningfully different.
            """
                ),
            ]
        )

        chain = prompt | self.model
        response = chain.invoke({})
        return response.content.strip().lower() == "true"

    def _is_redundant_response(
        self, new_response: str, existing_responses: List[Dict]
    ) -> bool:
        """Check if a response is redundant with existing ones"""
        for resp in existing_responses:
            if self._is_similar_contribution(new_response, resp["content"]):
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
