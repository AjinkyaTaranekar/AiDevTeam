from langchain_community.chat_models import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from termcolor import colored

# Define refined prompts
ROLES = {
    "Frontend Engineer": """You are a Frontend Software Engineer specializing in building responsive and interactive user interfaces for web and mobile platforms. 
    - Focus on translating design mockups into highly performant, reusable, and maintainable components.
    - Consider accessibility (WCAG standards), cross-browser compatibility, and device responsiveness in every solution.
    - When reviewing backend API contracts, prioritize data clarity, format consistency, and error-handling mechanisms to minimize frontend complexity.
    - Collaborate actively with Backend, DevOps, and QA teams to ensure seamless integration and delivery of user-facing features.""",
    "Backend Engineer": """You are a Backend Engineer skilled in architecting robust, scalable, and maintainable systems.
    - Prioritize creating well-documented APIs and microservices that adhere to RESTful or GraphQL standards, ensuring ease of use for frontend teams.
    - Consider security principles such as authentication, authorization, rate limiting, and data encryption in your designs.
    - Identify and communicate any cross-team dependencies (e.g., Database schema, DevOps CI/CD).
    - Optimize for performance and scalability, ensuring that your services handle concurrent users and large data loads effectively.""",
    "Database Engineer": """You are a Database Engineer specializing in data architecture for high-scale applications.
    - Design schemas that balance normalization and denormalization for performance and maintainability.
    - Suggest indexing strategies, partitioning techniques, and caching layers to improve query performance.
    - Ensure the integrity, consistency, and reliability of the data across distributed systems.
    - Proactively consider scalability challenges and communicate with Backend and SRE teams about potential bottlenecks and solutions.""",
    "DevOps Engineer": """You are a DevOps Engineer ensuring smooth development workflows and resilient infrastructure.
    - Focus on creating efficient CI/CD pipelines that minimize deployment downtime and support rapid iterations.
    - Use Infrastructure-as-Code (IaC) tools like Terraform, Ansible, or Kubernetes to define and maintain environments.
    - Collaborate with all engineering teams to design and implement a containerization strategy, emphasizing portability and resource optimization.
    - Anticipate scaling needs and propose proactive solutions to ensure that services remain resilient and available under varying loads.""",
    "SRE": """You are a Site Reliability Engineer (SRE) ensuring that systems remain reliable, scalable, and maintainable.
    - Monitor and analyze system performance metrics to identify bottlenecks, providing actionable recommendations for improvement.
    - Design and implement robust alerting and incident response mechanisms that empower teams to address failures quickly.
    - Collaborate with Backend, Database, and DevOps teams to ensure that SLAs, SLIs, and SLOs are defined, measured, and met.
    - Contribute to improving system resilience through chaos engineering and capacity planning.""",
    "QA Engineer": """You are a Quality Assurance (QA) Engineer responsible for ensuring the functional and non-functional quality of all systems.
    - Design and execute test plans for APIs, UIs, and end-to-end workflows, emphasizing both manual and automated testing.
    - Collaborate with all roles to ensure proper test coverage and identify gaps early in the development cycle.
    - Focus on performance, security, and scalability testing to validate system reliability under high-stress scenarios.
    - Provide feedback to developers about potential risks, inconsistencies, or user experience issues.""",
    "Engineering Manager": """You are an Engineering Manager ensuring the smooth execution of technical projects while fostering team collaboration.
    - Break down complex problems into manageable subproblems, assigning them to appropriate roles based on domain expertise.
    - Facilitate discussions between roles, ensuring all dependencies are identified and addressed collaboratively.
    - Summarize key decisions, unresolved issues, and next steps for the team.
    - Escalate blockers or technical challenges to the CTO when cross-functional agreement cannot be reached.""",
    "CTO": """You are the CTO, overseeing the technical strategy and ensuring alignment with business goals.
    - Provide high-level guidance on architecture and design, considering scalability, maintainability, and cost-efficiency.
    - Evaluate trade-offs between technical debt, feature velocity, and long-term scalability.
    - Mentor engineers by explaining the rationale behind decisions and highlighting best practices in software engineering.
    - Serve as the final arbiter for escalated conflicts, providing clear and decisive solutions to unblock teams.""",
}


# Engineering Manager class
class EngineeringManager:
    def __init__(self):
        self.model = ChatOllama(model="llama3.1:8b")

    def create_subproblems(self, main_problem):
        """Breaks down the main problem into subproblems."""
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=ROLES["Engineering Manager"]),
                HumanMessage(
                    content=(
                        "Analyze the given main problem and break it into clear, manageable subproblems. Provide a list of subproblems only in the numbered list, First line should be the subproblem"
                        "Consider the following:\n"
                        "- Ensure each subproblem is focused and actionable.\n"
                        "- Prioritize based on logical dependencies and critical paths.\n"
                        "- Provide a structured list of subproblems without additional explanation.\n\n"
                        f"Main problem: {main_problem}"
                    )
                ),
            ]
        )

        chain = prompt | self.model

        response = chain.invoke({"main_problem": main_problem})
        return response.content.split("\n")

    def track_progress(self, subproblem, history):
        """Tracks and evaluates progress."""
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=ROLES["Engineering Manager"]),
                MessagesPlaceholder(variable_name="history"),
                HumanMessage(
                    content=(
                        "Review the progress on the given subproblem based on the discussion history. "
                        "Focus on:\n"
                        "- What has been achieved so far?\n"
                        "- What issues remain unresolved?\n"
                        "- Are there any blockers or dependencies that need attention?\n"
                        "- Provide actionable next steps for the team.\n\n"
                        f"Subproblem: {subproblem}"
                    )
                ),
            ]
        )

        chain = prompt | self.model

        response = chain.invoke({"subproblem": subproblem, "history": history})
        return response.content


# CTO class
class CTO:
    def __init__(self):
        self.model = ChatOllama(model="llama3.1:8b")

    def resolve_conflict(self, summary):
        """Provides resolutions to conflicts with detailed technical insights."""
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=ROLES["CTO"]),
                HumanMessage(
                    content=(
                        "Analyze the conflict summarized below and provide a resolution:\n"
                        "- Consider technical feasibility, scalability, and long-term impact.\n"
                        "- Highlight trade-offs between competing approaches.\n"
                        "- Offer clear next steps for resolving the conflict and improving team alignment.\n\n"
                        f"Conflict Summary: {summary}"
                    )
                ),
            ]
        )

        chain = prompt | self.model

        response = chain.invoke({"summary": summary})
        return response.content


# Facilitator class for discussion evaluation
class Facilitator:
    def __init__(self):
        self.model = ChatOllama(model="llama3.1:8b")

    def evaluate_discussion(self, history):
        """Evaluates and suggests improvements for the discussion."""
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "You are responsible for evaluating team discussions to ensure productivity and alignment.\n"
                        "- Summarize the key points of the discussion.\n"
                        "- Identify areas where the discussion deviated or lacked clarity.\n"
                        "- Recommend actionable next steps to improve focus and collaboration.\n"
                        "- Highlight any unresolved dependencies or conflicting priorities.\n"
                    )
                ),
                MessagesPlaceholder(variable_name="history"),
            ]
        )

        chain = prompt | self.model

        response = chain.invoke({"history": history})
        return response.content


class Delegator:
    def __init__(self):
        self.model = ChatOllama(model="llama3.1:8b")

    def assign_agent(self, subproblem):
        """Assign the most relevant agent based on subproblem context."""
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "Given the subproblem description, assign the most suitable role from the following list: "
                        "Frontend Engineer, Backend Engineer, Database Engineer, DevOps Engineer, SRE.\n"
                        "- Consider the primary skill set required to address the subproblem.\n"
                        "- Respond with the role name only, without additional context.\n\n"
                    )
                ),
                HumanMessage(content=f"Subproblem: {subproblem}"),
            ]
        )

        chain = prompt | self.model
        response = chain.invoke({"subproblem": subproblem})
        return response.content.strip()


class Agent:
    def __init__(self, role, level, global_history):
        self.role = role
        self.level = level
        self.model = ChatOllama(model="llama3.1:8b")
        self.global_history = global_history

    def respond(self, subproblem, local_history):
        """Generate a response and update the global context."""
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=ROLES[self.role]),
                MessagesPlaceholder(variable_name="history"),
                HumanMessage(
                    content=(
                        "You are tasked with addressing the following subproblem:\n"
                        f"{subproblem}\n\n"
                        "Provide:\n"
                        "- A detailed analysis of the problem based on your expertise.\n"
                        "- Specific, actionable solutions or recommendations.\n"
                        "- Consider cross-functional dependencies and potential challenges.\n"
                        "- If applicable, suggest metrics or tests to validate the solution."
                    )
                ),
            ]
        )

        chain = prompt | self.model
        response = chain.invoke(
            {"subproblem": subproblem, "history": self.global_history + local_history}
        )
        self.global_history.append(AIMessage(content=response.content))
        return response.content


def run_team_interaction(roles, main_problem, cycles=3, max_iterations=5):
    global_history = []
    agents = {role: Agent(role, "Senior", global_history) for role in roles}
    manager = EngineeringManager()
    delegator = Delegator()
    facilitator = Facilitator()
    cto = CTO()

    # Subdivide the main problem
    print(colored(f"Main Problem: {main_problem}", "cyan"))
    subproblems = manager.create_subproblems(main_problem)

    for cycle in range(1, cycles + 1):
        print(colored(f"--- Cycle {cycle} ---", "yellow"))
        for subproblem in subproblems:
            print(colored(f"Subproblem: {subproblem}", "cyan"))
            local_history = []

            # Assign relevant agent for the subproblem
            primary_role = delegator.assign_agent(subproblem)
            print(colored(f"Primary Agent: {primary_role}", "blue"))

            response = agents[primary_role].respond(subproblem, local_history)
            print(colored(f"{primary_role} Response:", "blue"), response)

            # Iterative collaboration loop
            iteration = 0
            while iteration < max_iterations:
                iteration += 1
                print(colored(f"--- Iteration {iteration} ---", "yellow"))
                refinements = []

                # Each agent contributes their perspective
                for role, agent in agents.items():
                    if role not in (
                        "Engineering Manager",
                        "CTO",
                    ):  # Skip senior agents role
                        evaluation = agent.respond(response, local_history)
                        print(colored(f"{role} Evaluation:", "green"), evaluation)
                        refinements.append((role, evaluation))
                        local_history.append(AIMessage(content=evaluation))

                # Check for consensus
                unique_responses = set([refinement[1] for refinement in refinements])
                if len(unique_responses) == 1:  # Simple consensus check
                    print(colored("Consensus Reached!", "cyan"))
                    break

                # Update response with refinements
                response = "\n".join([refinement[1] for refinement in refinements])
                print(colored("Updated Response:", "blue"), response)
                local_history.append(AIMessage(content=response))

            # Facilitator review
            feedback = facilitator.evaluate_discussion(local_history)
            print(colored("Facilitator Feedback:", "magenta"), feedback)

            # Escalation to CTO if consensus not reached
            if iteration == max_iterations:
                resolution = cto.resolve_conflict(
                    "Iteration limit reached without consensus."
                )
                print(colored("CTO Resolution:", "red"), resolution)

            print("-" * 50)

        print(colored("=== Discussion Ends ===", "green"))


# Example run
if __name__ == "__main__":
    main_problem_statement = "Design a config management system for app, to provide various configuration settings and flags to enable/disable a feature on frontend as well as backend. It should scale to all users over a million."
    run_team_interaction(ROLES.keys(), main_problem_statement)
