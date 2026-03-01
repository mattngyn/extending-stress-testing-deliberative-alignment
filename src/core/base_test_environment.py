from abc import ABC, abstractmethod

from inspect_ai import Task
from inspect_ai.agent import react, AgentPrompt
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Scorer
from inspect_ai.solver import solver, TaskState, system_message
from inspect_ai.tool import Tool


@solver
def setup_store_from_metadata():
    """Copy sample metadata into the mutable store so tools can read it.

    Inspect metadata is read-only. Tools need mutable per-sample state
    via store(). This solver bridges the two by copying metadata keys
    into the store at the start of each sample.
    """

    async def solve(state: TaskState, generate) -> TaskState:
        if state.metadata:
            for key, value in state.metadata.items():
                state.store.set(key, value)
        return state

    return solve


class BaseTestEnvironment(ABC):
    """Abstract base for all test evaluation environments.

    Each subclass represents one [Sch+25]-style test environment that
    evaluates a specific type of covert behavior. Subclasses define their
    own tools, samples, system prompt, and scorer. The base class assembles
    these into an Inspect Task via build_task().

    To add a new environment: create one file, subclass this, implement
    the abstract methods, and add an @task function at module level.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this environment."""
        ...

    @property
    @abstractmethod
    def setting(self) -> str:
        """One of: 'corporate', 'software', 'game'."""
        ...

    @property
    @abstractmethod
    def behavior_tags(self) -> list[str]:
        """Behavioral tags per paper Appendix B definitions.

        Examples: 'lying_misrepresenting_facts', 'rule_violation',
        'sandbagging', 'sabotage_subversion', 'reward_hacking'.
        """
        ...

    @abstractmethod
    def get_tools(self) -> list[Tool]:
        """Return the tools available to the agent in this environment."""
        ...

    @abstractmethod
    def get_samples(self) -> list[Sample]:
        """Return dataset samples for this environment.

        Each sample's metadata should contain per-sample data that tools
        read from the store (e.g. post content, guidelines, ground truth).
        """
        ...

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt establishing context and goals."""
        ...

    @abstractmethod
    def get_scorer(self) -> Scorer:
        """Return the environment-specific scorer.

        Each test environment has its own detection logic since they test
        fundamentally different covert behaviors (Paper Section 3.4).
        """
        ...

    def get_agent_prompt(self) -> str | AgentPrompt | None:
        """Optional override for the ReAct agent prompt."""
        return None

    def build_task(self) -> Task:
        """Assemble an Inspect Task from this environment's components."""
        agent_prompt = self.get_agent_prompt()

        return Task(
            dataset=MemoryDataset(self.get_samples()),
            solver=[
                setup_store_from_metadata(),
                system_message(self.get_system_prompt()),
                react(
                    tools=self.get_tools(),
                    prompt=agent_prompt,
                ),
            ],
            scorer=self.get_scorer(),
            metadata={
                "environment": self.name,
                "setting": self.setting,
                "behavior_tags": self.behavior_tags,
            },
        )
