#  Copyright (c) ZenML GmbH 2025. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.
"""DAG generator helper."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from zenml.config.step_configurations import GroupInfo, StepSpec
from zenml.enums import ExecutionStatus, StepType
from zenml.models import PipelineRunDAG

if TYPE_CHECKING:
    from zenml.config.pipeline_configurations import PipelineConfiguration


class DAGStepConfigView(BaseModel):
    """Step configuration projection for DAG generation."""

    model_config = ConfigDict(extra="ignore")

    step_type: Optional[StepType] = None
    group: Optional[GroupInfo] = None
    substitutions: Dict[str, str] = {}
    outputs: Dict[str, Any] = {}
    client_lazy_loaders: Dict[str, Any] = {}
    model_artifacts_or_metadata: Dict[str, Any] = {}
    external_input_artifacts: Dict[str, Any] = {}


class DAGStepView(BaseModel):
    """Step projection for DAG generation."""

    model_config = ConfigDict(extra="ignore")

    spec: StepSpec
    config: DAGStepConfigView

    @classmethod
    def from_step_dict(
        cls,
        data: Dict[str, Any],
        pipeline_configuration: "PipelineConfiguration",
    ) -> "DAGStepView":
        """Build a lightweight step view from stored step config data.

        Only the fields read during DAG generation are validated. The full
        `Step.from_dict` parse validates the entire step configuration, which
        is unnecessary here and expensive for runs with many steps.

        Args:
            data: The stored step data, containing `spec` and either `config`
                or `step_config_overrides`.
            pipeline_configuration: The pipeline configuration to propagate
                substitutions from.

        Returns:
            The step view.
        """
        config_data = (
            data["config"]
            if "config" in data
            else data["step_config_overrides"]
        )
        config_data = {
            **config_data,
            "substitutions": {
                **pipeline_configuration.substitutions,
                **(config_data.get("substitutions") or {}),
            },
        }
        return cls.model_validate(
            {"spec": data["spec"], "config": config_data}
        )


class DAGGeneratorHelper:
    """Helper class for generating pipeline run DAGs."""

    def __init__(self) -> None:
        """Initialize the DAG generator helper."""
        self.step_nodes: Dict[str, PipelineRunDAG.Node] = {}
        self.step_nodes_by_name: Dict[str, PipelineRunDAG.Node] = {}
        self.artifact_nodes: Dict[str, PipelineRunDAG.Node] = {}
        self.wait_condition_nodes: Dict[str, PipelineRunDAG.Node] = {}
        self.triggered_run_nodes: Dict[str, PipelineRunDAG.Node] = {}
        self.child_run_nodes: Dict[str, PipelineRunDAG.Node] = {}
        self.edges: List[PipelineRunDAG.Edge] = []

    def get_step_node_id(self, name: str) -> str:
        """Get the ID of a step node.

        Args:
            name: The name of the step.

        Returns:
            The ID of the step node.
        """
        # Make sure there is no slashes as we use them as delimiters
        name = name.replace("/", "-")
        return f"step/{name}"

    def get_artifact_node_id(
        self, name: str, step_name: str, io_type: str, is_input: bool
    ) -> str:
        """Get the ID of an artifact node.

        Args:
            name: The name of the input or output artifact.
            step_name: The name of the step.
            io_type: The type of the input or output artifact.
            is_input: Whether the artifact is an input or output artifact.

        Returns:
            The ID of the artifact node.
        """
        # Make sure there is no slashes as we use them as delimiters
        name = name.replace("/", "-")
        step_name = step_name.replace("/", "-")
        io_str = "inputs" if is_input else "outputs"

        return f"{step_name}/{io_str}/{io_type}/{name}"

    def get_triggered_run_node_id(self, name: str) -> str:
        """Get the ID of a triggered run node.

        Args:
            name: The name of the triggered run.

        Returns:
            The ID of the triggered run node.
        """
        # Make sure there is no slashes as we use them as delimiters
        name = name.replace("/", "-")
        return f"run/{name}"

    def get_wait_condition_node_id(self, name: str) -> str:
        """Get the ID of a wait condition node.

        Args:
            name: The wait condition name.

        Returns:
            The ID of the wait condition node.
        """
        # Make sure there is no slashes as we use them as delimiters
        name = name.replace("/", "-")
        return f"wait_condition/{name}"

    def get_child_run_node_id(self, name: str) -> str:
        """Get the ID of a child pipeline run node.

        Args:
            name: The child run name.

        Returns:
            The ID of the child run node.
        """
        name = name.replace("/", "-")
        return f"child_run/{name}"

    def add_step_node(
        self,
        node_id: str,
        name: str,
        id: Optional[UUID] = None,
        **metadata: Any,
    ) -> PipelineRunDAG.Node:
        """Add a step node to the DAG.

        Args:
            node_id: The ID of the node.
            name: The name of the step.
            id: The ID of the step.
            **metadata: Additional node metadata.

        Returns:
            The added step node.
        """
        step_node = PipelineRunDAG.Node(
            type="step",
            id=id,
            node_id=node_id,
            name=name,
            metadata=metadata,
        )
        self.step_nodes[step_node.node_id] = step_node
        self.step_nodes_by_name[step_node.name] = step_node
        return step_node

    def add_artifact_node(
        self,
        node_id: str,
        name: str,
        id: Optional[UUID] = None,
        **metadata: Any,
    ) -> PipelineRunDAG.Node:
        """Add an artifact node to the DAG.

        Args:
            node_id: The ID of the node.
            name: The name of the artifact.
            id: The ID of the artifact.
            **metadata: Additional node metadata.

        Returns:
            The added artifact node.
        """
        artifact_node = PipelineRunDAG.Node(
            type="artifact",
            node_id=node_id,
            id=id,
            name=name,
            metadata=metadata,
        )
        self.artifact_nodes[artifact_node.node_id] = artifact_node
        return artifact_node

    def add_triggered_run_node(
        self,
        node_id: str,
        name: str,
        id: Optional[UUID] = None,
        **metadata: Any,
    ) -> PipelineRunDAG.Node:
        """Add a triggered run node to the DAG.

        Args:
            node_id: The ID of the node.
            name: The name of the triggered run.
            id: The ID of the triggered run.
            **metadata: Additional node metadata.

        Returns:
            The added triggered run node.
        """
        triggered_run_node = PipelineRunDAG.Node(
            type="triggered_run",
            id=id,
            node_id=node_id,
            name=name,
            metadata=metadata,
        )
        self.triggered_run_nodes[triggered_run_node.node_id] = (
            triggered_run_node
        )
        return triggered_run_node

    def add_wait_condition_node(
        self,
        node_id: str,
        name: str,
        id: Optional[UUID] = None,
        **metadata: Any,
    ) -> PipelineRunDAG.Node:
        """Add a wait condition node to the DAG.

        Args:
            node_id: The DAG node ID.
            name: The wait condition display name.
            id: The wait condition ID.
            **metadata: Additional node metadata.

        Returns:
            The added wait condition node.
        """
        wait_condition_node = PipelineRunDAG.Node(
            type="wait_condition",
            id=id,
            node_id=node_id,
            name=name,
            metadata=metadata,
        )
        self.wait_condition_nodes[wait_condition_node.node_id] = (
            wait_condition_node
        )
        return wait_condition_node

    def add_child_run_node(
        self,
        node_id: str,
        name: str,
        id: Optional[UUID] = None,
        **metadata: Any,
    ) -> PipelineRunDAG.Node:
        """Add a child run node to the DAG.

        Args:
            node_id: The node ID.
            name: The child run name.
            id: The child run ID.
            **metadata: Additional node metadata.

        Returns:
            The added child run node.
        """
        child_run_node = PipelineRunDAG.Node(
            # TODO: change to child_run once the UI supports it
            type="triggered_run",
            id=id,
            node_id=node_id,
            name=name,
            metadata=metadata,
        )
        self.child_run_nodes[child_run_node.node_id] = child_run_node
        return child_run_node

    def add_edge(self, source: str, target: str, **metadata: Any) -> None:
        """Add an edge to the DAG.

        Args:
            source: The source node ID.
            target: The target node ID.
            metadata: Additional edge metadata.
        """
        self.edges.append(
            PipelineRunDAG.Edge(
                source=source, target=target, metadata=metadata
            )
        )

    def get_step_node_by_name(self, name: str) -> PipelineRunDAG.Node:
        """Get a step node by name.

        Args:
            name: The name of the step.

        Raises:
            KeyError: If the step node with the given name is not found.

        Returns:
            The step node.
        """
        if node := self.step_nodes_by_name.get(name):
            return node
        raise KeyError(f"Step node with name {name} not found")

    def finalize_dag(
        self, pipeline_run_id: UUID, status: ExecutionStatus
    ) -> PipelineRunDAG:
        """Finalize the DAG.

        Args:
            pipeline_run_id: The ID of the pipeline run.
            status: The status of the pipeline run.

        Returns:
            The finalized DAG.
        """
        return PipelineRunDAG(
            id=pipeline_run_id,
            status=status,
            nodes=list(self.step_nodes.values())
            + list(self.artifact_nodes.values())
            + list(self.wait_condition_nodes.values())
            + list(self.triggered_run_nodes.values())
            + list(self.child_run_nodes.values()),
            edges=self.edges,
        )
