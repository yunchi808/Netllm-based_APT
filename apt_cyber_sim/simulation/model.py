"""Data model for the simulation environment.

Vendored from CyberBattleSim to make APT evaluation self-contained.
"""

from datetime import datetime, time
from typing import NamedTuple, List, Dict, Optional, Union, Tuple, Iterator
import dataclasses
from dataclasses import dataclass

# Optional visualization dependency (not required for env stepping)
try:  # pragma: no cover
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore

from enum import Enum, IntEnum
import boolean
import networkx as nx

# Optional YAML dependency (only used for serialization helpers)
try:  # pragma: no cover
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

import random

VERSION_TAG = "0.1.0"

ALGEBRA = boolean.BooleanAlgebra()

# Type alias for identifiers
NodeID = str

# A unique identifier
ID = str

# a (login,password/token) credential pair is abstracted as just a unique
# string identifier
CredentialID = str

# Intrinsic value of a reaching a given node in [0,100]
NodeValue = int


PortName = str


@dataclass
class ListeningService:
    """A service port on a given node accepting connection initiated
    with the specified allowed credentials """

    # Name of the port the service is listening to
    name: PortName
    # credential allowed to authenticate with the service
    allowedCredentials: List[CredentialID] = dataclasses.field(default_factory=list)
    # whether the service is running or stopped
    running: bool = True
    # Weight used to evaluate the cost of not running the service
    sla_weight = 1.0


x = ListeningService(name="d")
VulnerabilityID = str

# Probability rate
Probability = float

# The name of a node property indicating the presence of a
# service, component, feature or vulnerability on a given node.
PropertyName = str


class Rates(NamedTuple):
    """Probabilities associated with a given vulnerability"""

    probingDetectionRate: Probability = 0.0
    exploitDetectionRate: Probability = 0.0
    successRate: Probability = 1.0


class VulnerabilityType(Enum):
    """Is the vulnerability exploitable locally or remotely?"""

    LOCAL = 1
    REMOTE = 2


class PrivilegeLevel(IntEnum):
    """Access privilege level on a given node"""

    NoAccess = 0
    LocalUser = 1
    Admin = 2
    System = 3
    MAXIMUM = 3


def escalate(current_level, escalation_level: "PrivilegeLevel") -> "PrivilegeLevel":
    return PrivilegeLevel(max(int(current_level), int(escalation_level)))


class VulnerabilityOutcome:
    """Outcome of exploiting a given vulnerability"""


class LateralMove(VulnerabilityOutcome):
    """Lateral movement to the target node"""

    success: bool


class CustomerData(VulnerabilityOutcome):
    """Access customer data on target node"""


class PrivilegeEscalation(VulnerabilityOutcome):
    """Privilege escalation outcome"""

    def __init__(self, level: PrivilegeLevel):
        self.level = level

    @property
    def tag(self):
        """Escalation tag that gets added to node properties when
        the escalation level is reached for that node"""

        return f"privilege_{self.level}"


class SystemEscalation(PrivilegeEscalation):
    """Escalation to SYSTEM privileges"""

    def __init__(self):
        super().__init__(PrivilegeLevel.System)


class AdminEscalation(PrivilegeEscalation):
    """Escalation to local administrator privileges"""

    def __init__(self):
        super().__init__(PrivilegeLevel.Admin)


class ProbeSucceeded(VulnerabilityOutcome):
    """Probing succeeded"""

    def __init__(self, discovered_properties: List[PropertyName]):
        self.discovered_properties = discovered_properties


class ProbeFailed(VulnerabilityOutcome):
    """Probing failed"""


class ExploitFailed(VulnerabilityOutcome):
    """This is for situations where the exploit fails """


class CachedCredential(NamedTuple):
    """Encodes a machine-port-credential triplet"""

    node: NodeID
    port: PortName
    credential: CredentialID


class LeakedCredentials(VulnerabilityOutcome):
    """A set of credentials obtained by exploiting a vulnerability"""

    credentials: List[CachedCredential]

    def __init__(self, credentials: List[CachedCredential]):
        self.credentials = credentials


class LeakedNodesId(VulnerabilityOutcome):
    """A set of node IDs obtained by exploiting a vulnerability"""

    def __init__(self, nodes: List[NodeID]):
        self.nodes = nodes


VulnerabilityOutcomes = Union[
    LeakedCredentials,
    LeakedNodesId,
    PrivilegeEscalation,
    AdminEscalation,
    SystemEscalation,
    CustomerData,
    LateralMove,
    ExploitFailed,
]


class AttackResult:
    """The result of attempting a specific attack (either local or remote)"""

    success: bool
    expected_outcome: Union[VulnerabilityOutcomes, None]


class Precondition:
    """A predicate logic expression defining the condition under which a given
    feature or vulnerability is present or not.

    The symbols used in the expression refer to properties associated with
    the corresponding node.
    """

    expression: boolean.Expression

    def __init__(self, expression: Union[boolean.Expression, str]):
        if isinstance(expression, boolean.Expression):
            self.expression = expression
        else:
            self.expression = ALGEBRA.parse(expression)


class VulnerabilityInfo(NamedTuple):
    """Definition of a known vulnerability"""

    description: str
    type: VulnerabilityType
    outcome: VulnerabilityOutcome
    precondition: Precondition = Precondition("true")
    rates: Rates = Rates()
    URL: str = ""
    cost: float = 1.0
    reward_string: str = ""


VulnerabilityLibrary = Dict[VulnerabilityID, VulnerabilityInfo]


class RulePermission(Enum):
    """Determine if a rule is blocks or allows traffic"""

    ALLOW = 0
    BLOCK = 1


class FirewallRule(NamedTuple):
    """A firewall rule"""

    port: PortName
    permission: RulePermission
    reason: str = ""


class FirewallConfiguration(NamedTuple):
    """Firewall configuration on a given node."""

    outgoing: List[FirewallRule] = [
        FirewallRule("RDP", RulePermission.ALLOW),
        FirewallRule("SSH", RulePermission.ALLOW),
        FirewallRule("HTTPS", RulePermission.ALLOW),
        FirewallRule("MySQL", RulePermission.ALLOW),
        FirewallRule("HTTP", RulePermission.ALLOW),
    ]
    incoming: List[FirewallRule] = [
        FirewallRule("RDP", RulePermission.ALLOW),
        FirewallRule("SSH", RulePermission.ALLOW),
        FirewallRule("HTTPS", RulePermission.ALLOW),
        FirewallRule("MySQL", RulePermission.ALLOW),
        FirewallRule("HTTP", RulePermission.ALLOW),
    ]


class MachineStatus(Enum):
    """Machine running status"""

    Stopped = 0
    Running = 1
    Imaging = 2


@dataclass
class NodeInfo:
    """A computer node in the enterprise network"""

    services: List[ListeningService]
    vulnerabilities: VulnerabilityLibrary = dataclasses.field(default_factory=dict)
    value: NodeValue = 0
    properties: List[PropertyName] = dataclasses.field(default_factory=list)
    firewall: FirewallConfiguration = FirewallConfiguration()
    agent_installed: bool = False
    privilege_level: PrivilegeLevel = PrivilegeLevel.NoAccess
    reimagable: bool = True
    last_reimaging: Optional[time] = None
    owned_string: str = ""
    status = MachineStatus.Running
    sla_weight: float = 1.0


class Identifiers(NamedTuple):
    """Define the global set of identifiers used in the definition of a given environment."""

    properties: List[PropertyName] = []
    ports: List[PortName] = []
    local_vulnerabilities: List[VulnerabilityID] = []
    remote_vulnerabilities: List[VulnerabilityID] = []


def iterate_network_nodes(network: nx.graph.Graph) -> Iterator[Tuple[NodeID, NodeInfo]]:
    for nodeid, nodevalue in network.nodes.items():
        node_data: NodeInfo = nodevalue["data"]
        yield nodeid, node_data


class Environment(NamedTuple):
    network: nx.DiGraph
    vulnerability_library: VulnerabilityLibrary
    identifiers: Identifiers
    creationTime: datetime = datetime.utcnow()
    lastModified: datetime = datetime.utcnow()
    version: str = VERSION_TAG

    def nodes(self) -> Iterator[Tuple[NodeID, NodeInfo]]:
        return iterate_network_nodes(self.network)

    def get_node(self, node_id: NodeID) -> NodeInfo:
        node_info: NodeInfo = self.network.nodes[node_id]["data"]
        return node_info

    def plot_environment_graph(self) -> None:
        if plt is None:  # pragma: no cover
            raise ImportError("matplotlib is required for plot_environment_graph()")
        nx.draw(
            self.network,
            with_labels=True,
            node_color=[n["data"].value for i, n in self.network.nodes.items()],
            cmap=plt.cm.Oranges,  # type: ignore
        )


def create_network(nodes: Dict[NodeID, NodeInfo]) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from([(k, {"data": v}) for (k, v) in list(nodes.items())])
    return graph


def collect_ports_from_vuln(vuln: VulnerabilityInfo) -> List[PortName]:
    if isinstance(vuln.outcome, LeakedCredentials):
        return [c.port for c in vuln.outcome.credentials]
    else:
        return []


def collect_vulnerability_ids_from_nodes_bytype(
    nodes: Iterator[Tuple[NodeID, NodeInfo]],
    global_vulnerabilities: VulnerabilityLibrary,
    type: VulnerabilityType,
) -> List[VulnerabilityID]:
    return sorted(
        list(
            {
                id
                for _, node_info in nodes
                for id, v in node_info.vulnerabilities.items()
                if v.type == type
            }.union(id for id, v in global_vulnerabilities.items() if v.type == type)
        )
    )


def collect_properties_from_nodes(nodes: Iterator[Tuple[NodeID, NodeInfo]]) -> List[PropertyName]:
    return sorted({p for _, node_info in nodes for p in node_info.properties})


def collect_ports_from_nodes(
    nodes: Iterator[Tuple[NodeID, NodeInfo]], vulnerability_library: VulnerabilityLibrary
) -> List[PortName]:
    return sorted(
        list(
            {
                port
                for _, v in vulnerability_library.items()
                for port in collect_ports_from_vuln(v)
            }.union(
                {
                    port
                    for _, node_info in nodes
                    for _, v in node_info.vulnerabilities.items()
                    for port in collect_ports_from_vuln(v)
                }.union({service.name for _, node_info in nodes for service in node_info.services})
            )
        )
    )


def collect_ports_from_environment(environment: Environment) -> List[PortName]:
    return collect_ports_from_nodes(environment.nodes(), environment.vulnerability_library)


def infer_constants_from_nodes(
    nodes: Iterator[Tuple[NodeID, NodeInfo]], vulnerabilities: Dict[VulnerabilityID, VulnerabilityInfo]
) -> Identifiers:
    return Identifiers(
        properties=collect_properties_from_nodes(nodes),
        ports=collect_ports_from_nodes(nodes, vulnerabilities),
        local_vulnerabilities=collect_vulnerability_ids_from_nodes_bytype(
            nodes, vulnerabilities, VulnerabilityType.LOCAL
        ),
        remote_vulnerabilities=collect_vulnerability_ids_from_nodes_bytype(
            nodes, vulnerabilities, VulnerabilityType.REMOTE
        ),
    )


def infer_constants_from_network(network: nx.Graph, vulnerabilities: Dict[VulnerabilityID, VulnerabilityInfo]) -> Identifiers:
    return infer_constants_from_nodes(iterate_network_nodes(network), vulnerabilities)


SAMPLE_IDENTIFIERS = Identifiers(
    ports=["RDP", "SSH", "SMB", "HTTP", "HTTPS", "WMI", "SQL"],
    properties=["Windows", "Linux", "HyperV-VM", "Azure-VM", "Win7", "Win10", "PortRDPOpen", "GuestAccountEnabled"],
)


def assign_random_labels(
    graph: nx.DiGraph, vulnerabilities: VulnerabilityLibrary = dict([]), identifiers: Identifiers = SAMPLE_IDENTIFIERS
) -> nx.DiGraph:
    graph = nx.relabel_nodes(graph, {i: str(i) for i in graph.nodes})

    def create_random_firewall_configuration() -> FirewallConfiguration:
        return FirewallConfiguration(
            outgoing=[
                FirewallRule(port=p, permission=RulePermission.ALLOW)
                for p in random.sample(identifiers.ports, k=random.randint(0, len(identifiers.ports)))
            ],
            incoming=[
                FirewallRule(port=p, permission=RulePermission.ALLOW)
                for p in random.sample(identifiers.ports, k=random.randint(0, len(identifiers.ports)))
            ],
        )

    def create_random_properties() -> List[PropertyName]:
        return list(random.sample(identifiers.properties, k=random.randint(0, len(identifiers.properties))))

    def pick_random_global_vulnerabilities() -> VulnerabilityLibrary:
        count = random.random()
        return {k: v for (k, v) in vulnerabilities.items() if random.random() > count}

    def add_leak_neighbors_vulnerability(library: VulnerabilityLibrary, node_id: NodeID) -> None:
        neighbors = {t for (s, t) in graph.edges() if s == node_id}
        if len(neighbors) > 0:
            library["RecentlyAccessedMachines"] = VulnerabilityInfo(
                description="AzureVM info, including public IP address",
                type=VulnerabilityType.LOCAL,
                outcome=LeakedNodesId(list(neighbors)),
            )

    def create_random_vulnerabilities(node_id: NodeID) -> VulnerabilityLibrary:
        library = pick_random_global_vulnerabilities()
        add_leak_neighbors_vulnerability(library, node_id)
        return library

    entry_node_index = random.randrange(len(graph.nodes))
    entry_node_id, entry_node_data = list(graph.nodes(data=True))[entry_node_index]
    graph.nodes[entry_node_id].clear()
    node_data = NodeInfo(
        services=[],
        value=0,
        properties=create_random_properties(),
        vulnerabilities=create_random_vulnerabilities(entry_node_id),
        firewall=create_random_firewall_configuration(),
        agent_installed=True,
        reimagable=False,
        privilege_level=PrivilegeLevel.Admin,
    )
    graph.nodes[entry_node_id].update({"data": node_data})

    def create_random_node_data(node_id: NodeID) -> NodeInfo:
        return NodeInfo(
            services=[],
            value=random.randint(0, 100),
            properties=create_random_properties(),
            vulnerabilities=create_random_vulnerabilities(node_id),
            firewall=create_random_firewall_configuration(),
            agent_installed=False,
            privilege_level=PrivilegeLevel.NoAccess,
        )

    for node in list(graph.nodes):
        if node != entry_node_id:
            graph.nodes[node].clear()
            graph.nodes[node].update({"data": create_random_node_data(node)})

    return graph


def setup_yaml_serializer() -> None:
    if yaml is None:  # pragma: no cover
        raise ImportError("PyYAML is required for YAML serialization helpers.")

    yaml.add_representer(  # type: ignore
        Precondition,
        lambda dumper, data: dumper.represent_scalar("!BooleanExpression", str(data.expression)),
    )
    yaml.SafeLoader.add_constructor(  # type: ignore
        "!BooleanExpression", lambda loader, expression: Precondition(loader.construct_scalar(expression))
    )
    yaml.add_constructor(  # type: ignore
        "!BooleanExpression", lambda loader, expression: Precondition(loader.construct_scalar(expression))
    )

    yaml.add_representer(  # type: ignore
        VulnerabilityType, lambda dumper, data: dumper.represent_scalar("!VulnerabilityType", str(data.name))
    )

    yaml.SafeLoader.add_constructor(  # type: ignore
        "!VulnerabilityType", lambda loader, expression: VulnerabilityType[loader.construct_scalar(expression)]
    )
    yaml.add_constructor(  # type: ignore
        "!VulnerabilityType", lambda loader, expression: VulnerabilityType[loader.construct_scalar(expression)]
    )

