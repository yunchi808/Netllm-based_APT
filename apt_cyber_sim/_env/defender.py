# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Defines stock defender agents for the CyberBattle simulation."""

import logging
import random
from abc import abstractmethod

import numpy

from apt_cyber_sim.simulation.model import Environment
from apt_cyber_sim.simulation.actions import DefenderAgentActions
from ..simulation import model


class DefenderAgent:
    """Define the step function for a defender agent.
    Gets called after each step executed by the attacker agent."""

    @abstractmethod
    def step(self, environment: Environment, actions: DefenderAgentActions, t: int):
        None


class ScanAndReimageCompromisedMachines(DefenderAgent):
    """A defender agent that scans a subset of network nodes and re-images nodes."""

    def __init__(self, probability: float, scan_capacity: int, scan_frequency: int):
        self.probability = probability
        self.scan_capacity = scan_capacity
        self.scan_frequency = scan_frequency

    def step(self, environment: Environment, actions: DefenderAgentActions, t: int):
        if t % self.scan_frequency == 0:
            scanned_nodes = random.choices(list(environment.network.nodes), k=self.scan_capacity)
            for node_id in scanned_nodes:
                node_info = environment.get_node(node_id)
                if node_info.status == model.MachineStatus.Running and node_info.agent_installed:
                    is_malware_detected = numpy.random.random() <= self.probability
                    if is_malware_detected:
                        if node_info.reimagable:
                            logging.info(f"Defender detected malware, reimaging node {node_id}")
                            actions.reimage_node(node_id)
                        else:
                            logging.info(f"Defender detected malware, but node cannot be reimaged {node_id}")


class ExternalRandomEvents(DefenderAgent):
    """A 'defender' that randomly alters network node configuration."""

    def step(self, environment: Environment, actions: DefenderAgentActions, t: int):
        self.patch_vulnerabilities_at_random(environment)
        self.stop_service_at_random(environment, actions)
        self.plant_vulnerabilities_at_random(environment)
        self.firewall_change_remove(environment)
        self.firewall_change_add(environment)

    def patch_vulnerabilities_at_random(self, environment: Environment, probability: float = 0.1) -> None:
        for _, node_data in environment.nodes():
            remove_vulnerability = numpy.random.random() <= probability
            if remove_vulnerability and len(node_data.vulnerabilities) > 0:
                choice = random.choice(list(node_data.vulnerabilities))
                node_data.vulnerabilities.pop(choice)

    def stop_service_at_random(
        self, environment: Environment, actions: DefenderAgentActions, probability: float = 0.1
    ) -> None:
        for node_id, node_data in environment.nodes():
            remove_service = numpy.random.random() <= probability
            if remove_service and len(node_data.services) > 0:
                service = random.choice(node_data.services)
                actions.stop_service(node_id, service.name)

    def plant_vulnerabilities_at_random(self, environment: Environment, probability: float = 0.1) -> None:
        for _, node_data in environment.nodes():
            add_vulnerability = numpy.random.random() <= probability
            if add_vulnerability and len(environment.vulnerability_library) > 0:
                choice = random.choice(list(environment.vulnerability_library))
                node_data.vulnerabilities[choice] = environment.vulnerability_library[choice]

    def firewall_change_remove(self, environment: Environment, probability: float = 0.1) -> None:
        for _, node_data in environment.nodes():
            firewall_change = numpy.random.random() <= probability
            if firewall_change and len(node_data.firewall.incoming) > 0:
                node_data.firewall.incoming.pop()

    def firewall_change_add(self, environment: Environment, probability: float = 0.1) -> None:
        for _, node_data in environment.nodes():
            firewall_change = numpy.random.random() <= probability
            if firewall_change and len(node_data.firewall.incoming) > 0:
                node_data.firewall.incoming.append(node_data.firewall.incoming[-1])

