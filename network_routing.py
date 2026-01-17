"""
Network Routing Optimization using Uniform-Cost Search
Industry-level implementation for OSPF-like protocols

This module demonstrates UCS application in computer network routing,
simulating real-world scenarios like OSPF (Open Shortest Path First) protocol.
"""

import random
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from uniform_cost_search import Graph, UniformCostSearch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class NetworkNode:
    """Represents a network router or switch."""
    id: str
    ip_address: str
    location: Tuple[float, float]  # (lat, lon) or (x, y)
    bandwidth: int  # Mbps
    processing_delay: float  # milliseconds


@dataclass
class NetworkLink:
    """Represents a network connection between nodes."""
    source: str
    destination: str
    bandwidth: int  # Mbps
    latency: float  # milliseconds
    reliability: float  # 0.0 to 1.0
    cost_per_mb: float  # monetary cost


class NetworkTopology:
    """
    Manages network topology and converts to graph for UCS routing.
    Simulates real-world network scenarios with dynamic costs.
    """
    
    def __init__(self):
        self.nodes: Dict[str, NetworkNode] = {}
        self.links: List[NetworkLink] = []
        self.graph = Graph(directed=True)
        self.traffic_loads: Dict[Tuple[str, str], float] = {}
    
    def add_node(self, node: NetworkNode) -> None:
        """Add a network node to the topology."""
        self.nodes[node.id] = node
    
    def add_link(self, link: NetworkLink) -> None:
        """Add a network link to the topology."""
        self.links.append(link)
        # Calculate dynamic cost based on multiple factors
        cost = self._calculate_link_cost(link)
        self.graph.add_edge(link.source, link.destination, cost)
        
        # Add reverse link for bidirectional communication
        reverse_cost = self._calculate_link_cost(NetworkLink(
            link.destination, link.source, link.bandwidth, 
            link.latency, link.reliability, link.cost_per_mb
        ))
        self.graph.add_edge(link.destination, link.source, reverse_cost)
    
    def _calculate_link_cost(self, link: NetworkLink) -> float:
        """
        Calculate comprehensive link cost for routing decisions.
        Combines latency, bandwidth, reliability, and monetary cost.
        """
        # Base cost from latency (primary factor)
        latency_cost = link.latency
        
        # Bandwidth penalty (lower bandwidth = higher cost)
        bandwidth_penalty = 1000 / max(link.bandwidth, 1)
        
        # Reliability penalty (lower reliability = higher cost)
        reliability_penalty = (1 - link.reliability) * 100
        
        # Monetary cost
        monetary_cost = link.cost_per_mb * 10
        
        # Current traffic load penalty
        traffic_penalty = 0
        if (link.source, link.destination) in self.traffic_loads:
            load = self.traffic_loads[(link.source, link.destination)]
            utilization = load / link.bandwidth
            traffic_penalty = utilization * 50  # Penalty for high utilization
        
        total_cost = latency_cost + bandwidth_penalty + reliability_penalty + monetary_cost + traffic_penalty
        return total_cost
    
    def update_traffic_load(self, source: str, destination: str, load: float) -> None:
        """Update traffic load on a specific link."""
        self.traffic_loads[(source, destination)] = load
        # Recalculate costs for affected links
        self._recalculate_costs()
    
    def _recalculate_costs(self) -> None:
        """Recalculate all link costs based on current conditions."""
        self.graph = Graph(directed=True)
        for link in self.links:
            cost = self._calculate_link_cost(link)
            self.graph.add_edge(link.source, link.destination, cost)
            
            reverse_link = NetworkLink(
                link.destination, link.source, link.bandwidth,
                link.latency, link.reliability, link.cost_per_mb
            )
            reverse_cost = self._calculate_link_cost(reverse_link)
            self.graph.add_edge(link.destination, link.source, reverse_cost)
    
    def get_optimal_route(self, source: str, destination: str) -> Tuple[Optional[List[str]], Optional[float]]:
        """Find optimal route using UCS."""
        ucs = UniformCostSearch(self.graph)
        return ucs.search(source, destination)
    
    def simulate_network_failure(self, failed_node: str) -> None:
        """Simulate network node failure."""
        # Remove node and all connected links
        if failed_node in self.nodes:
            del self.nodes[failed_node]
            self.links = [link for link in self.links 
                         if link.source != failed_node and link.destination != failed_node]
            self._recalculate_costs()


class NetworkRoutingSimulator:
    """
    Advanced network routing simulator with UCS optimization.
    Simulates real-world scenarios including dynamic traffic, failures, and load balancing.
    """
    
    def __init__(self, topology: NetworkTopology):
        self.topology = topology
        self.routing_table: Dict[str, Dict[str, List[str]]] = {}
        self.performance_metrics = {
            'total_packets_routed': 0,
            'average_latency': 0,
            'packet_loss_rate': 0,
            'network_utilization': 0
        }
    
    def build_routing_tables(self) -> None:
        """Build routing tables for all nodes using UCS."""
        nodes = list(self.topology.nodes.keys())
        
        for source in nodes:
            self.routing_table[source] = {}
            ucs = UniformCostSearch(self.topology.graph)
            
            for destination in nodes:
                if source != destination:
                    try:
                        path, cost = ucs.search(source, destination)
                        if path:
                            self.routing_table[source][destination] = path
                    except:
                        # Handle unreachable destinations
                        pass
    
    def route_packet(self, source: str, destination: str, packet_size: float = 1.0) -> Dict[str, any]:
        """
        Route a packet from source to destination.
        Returns routing information and performance metrics.
        """
        if source not in self.routing_table or destination not in self.routing_table[source]:
            return {'success': False, 'error': 'No route available'}
        
        path = self.routing_table[source][destination]
        
        # Calculate total latency and cost
        total_latency = 0
        total_cost = 0
        
        for i in range(len(path) - 1):
            for link in self.topology.links:
                if link.source == path[i] and link.destination == path[i+1]:
                    total_latency += link.latency
                    total_cost += link.cost_per_mb * packet_size
                    break
        
        # Update traffic loads
        for i in range(len(path) - 1):
            current_load = self.topology.traffic_loads.get((path[i], path[i+1]), 0)
            self.topology.update_traffic_load(path[i], path[i+1], current_load + packet_size)
        
        # Update performance metrics
        self.performance_metrics['total_packets_routed'] += 1
        self.performance_metrics['average_latency'] = (
            (self.performance_metrics['average_latency'] * (self.performance_metrics['total_packets_routed'] - 1) + total_latency) /
            self.performance_metrics['total_packets_routed']
        )
        
        return {
            'success': True,
            'path': path,
            'latency': total_latency,
            'cost': total_cost,
            'hops': len(path) - 1
        }
    
    def simulate_traffic_pattern(self, duration: int, packet_rate: float = 10.0) -> Dict[str, any]:
        """
        Simulate network traffic over time.
        
        Args:
            duration: Simulation duration in seconds
            packet_rate: Packets per second
        
        Returns:
            Simulation results and performance metrics
        """
        nodes = list(self.topology.nodes.keys())
        packets_sent = 0
        packets_delivered = 0
        total_latency = 0
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Generate random packet
            source = random.choice(nodes)
            destination = random.choice([n for n in nodes if n != destination])
            packet_size = random.uniform(0.1, 10.0)  # MB
            
            # Route packet
            result = self.route_packet(source, destination, packet_size)
            
            if result['success']:
                packets_delivered += 1
                total_latency += result['latency']
            
            packets_sent += 1
            time.sleep(1.0 / packet_rate)
        
        return {
            'packets_sent': packets_sent,
            'packets_delivered': packets_delivered,
            'delivery_rate': packets_delivered / packets_sent,
            'average_latency': total_latency / max(packets_delivered, 1),
            'performance_metrics': self.performance_metrics
        }
    
    def visualize_network(self, save_path: str = None) -> None:
        """Visualize the network topology using NetworkX and Matplotlib."""
        G = nx.DiGraph()
        
        # Add nodes
        for node_id, node in self.topology.nodes.items():
            G.add_node(node_id, pos=node.location)
        
        # Add edges with weights
        for link in self.topology.links:
            cost = self.topology._calculate_link_cost(link)
            G.add_edge(link.source, link.destination, weight=cost)
        
        # Draw the network
        pos = nx.get_node_attributes(G, 'pos')
        plt.figure(figsize=(12, 8))
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
        
        # Draw edges with width based on cost
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        edge_widths = [3 * (1 - w/max_weight) + 0.5 for w in weights]
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', 
                              arrows=True, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        # Draw edge labels (costs)
        edge_labels = {(u, v): f'{G[u][v]["weight"]:.1f}' for u, v in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
        
        plt.title("Network Topology with UCS Routing Costs")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_sample_network() -> NetworkTopology:
    """Create a sample network topology for demonstration."""
    topology = NetworkTopology()
    
    # Create nodes (routers/switches)
    nodes = [
        NetworkNode("R1", "192.168.1.1", (0, 0), 1000, 0.5),
        NetworkNode("R2", "192.168.1.2", (2, 1), 800, 0.6),
        NetworkNode("R3", "192.168.1.3", (4, 0), 1200, 0.4),
        NetworkNode("R4", "192.168.1.4", (1, 3), 600, 0.8),
        NetworkNode("R5", "192.168.1.5", (3, 3), 900, 0.5),
        NetworkNode("R6", "192.168.1.6", (5, 2), 1100, 0.4),
    ]
    
    for node in nodes:
        topology.add_node(node)
    
    # Create links with realistic parameters
    links = [
        NetworkLink("R1", "R2", 100, 5, 0.95, 0.01),
        NetworkLink("R1", "R4", 80, 8, 0.90, 0.02),
        NetworkLink("R2", "R3", 120, 3, 0.98, 0.01),
        NetworkLink("R2", "R5", 90, 6, 0.92, 0.015),
        NetworkLink("R3", "R6", 150, 4, 0.96, 0.008),
        NetworkLink("R4", "R5", 70, 7, 0.88, 0.025),
        NetworkLink("R5", "R6", 100, 5, 0.94, 0.012),
        NetworkLink("R3", "R5", 110, 4, 0.93, 0.011),
    ]
    
    for link in links:
        topology.add_link(link)
    
    return topology


def main():
    """Main demonstration of network routing with UCS."""
    print("Network Routing Optimization using Uniform-Cost Search")
    print("=" * 60)
    
    # Create network topology
    topology = create_sample_network()
    print(f"Created network with {len(topology.nodes)} nodes and {len(topology.links)} links")
    
    # Initialize simulator
    simulator = NetworkRoutingSimulator(topology)
    
    # Build routing tables
    print("\nBuilding routing tables using UCS...")
    simulator.build_routing_tables()
    print("Routing tables built successfully!")
    
    # Test routing between specific nodes
    print("\nTesting routing between nodes:")
    test_pairs = [("R1", "R6"), ("R4", "R3"), ("R2", "R5")]
    
    for source, dest in test_pairs:
        result = simulator.route_packet(source, dest, 5.0)
        if result['success']:
            print(f"Route {source} -> {dest}: {' -> '.join(result['path'])}")
            print(f"  Latency: {result['latency']:.2f}ms, Cost: ${result['cost']:.4f}, Hops: {result['hops']}")
        else:
            print(f"No route found from {source} to {dest}")
    
    # Simulate traffic pattern
    print("\nSimulating network traffic (10 seconds)...")
    simulation_results = simulator.simulate_traffic_pattern(duration=10, packet_rate=5.0)
    
    print(f"Packets sent: {simulation_results['packets_sent']}")
    print(f"Packets delivered: {simulation_results['packets_delivered']}")
    print(f"Delivery rate: {simulation_results['delivery_rate']:.2%}")
    print(f"Average latency: {simulation_results['average_latency']:.2f}ms")
    
    # Visualize network
    print("\nGenerating network visualization...")
    simulator.visualize_network()
    
    # Test network failure recovery
    print("\nTesting network failure recovery...")
    print("Simulating failure of node R3...")
    topology.simulate_network_failure("R3")
    
    # Rebuild routing tables
    simulator.build_routing_tables()
    
    # Test routing after failure
    result = simulator.route_packet("R1", "R6", 5.0)
    if result['success']:
        print(f"New route R1 -> R6: {' -> '.join(result['path'])}")
        print(f"  Latency: {result['latency']:.2f}ms, Cost: ${result['cost']:.4f}")
    else:
        print("No route available after R3 failure")


if __name__ == "__main__":
    main()
