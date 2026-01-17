"""
Uniform-Cost Search Lab Manual
Professional and Industry-Level Implementation

This comprehensive lab manual provides hands-on exercises and tutorials
for mastering Uniform-Cost Search algorithm in real-world applications.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uniform_cost_search import Graph, UniformCostSearch, UCSComparator
from network_routing import NetworkTopology, NetworkRoutingSimulator
from logistics_optimization import LogisticsNetwork, LogisticsOptimizer
from performance_benchmarks import PerformanceBenchmark, GraphGenerator
from visualization_tools import UCSVisualizer, InteractiveGraphExplorer
import matplotlib.pyplot as plt


class LabManual:
    """Comprehensive lab manual with progressive exercises."""
    
    def __init__(self):
        self.current_exercise = 0
        self.exercises = [
            self.exercise_1_basic_ucs,
            self.exercise_2_graph_representations,
            self.exercise_3_network_routing,
            self.exercise_4_logistics_optimization,
            self.exercise_5_performance_analysis,
            self.exercise_6_advanced_topics
        ]
    
    def run_lab(self):
        """Run the complete lab manual."""
        print("=" * 70)
        print("UNIFORM-COST SEARCH PROFESSIONAL LAB MANUAL")
        print("=" * 70)
        print("\nThis lab covers UCS from fundamentals to industry applications.")
        print("Each exercise builds upon previous concepts.\n")
        
        for i, exercise in enumerate(self.exercises, 1):
            print(f"\n{'='*20} EXERCISE {i} {'='*20}")
            exercise()
            
            if i < len(self.exercises):
                input("\nPress Enter to continue to next exercise...")
        
        print("\n" + "="*70)
        print("LAB COMPLETED! You've mastered Uniform-Cost Search!")
        print("="*70)
    
    def exercise_1_basic_ucs(self):
        """Exercise 1: Basic UCS Implementation and Understanding"""
        print("\nEXERCISE 1: Basic Uniform-Cost Search")
        print("-" * 40)
        print("Objectives:")
        print("- Understand UCS algorithm fundamentals")
        print("- Implement basic graph creation")
        print("- Run UCS on simple graphs")
        print("- Analyze path optimality")
        
        # Create simple graph
        graph = Graph(directed=False)
        edges = [
            ('A', 'B', 2), ('A', 'C', 5), ('B', 'D', 1),
            ('C', 'D', 2), ('B', 'E', 4), ('D', 'E', 1)
        ]
        
        for from_node, to_node, cost in edges:
            graph.add_edge(from_node, to_node, cost)
        
        print("\nGraph created with edges:")
        for from_node, to_node, cost in edges:
            print(f"  {from_node} --({cost})-- {to_node}")
        
        # Run UCS
        ucs = UniformCostSearch(graph)
        path, cost = ucs.search('A', 'E')
        
        print(f"\nUCS Path from A to E: {' -> '.join(path)}")
        print(f"Total Cost: {cost}")
        print(f"Performance Metrics: {ucs.get_performance_metrics()}")
        
        # Verification
        print("\nVerification:")
        print("Expected path: A -> B -> D -> E (cost: 2 + 1 + 1 = 4)")
        print(f"Actual path cost: {cost}")
        print(f"Optimal: {'✓' if cost == 4 else '✗'}")
    
    def exercise_2_graph_representations(self):
        """Exercise 2: Different Graph Representations"""
        print("\nEXERCISE 2: Graph Representations and UCS Behavior")
        print("-" * 50)
        print("Objectives:")
        print("- Test UCS on different graph types")
        print("- Understand graph structure impact")
        print("- Compare with other algorithms")
        
        # Test different graph types
        graph_types = [
            ("Linear Chain", self._create_linear_graph),
            ("Star Graph", self._create_star_graph),
            ("Complete Graph", self._create_complete_graph),
            ("Tree Structure", self._create_tree_graph)
        ]
        
        for name, creator in graph_types:
            print(f"\n{name}:")
            graph = creator()
            ucs = UniformCostSearch(graph)
            
            # Find path from first to last node
            nodes = list(graph.get_nodes())
            if len(nodes) >= 2:
                start, goal = nodes[0], nodes[-1]
                path, cost = ucs.search(start, goal)
                
                print(f"  Path {start} -> {goal}: {path}")
                print(f"  Cost: {cost}")
                print(f"  Nodes explored: {ucs.get_performance_metrics()['nodes_explored']}")
    
    def exercise_3_network_routing(self):
        """Exercise 3: Network Routing Application"""
        print("\nEXERCISE 3: Network Routing with UCS")
        print("-" * 40)
        print("Objectives:")
        print("- Apply UCS to network routing")
        print("- Understand dynamic cost calculation")
        print("- Simulate network scenarios")
        
        # Create network topology
        from network_routing import create_sample_network
        topology = create_sample_network()
        
        print(f"Network with {len(topology.nodes)} nodes and {len(topology.links)} links")
        
        # Find optimal routes
        test_routes = [("R1", "R6"), ("R4", "R3"), ("R2", "R5")]
        
        for source, dest in test_routes:
            path, cost = topology.get_optimal_route(source, dest)
            if path:
                print(f"\nRoute {source} -> {dest}:")
                print(f"  Path: {' -> '.join(path)}")
                print(f"  Cost: {cost:.2f}")
            else:
                print(f"\nNo route found from {source} to {dest}")
    
    def exercise_4_logistics_optimization(self):
        """Exercise 4: Logistics and Supply Chain"""
        print("\nEXERCISE 4: Logistics Optimization")
        print("-" * 35)
        print("Objectives:")
        print("- Apply UCS to delivery routing")
        print("- Consider time windows and constraints")
        print("- Optimize fleet operations")
        
        # Create logistics network
        from logistics_optimization import create_sample_logistics_network
        network = create_sample_logistics_network()
        
        print(f"Logistics network:")
        print(f"  Locations: {len(network.locations)}")
        print(f"  Vehicles: {len(network.vehicles)}")
        print(f"  Orders: {len(network.orders)}")
        
        # Optimize single vehicle
        optimizer = LogisticsOptimizer(network)
        from datetime import datetime
        today = datetime.now()
        
        result = optimizer.optimize_single_vehicle_routes("V1", today)
        print(f"\nVehicle V1 optimized route:")
        print(f"  Route: {' -> '.join(result['route'])}")
        print(f"  Cost: ${result['total_cost']:.2f}")
        print(f"  Distance: {result['total_distance']:.1f} km")
    
    def exercise_5_performance_analysis(self):
        """Exercise 5: Performance Analysis and Benchmarking"""
        print("\nEXERCISE 5: Performance Analysis")
        print("-" * 35)
        print("Objectives:")
        print("- Benchmark UCS against other algorithms")
        print("- Analyze scalability")
        print("- Understand performance characteristics")
        
        # Create benchmark
        benchmark = PerformanceBenchmark()
        
        # Test on different graph sizes
        sizes = [10, 25, 50]
        results = {}
        
        for size in sizes:
            print(f"\nTesting graph size: {size}")
            graph = GraphGenerator.generate_random_graph(size, edge_probability=0.3)
            nodes = list(graph.get_nodes())
            start, goal = nodes[0], nodes[-1]
            
            # Compare algorithms
            comparator = UCSComparator(graph)
            comparison_results = comparator.compare_algorithms(start, goal)
            
            print(f"  Results for size {size}:")
            for algo, result in comparison_results.items():
                if result['success']:
                    print(f"    {algo}: Cost={result['cost']:.2f}, "
                          f"Time={result['metrics'].get('search_time', 0):.4f}s")
    
    def exercise_6_advanced_topics(self):
        """Exercise 6: Advanced Topics and Applications"""
        print("\nEXERCISE 6: Advanced UCS Topics")
        print("-" * 35)
        print("Objectives:")
        print("- Explore UCS variations")
        print("- Understand heuristic integration")
        print("- Apply to complex scenarios")
        
        # Demonstrate UCS with heuristics (A* comparison)
        graph = Graph(directed=False)
        edges = [
            ((0,0), (1,0), 1), ((1,0), (2,0), 1), ((2,0), (3,0), 1),
            ((0,0), (0,1), 1), ((0,1), (0,2), 1), ((0,2), (0,3), 1),
            ((1,0), (1,1), 2), ((1,1), (1,2), 2), ((1,2), (1,3), 2),
            ((2,0), (2,1), 3), ((2,1), (2,2), 3), ((2,2), (2,3), 3),
            ((0,1), (1,1), 1), ((1,1), (2,1), 1), ((2,1), (3,1), 1),
            ((0,2), (1,2), 2), ((1,2), (2,2), 2), ((2,2), (3,2), 2),
            ((0,3), (1,3), 3), ((1,3), (2,3), 3), ((2,3), (3,3), 3)
        ]
        
        for from_node, to_node, cost in edges:
            graph.add_edge(from_node, to_node, cost)
        
        # Define heuristic function
        def manhattan_distance(node1, node2):
            return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])
        
        # Compare UCS vs A*
        start, goal = (0, 0), (3, 3)
        
        # Pure UCS
        ucs = UniformCostSearch(graph)
        ucs_path, ucs_cost = ucs.search(start, goal)
        
        # A* (UCS with heuristic)
        astar = UniformCostSearch(graph, heuristic=manhattan_distance)
        astar_path, astar_cost = astar.search(start, goal)
        
        print(f"Grid pathfinding from {start} to {goal}:")
        print(f"UCS:  Path={ucs_path}, Cost={ucs_cost}, "
              f"Nodes explored={ucs.get_performance_metrics()['nodes_explored']}")
        print(f"A*:   Path={astar_path}, Cost={astar_cost}, "
              f"Nodes explored={astar.get_performance_metrics()['nodes_explored']}")
        
        print(f"\nKey insights:")
        print(f"- Both algorithms find optimal paths")
        print(f"- A* explores fewer nodes with good heuristic")
        print(f"- UCS is guaranteed optimal without heuristic knowledge")
    
    def _create_linear_graph(self):
        """Create a linear chain graph."""
        graph = Graph(directed=False)
        for i in range(5):
            graph.add_edge(f"N{i}", f"N{i+1}", i+1)
        return graph
    
    def _create_star_graph(self):
        """Create a star graph."""
        graph = Graph(directed=False)
        center = "Center"
        for i in range(4):
            graph.add_edge(center, f"Leaf{i}", i+2)
        return graph
    
    def _create_complete_graph(self):
        """Create a complete graph."""
        return GraphGenerator.generate_complete_graph(4)
    
    def _create_tree_graph(self):
        """Create a tree graph."""
        return GraphGenerator.generate_tree_graph(7)


def main():
    """Run the complete lab manual."""
    lab = LabManual()
    lab.run_lab()


if __name__ == "__main__":
    main()
