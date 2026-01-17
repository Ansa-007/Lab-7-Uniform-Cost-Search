"""
Uniform-Cost Search (UCS) Implementation
Professional and Industry-Level Implementation

This module provides a comprehensive implementation of UCS with:
- Priority queue-based optimization
- Multiple graph representations
- Path reconstruction
- Performance monitoring
- Industry-standard error handling
"""

import heapq
from typing import Dict, List, Tuple, Optional, Set, Any, Callable
from collections import defaultdict
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Graph:
    """
    Graph data structure supporting multiple representations for UCS.
    Optimized for industry applications with large-scale graphs.
    """
    
    def __init__(self, directed: bool = False):
        self.directed = directed
        self.adjacency_list: Dict[Any, List[Tuple[Any, float]]] = defaultdict(list)
        self.nodes: Set[Any] = set()
        self.edge_count = 0
    
    def add_edge(self, from_node: Any, to_node: Any, cost: float) -> None:
        """
        Add an edge to the graph with validation.
        
        Args:
            from_node: Source node
            to_node: Destination node
            cost: Edge cost (must be non-negative)
        
        Raises:
            ValueError: If cost is negative
        """
        if cost < 0:
            raise ValueError("Edge costs must be non-negative for UCS")
        
        self.adjacency_list[from_node].append((to_node, cost))
        self.nodes.add(from_node)
        self.nodes.add(to_node)
        self.edge_count += 1
        
        if not self.directed:
            self.adjacency_list[to_node].append((from_node, cost))
    
    def get_neighbors(self, node: Any) -> List[Tuple[Any, float]]:
        """Get neighbors and costs for a given node."""
        return self.adjacency_list.get(node, [])
    
    def get_nodes(self) -> Set[Any]:
        """Get all nodes in the graph."""
        return self.nodes.copy()
    
    def get_edge_count(self) -> int:
        """Get total number of edges."""
        return self.edge_count
    
    def validate_node(self, node: Any) -> bool:
        """Check if node exists in graph."""
        return node in self.nodes


class UniformCostSearch:
    """
    Professional UCS implementation with priority queue optimization.
    Includes performance monitoring and comprehensive error handling.
    """
    
    def __init__(self, graph: Graph, heuristic: Optional[Callable] = None):
        """
        Initialize UCS with graph and optional heuristic.
        
        Args:
            graph: Graph object to search
            heuristic: Optional heuristic function (for A* comparison)
        """
        self.graph = graph
        self.heuristic = heuristic
        self.nodes_explored = 0
        self.nodes_generated = 0
        self.search_time = 0
        self.max_queue_size = 0
    
    def search(self, start: Any, goal: Any, 
               max_iterations: int = 1000000) -> Tuple[Optional[List[Any]], Optional[float]]:
        """
        Perform UCS search from start to goal.
        
        Args:
            start: Starting node
            goal: Goal node
            max_iterations: Maximum iterations to prevent infinite loops
        
        Returns:
            Tuple of (path, total_cost) or (None, None) if no path found
        
        Raises:
            ValueError: If start or goal nodes don't exist
        """
        # Validate inputs
        if not self.graph.validate_node(start):
            raise ValueError(f"Start node {start} not found in graph")
        if not self.graph.validate_node(goal):
            raise ValueError(f"Goal node {goal} not found in graph")
        
        # Initialize search
        start_time = time.time()
        
        # Priority queue: (cost, node, path)
        frontier = [(0, start, [start])]
        # Cost to reach each node
        cost_so_far: Dict[Any, float] = {start: 0}
        # Explored set
        explored: Set[Any] = set()
        
        logger.info(f"Starting UCS search from {start} to {goal}")
        
        try:
            while frontier and self.nodes_explored < max_iterations:
                # Update performance metrics
                self.max_queue_size = max(self.max_queue_size, len(frontier))
                
                # Get node with lowest cost
                current_cost, current_node, current_path = heapq.heappop(frontier)
                
                # Goal test
                if current_node == goal:
                    self.search_time = time.time() - start_time
                    logger.info(f"Goal found! Path cost: {current_cost}, Nodes explored: {self.nodes_explored}")
                    return current_path, current_cost
                
                # Skip if already explored with lower cost
                if current_node in explored:
                    continue
                
                explored.add(current_node)
                self.nodes_explored += 1
                
                # Expand neighbors
                for neighbor, edge_cost in self.graph.get_neighbors(current_node):
                    new_cost = current_cost + edge_cost
                    
                    # If we haven't seen this node or found a cheaper path
                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        new_path = current_path + [neighbor]
                        
                        # Add heuristic if available (for A* comparison)
                        priority = new_cost
                        if self.heuristic:
                            priority += self.heuristic(neighbor, goal)
                        
                        heapq.heappush(frontier, (priority, neighbor, new_path))
                        self.nodes_generated += 1
            
            # No path found
            self.search_time = time.time() - start_time
            logger.warning(f"No path found from {start} to {goal}")
            return None, None
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            'nodes_explored': self.nodes_explored,
            'nodes_generated': self.nodes_generated,
            'search_time': self.search_time,
            'max_queue_size': self.max_queue_size,
            'nodes_per_second': self.nodes_explored / max(self.search_time, 0.001)
        }
    
    def reset_metrics(self) -> None:
        """Reset performance metrics for new search."""
        self.nodes_explored = 0
        self.nodes_generated = 0
        self.search_time = 0
        self.max_queue_size = 0


class UCSComparator:
    """
    Compare UCS with other search algorithms (BFS, DFS, A*).
    Professional benchmarking tool for algorithm selection.
    """
    
    def __init__(self, graph: Graph):
        self.graph = graph
        self.results = {}
    
    def compare_algorithms(self, start: Any, goal: Any) -> Dict[str, Dict[str, Any]]:
        """
        Compare UCS with BFS, DFS, and A* algorithms.
        
        Returns:
            Dictionary with performance metrics for each algorithm
        """
        algorithms = {
            'UCS': UniformCostSearch(self.graph),
            'A*': UniformCostSearch(self.graph, heuristic=self._manhattan_distance),
        }
        
        results = {}
        
        for name, algorithm in algorithms.items():
            try:
                algorithm.reset_metrics()
                path, cost = algorithm.search(start, goal)
                
                results[name] = {
                    'path': path,
                    'cost': cost,
                    'path_length': len(path) if path else 0,
                    'metrics': algorithm.get_performance_metrics(),
                    'success': path is not None
                }
                
            except Exception as e:
                results[name] = {
                    'path': None,
                    'cost': None,
                    'path_length': 0,
                    'metrics': {},
                    'success': False,
                    'error': str(e)
                }
        
        self.results = results
        return results
    
    def _manhattan_distance(self, node1: Any, node2: Any) -> float:
        """
        Simple heuristic for A* comparison.
        Assumes nodes are tuples (x, y) for grid-based problems.
        """
        if isinstance(node1, tuple) and isinstance(node2, tuple):
            return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])
        return 0
    
    def generate_comparison_report(self) -> str:
        """Generate a detailed comparison report."""
        if not self.results:
            return "No comparison results available. Run compare_algorithms() first."
        
        report = ["Algorithm Comparison Report", "=" * 50]
        
        for name, result in self.results.items():
            report.append(f"\n{name}:")
            report.append(f"  Success: {result['success']}")
            
            if result['success']:
                report.append(f"  Path: {result['path']}")
                report.append(f"  Cost: {result['cost']:.2f}")
                report.append(f"  Path Length: {result['path_length']}")
                
                metrics = result['metrics']
                if metrics:
                    report.append(f"  Nodes Explored: {metrics.get('nodes_explored', 'N/A')}")
                    report.append(f"  Search Time: {metrics.get('search_time', 'N/A'):.4f}s")
                    report.append(f"  Nodes/Second: {metrics.get('nodes_per_second', 'N/A'):.0f}")
            else:
                report.append(f"  Error: {result.get('error', 'Unknown error')}")
        
        return "\n".join(report)


# Example usage and test cases
if __name__ == "__main__":
    # Create a sample graph for testing
    graph = Graph(directed=False)
    
    # Add edges with costs
    edges = [
        ('A', 'B', 4), ('A', 'C', 2), ('B', 'C', 1),
        ('B', 'D', 5), ('C', 'D', 8), ('C', 'E', 10),
        ('D', 'E', 2), ('D', 'F', 6), ('E', 'F', 3)
    ]
    
    for from_node, to_node, cost in edges:
        graph.add_edge(from_node, to_node, cost)
    
    # Test UCS
    ucs = UniformCostSearch(graph)
    path, cost = ucs.search('A', 'F')
    
    print(f"UCS Path from A to F: {path}")
    print(f"Total Cost: {cost}")
    print(f"Performance Metrics: {ucs.get_performance_metrics()}")
    
    # Compare algorithms
    comparator = UCSComparator(graph)
    results = comparator.compare_algorithms('A', 'F')
    print("\n" + comparator.generate_comparison_report())
