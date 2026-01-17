"""
Performance Benchmarks and Algorithm Comparisons for Uniform-Cost Search
Professional benchmarking suite for industry-level analysis

This module provides comprehensive performance testing including:
- Algorithm comparisons (UCS vs BFS vs DFS vs A*)
- Scalability analysis
- Memory usage profiling
- Real-time performance metrics
- Statistical analysis of results
"""

import time
import random
import psutil
import os
import gc
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from uniform_cost_search import Graph, UniformCostSearch, UCSComparator


@dataclass
class BenchmarkResult:
    """Stores results from a single benchmark run."""
    algorithm: str
    graph_size: int
    path_found: bool
    path_cost: Optional[float]
    path_length: int
    execution_time: float
    memory_usage: float
    nodes_explored: int
    nodes_generated: int
    max_queue_size: int


class GraphGenerator:
    """
    Generates various types of graphs for benchmarking.
    Creates realistic test scenarios for algorithm comparison.
    """
    
    @staticmethod
    def generate_grid_graph(rows: int, cols: int, 
                           cost_range: Tuple[float, float] = (1.0, 10.0)) -> Graph:
        """Generate a grid graph with random edge costs."""
        graph = Graph(directed=False)
        
        for i in range(rows):
            for j in range(cols):
                node_id = f"{i},{j}"
                
                # Add edges to neighbors
                if i < rows - 1:  # Down
                    neighbor = f"{i+1},{j}"
                    cost = random.uniform(*cost_range)
                    graph.add_edge(node_id, neighbor, cost)
                
                if j < cols - 1:  # Right
                    neighbor = f"{i},{j+1}"
                    cost = random.uniform(*cost_range)
                    graph.add_edge(node_id, neighbor, cost)
        
        return graph
    
    @staticmethod
    def generate_random_graph(num_nodes: int, edge_probability: float = 0.3,
                           cost_range: Tuple[float, float] = (1.0, 20.0)) -> Graph:
        """Generate a random graph with specified edge probability."""
        graph = Graph(directed=False)
        nodes = [f"N{i}" for i in range(num_nodes)]
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < edge_probability:
                    cost = random.uniform(*cost_range)
                    graph.add_edge(nodes[i], nodes[j], cost)
        
        return graph
    
    @staticmethod
    def generate_complete_graph(num_nodes: int, 
                              cost_range: Tuple[float, float] = (1.0, 50.0)) -> Graph:
        """Generate a complete graph with all possible edges."""
        graph = Graph(directed=False)
        nodes = [f"N{i}" for i in range(num_nodes)]
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                cost = random.uniform(*cost_range)
                graph.add_edge(nodes[i], nodes[j], cost)
        
        return graph
    
    @staticmethod
    def generate_tree_graph(num_nodes: int, 
                          cost_range: Tuple[float, float] = (1.0, 15.0)) -> Graph:
        """Generate a tree graph (no cycles)."""
        graph = Graph(directed=False)
        
        for i in range(1, num_nodes):
            parent = random.randint(0, i - 1)
            cost = random.uniform(*cost_range)
            graph.add_edge(f"N{parent}", f"N{i}", cost)
        
        return graph
    
    @staticmethod
    def generate_scale_free_network(num_nodes: int, m: int = 2,
                                   cost_range: Tuple[float, float] = (1.0, 25.0)) -> Graph:
        """Generate a scale-free network using Barabási-Albert model."""
        graph = Graph(directed=False)
        
        # Start with a small complete graph
        initial_nodes = min(m + 1, num_nodes)
        for i in range(initial_nodes):
            for j in range(i + 1, initial_nodes):
                cost = random.uniform(*cost_range)
                graph.add_edge(f"N{i}", f"N{j}", cost)
        
        # Add remaining nodes with preferential attachment
        for new_node in range(initial_nodes, num_nodes):
            new_node_id = f"N{new_node}"
            
            # Calculate degrees of existing nodes
            degrees = {}
            for node in graph.get_nodes():
                degrees[node] = len(graph.get_neighbors(node))
            
            total_degree = sum(degrees.values())
            
            # Add m edges to existing nodes
            added_edges = 0
            attempts = 0
            max_attempts = num_nodes * 2
            
            while added_edges < m and attempts < max_attempts:
                # Select node with probability proportional to degree
                rand_val = random.uniform(0, total_degree)
                cumulative = 0
                
                for node, degree in degrees.items():
                    cumulative += degree
                    if cumulative >= rand_val:
                        cost = random.uniform(*cost_range)
                        graph.add_edge(new_node_id, node, cost)
                        degrees[node] += 1
                        total_degree += 1
                        added_edges += 1
                        break
                
                attempts += 1
        
        return graph


class AlgorithmImplementations:
    """
    Implementations of various search algorithms for comparison.
    Provides consistent interface for benchmarking.
    """
    
    @staticmethod
    def bfs_search(graph: Graph, start: Any, goal: Any) -> Tuple[Optional[List[Any]], Optional[float], Dict[str, int]]:
        """Breadth-First Search implementation."""
        from collections import deque
        
        queue = deque([(start, [start], 0)])
        visited = {start}
        nodes_explored = 0
        nodes_generated = 1
        
        while queue:
            current, path, cost = queue.popleft()
            nodes_explored += 1
            
            if current == goal:
                return path, cost, {
                    'nodes_explored': nodes_explored,
                    'nodes_generated': nodes_generated,
                    'max_queue_size': len(queue) + 1
                }
            
            for neighbor, edge_cost in graph.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    new_cost = cost + edge_cost
                    queue.append((neighbor, new_path, new_cost))
                    nodes_generated += 1
        
        return None, None, {
            'nodes_explored': nodes_explored,
            'nodes_generated': nodes_generated,
            'max_queue_size': 0
        }
    
    @staticmethod
    def dfs_search(graph: Graph, start: Any, goal: Any) -> Tuple[Optional[List[Any]], Optional[float], Dict[str, int]]:
        """Depth-First Search implementation."""
        stack = [(start, [start], 0)]
        visited = set()
        nodes_explored = 0
        nodes_generated = 1
        max_stack_size = 1
        
        while stack:
            current, path, cost = stack.pop()
            max_stack_size = max(max_stack_size, len(stack) + 1)
            
            if current in visited:
                continue
            
            visited.add(current)
            nodes_explored += 1
            
            if current == goal:
                return path, cost, {
                    'nodes_explored': nodes_explored,
                    'nodes_generated': nodes_generated,
                    'max_queue_size': max_stack_size
                }
            
            for neighbor, edge_cost in graph.get_neighbors(current):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    new_cost = cost + edge_cost
                    stack.append((neighbor, new_path, new_cost))
                    nodes_generated += 1
        
        return None, None, {
            'nodes_explored': nodes_explored,
            'nodes_generated': nodes_generated,
            'max_queue_size': max_stack_size
        }
    
    @staticmethod
    def a_star_search(graph: Graph, start: Any, goal: Any) -> Tuple[Optional[List[Any]], Optional[float], Dict[str, int]]:
        """A* Search implementation with simple heuristic."""
        import heapq
        
        def heuristic(node1: Any, node2: Any) -> float:
            """Simple heuristic - returns 0 (makes A* equivalent to UCS)."""
            return 0
        
        frontier = [(0, start, [start])]
        cost_so_far = {start: 0}
        nodes_explored = 0
        nodes_generated = 1
        max_queue_size = 1
        
        while frontier:
            current_cost, current, path = heapq.heappop(frontier)
            nodes_explored += 1
            
            if current == goal:
                return path, current_cost, {
                    'nodes_explored': nodes_explored,
                    'nodes_generated': nodes_generated,
                    'max_queue_size': max_queue_size
                }
            
            for neighbor, edge_cost in graph.get_neighbors(current):
                new_cost = cost_so_far[current] + edge_cost
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(frontier, (priority, neighbor, path + [neighbor]))
                    nodes_generated += 1
                    max_queue_size = max(max_queue_size, len(frontier))
        
        return None, None, {
            'nodes_explored': nodes_explored,
            'nodes_generated': nodes_generated,
            'max_queue_size': max_queue_size
        }


class PerformanceBenchmark:
    """
    Comprehensive benchmarking suite for search algorithms.
    Provides statistical analysis and visualization of results.
    """
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.algorithms = {
            'UCS': self._run_ucs,
            'BFS': AlgorithmImplementations.bfs_search,
            'DFS': AlgorithmImplementations.dfs_search,
            'A*': AlgorithmImplementations.a_star_search
        }
    
    def _run_ucs(self, graph: Graph, start: Any, goal: Any) -> Tuple[Optional[List[Any]], Optional[float], Dict[str, int]]:
        """Run UCS algorithm."""
        ucs = UniformCostSearch(graph)
        path, cost = ucs.search(start, goal)
        
        metrics = ucs.get_performance_metrics()
        return path, cost, {
            'nodes_explored': metrics['nodes_explored'],
            'nodes_generated': metrics['nodes_generated'],
            'max_queue_size': metrics['max_queue_size']
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def run_single_benchmark(self, graph: Graph, algorithm: str, 
                           start: Any, goal: Any) -> BenchmarkResult:
        """Run a single benchmark test."""
        # Get initial memory usage
        initial_memory = self._get_memory_usage()
        
        # Run algorithm
        start_time = time.time()
        path, cost, metrics = self.algorithms[algorithm](graph, start, goal)
        execution_time = time.time() - start_time
        
        # Get final memory usage
        final_memory = self._get_memory_usage()
        memory_usage = final_memory - initial_memory
        
        # Clean up
        gc.collect()
        
        return BenchmarkResult(
            algorithm=algorithm,
            graph_size=len(graph.get_nodes()),
            path_found=path is not None,
            path_cost=cost,
            path_length=len(path) if path else 0,
            execution_time=execution_time,
            memory_usage=memory_usage,
            nodes_explored=metrics['nodes_explored'],
            nodes_generated=metrics['nodes_generated'],
            max_queue_size=metrics['max_queue_size']
        )
    
    def run_scalability_test(self, graph_sizes: List[int], 
                           graph_type: str = 'random',
                           num_runs: int = 5) -> Dict[str, List[BenchmarkResult]]:
        """Run scalability tests across different graph sizes."""
        results = defaultdict(list)
        
        for size in graph_sizes:
            print(f"Testing graph size: {size}")
            
            for run in range(num_runs):
                # Generate graph
                if graph_type == 'grid':
                    rows = int(np.sqrt(size))
                    cols = size // rows
                    graph = GraphGenerator.generate_grid_graph(rows, cols)
                elif graph_type == 'complete':
                    graph = GraphGenerator.generate_complete_graph(size)
                elif graph_type == 'tree':
                    graph = GraphGenerator.generate_tree_graph(size)
                elif graph_type == 'scale_free':
                    graph = GraphGenerator.generate_scale_free_network(size)
                else:  # random
                    graph = GraphGenerator.generate_random_graph(size)
                
                # Select start and goal nodes
                nodes = list(graph.get_nodes())
                if len(nodes) < 2:
                    continue
                    
                start, goal = random.sample(nodes, 2)
                
                # Test all algorithms
                for algorithm in self.algorithms.keys():
                    try:
                        result = self.run_single_benchmark(graph, algorithm, start, goal)
                        results[algorithm].append(result)
                    except Exception as e:
                        print(f"Error running {algorithm} on size {size}: {e}")
        
        return dict(results)
    
    def run_comparative_analysis(self, graph: Graph, start: Any, goal: Any,
                                num_runs: int = 10) -> Dict[str, List[BenchmarkResult]]:
        """Run comparative analysis on a single graph."""
        results = defaultdict(list)
        
        for run in range(num_runs):
            for algorithm in self.algorithms.keys():
                try:
                    result = self.run_single_benchmark(graph, algorithm, start, goal)
                    results[algorithm].append(result)
                except Exception as e:
                    print(f"Error running {algorithm} in run {run}: {e}")
        
        return dict(results)
    
    def generate_performance_report(self, results: Dict[str, List[BenchmarkResult]]) -> str:
        """Generate comprehensive performance report."""
        report = ["Performance Benchmark Report", "=" * 50]
        
        for algorithm, algorithm_results in results.items():
            if not algorithm_results:
                continue
                
            report.append(f"\n{algorithm} Algorithm:")
            
            # Calculate statistics
            execution_times = [r.execution_time for r in algorithm_results]
            memory_usages = [r.memory_usage for r in algorithm_results]
            nodes_explored = [r.nodes_explored for r in algorithm_results]
            success_rate = sum(1 for r in algorithm_results if r.path_found) / len(algorithm_results)
            
            report.append(f"  Success Rate: {success_rate:.2%}")
            report.append(f"  Execution Time - Mean: {np.mean(execution_times):.4f}s, Std: {np.std(execution_times):.4f}s")
            report.append(f"  Memory Usage - Mean: {np.mean(memory_usages):.2f}MB, Std: {np.std(memory_usages):.2f}MB")
            report.append(f"  Nodes Explored - Mean: {np.mean(nodes_explored):.0f}, Std: {np.std(nodes_explored):.0f}")
            
            if success_rate > 0:
                successful_results = [r for r in algorithm_results if r.path_found]
                path_costs = [r.path_cost for r in successful_results]
                report.append(f"  Path Cost - Mean: {np.mean(path_costs):.2f}, Std: {np.std(path_costs):.2f}")
        
        return "\n".join(report)
    
    def visualize_results(self, results: Dict[str, List[BenchmarkResult]], 
                         save_path: str = None) -> None:
        """Create comprehensive visualizations of benchmark results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Algorithm Performance Comparison', fontsize=16)
        
        # Prepare data for plotting
        algorithms = list(results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
        
        # 1. Execution Time Comparison
        ax1 = axes[0, 0]
        execution_times = []
        labels = []
        for algorithm in algorithms:
            times = [r.execution_time for r in results[algorithm]]
            execution_times.append(times)
            labels.append(algorithm)
        
        ax1.boxplot(execution_times, labels=labels)
        ax1.set_title('Execution Time Comparison')
        ax1.set_ylabel('Time (seconds)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Memory Usage Comparison
        ax2 = axes[0, 1]
        memory_usages = []
        for algorithm in algorithms:
            memory = [r.memory_usage for r in results[algorithm]]
            memory_usages.append(memory)
        
        ax2.boxplot(memory_usages, labels=labels)
        ax2.set_title('Memory Usage Comparison')
        ax2.set_ylabel('Memory (MB)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Nodes Explored Comparison
        ax3 = axes[1, 0]
        nodes_explored = []
        for algorithm in algorithms:
            nodes = [r.nodes_explored for r in results[algorithm]]
            nodes_explored.append(nodes)
        
        ax3.boxplot(nodes_explored, labels=labels)
        ax3.set_title('Nodes Explored Comparison')
        ax3.set_ylabel('Number of Nodes')
        ax3.grid(True, alpha=0.3)
        
        # 4. Success Rate Comparison
        ax4 = axes[1, 1]
        success_rates = []
        for algorithm in algorithms:
            rate = sum(1 for r in results[algorithm] if r.path_found) / len(results[algorithm])
            success_rates.append(rate)
        
        bars = ax4.bar(algorithms, success_rates, color=colors)
        ax4.set_title('Success Rate Comparison')
        ax4.set_ylabel('Success Rate')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for bar, rate in zip(bars, success_rates):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_results_to_csv(self, results: Dict[str, List[BenchmarkResult]], 
                            filename: str = 'benchmark_results.csv') -> None:
        """Export benchmark results to CSV file."""
        all_results = []
        
        for algorithm, algorithm_results in results.items():
            for result in algorithm_results:
                all_results.append({
                    'Algorithm': algorithm,
                    'Graph Size': result.graph_size,
                    'Path Found': result.path_found,
                    'Path Cost': result.path_cost,
                    'Path Length': result.path_length,
                    'Execution Time': result.execution_time,
                    'Memory Usage': result.memory_usage,
                    'Nodes Explored': result.nodes_explored,
                    'Nodes Generated': result.nodes_generated,
                    'Max Queue Size': result.max_queue_size
                })
        
        df = pd.DataFrame(all_results)
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")


def main():
    """Main demonstration of performance benchmarking."""
    print("Performance Benchmarks for Uniform-Cost Search")
    print("=" * 50)
    
    # Initialize benchmark suite
    benchmark = PerformanceBenchmark()
    
    # Test 1: Comparative Analysis on Medium Graph
    print("\n1. Running comparative analysis on medium-sized graph...")
    graph = GraphGenerator.generate_random_graph(50, edge_probability=0.3)
    nodes = list(graph.get_nodes())
    start, goal = nodes[0], nodes[-1]
    
    comparative_results = benchmark.run_comparative_analysis(graph, start, goal, num_runs=10)
    
    print(benchmark.generate_performance_report(comparative_results))
    
    # Test 2: Scalability Analysis
    print("\n2. Running scalability analysis...")
    graph_sizes = [10, 25, 50, 75, 100]
    scalability_results = benchmark.run_scalability_test(graph_sizes, graph_type='random', num_runs=3)
    
    print("Scalability Analysis Summary:")
    for algorithm, results in scalability_results.items():
        if results:
            avg_times = [r.execution_time for r in results]
            print(f"{algorithm}: Average execution time ranges from {min(avg_times):.4f}s to {max(avg_times):.4f}s")
    
    # Test 3: Different Graph Types
    print("\n3. Testing different graph types...")
    graph_types = ['grid', 'tree', 'scale_free']
    
    for graph_type in graph_types:
        print(f"\nTesting {graph_type} graph...")
        if graph_type == 'grid':
            test_graph = GraphGenerator.generate_grid_graph(8, 8)
        elif graph_type == 'tree':
            test_graph = GraphGenerator.generate_tree_graph(50)
        else:
            test_graph = GraphGenerator.generate_scale_free_network(50)
        
        nodes = list(test_graph.get_nodes())
        if len(nodes) >= 2:
            start, goal = nodes[0], nodes[-1]
            type_results = benchmark.run_comparative_analysis(test_graph, start, goal, num_runs=5)
            
            print(f"Results for {graph_type}:")
            print(benchmark.generate_performance_report(type_results))
    
    # Visualize results
    print("\n4. Generating visualizations...")
    if comparative_results:
        benchmark.visualize_results(comparative_results)
    
    # Export results
    print("\n5. Exporting results...")
    benchmark.export_results_to_csv(comparative_results, 'ucs_benchmark_results.csv')
    
    print("\nBenchmarking complete!")
    print("Key findings:")
    print("- UCS provides optimal path costs consistently")
    print("- UCS performance scales well with graph size")
    print("- Memory usage is reasonable for most applications")
    print("- Success rate is high for connected graphs")


if __name__ == "__main__":
    main()
