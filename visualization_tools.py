"""
Visualization Tools for Uniform-Cost Search
Professional visualization suite for algorithm analysis and presentation

This module provides comprehensive visualization capabilities including:
- Real-time algorithm animation
- Interactive graph exploration
- Performance metric dashboards
- 3D network visualizations
- Comparative analysis charts
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrow
from matplotlib.widgets import Button, Slider
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from uniform_cost_search import Graph, UniformCostSearch
import time
from collections import deque


class UCSVisualizer:
    """
    Interactive visualizer for Uniform-Cost Search algorithm.
    Provides real-time animation and step-by-step exploration.
    """
    
    def __init__(self, graph: Graph, pos: Optional[Dict] = None):
        self.graph = graph
        self.pos = pos or self._generate_positions()
        self.fig = None
        self.ax = None
        self.animation_data = []
        self.current_step = 0
        
    def _generate_positions(self) -> Dict[str, Tuple[float, float]]:
        """Generate node positions for visualization."""
        nodes = list(self.graph.get_nodes())
        n = len(nodes)
        
        # Use spring layout for better visualization
        G = nx.Graph()
        for node in nodes:
            G.add_node(node)
        
        for node in nodes:
            for neighbor, cost in self.graph.get_neighbors(node):
                G.add_edge(node, neighbor, weight=cost)
        
        pos = nx.spring_layout(G, k=2/np.sqrt(n), iterations=50)
        return pos
    
    def setup_plot(self, start: str, goal: str) -> None:
        """Setup the matplotlib figure and axes."""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_title(f'Uniform-Cost Search: {start} → {goal}', fontsize=14, fontweight='bold')
        self.ax.axis('off')
        
        # Draw graph structure
        self._draw_graph_structure()
        
        # Add control buttons
        self._add_controls()
        
    def _draw_graph_structure(self) -> None:
        """Draw the underlying graph structure."""
        # Draw edges
        for node in self.graph.get_nodes():
            for neighbor, cost in self.graph.get_neighbors(node):
                if neighbor in self.pos:  # Avoid duplicate edges
                    x1, y1 = self.pos[node]
                    x2, y2 = self.pos[neighbor]
                    
                    # Draw edge
                    self.ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.3, linewidth=1)
                    
                    # Add cost label
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    self.ax.text(mid_x, mid_y, f'{cost:.1f}', fontsize=8, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        # Draw nodes
        for node, (x, y) in self.pos.items():
            circle = Circle((x, y), 0.03, color='lightblue', ec='black', linewidth=2)
            self.ax.add_patch(circle)
            self.ax.text(x, y, node, ha='center', va='center', fontsize=10, fontweight='bold')
    
    def _add_controls(self) -> None:
        """Add interactive controls to the plot."""
        # Add step button
        ax_button = plt.axes([0.8, 0.02, 0.1, 0.04])
        btn_step = Button(ax_button, 'Step')
        btn_step.on_clicked(self._step_forward)
        
        # Add reset button
        ax_reset = plt.axes([0.65, 0.02, 0.1, 0.04])
        btn_reset = Button(ax_reset, 'Reset')
        btn_reset.on_clicked(self._reset_animation)
        
        # Add speed slider
        ax_slider = plt.axes([0.15, 0.02, 0.4, 0.03])
        speed_slider = Slider(ax_slider, 'Speed', 0.1, 2.0, valinit=1.0)
        
    def animate_ucs(self, start: str, goal: str, interval: int = 1000) -> None:
        """
        Animate the UCS algorithm step by step.
        
        Args:
            start: Starting node
            goal: Goal node
            interval: Animation interval in milliseconds
        """
        # Run UCS and capture steps
        self.animation_data = self._capture_ucs_steps(start, goal)
        
        if not self.animation_data:
            print("No path found or animation data not available")
            return
        
        # Setup plot
        self.setup_plot(start, goal)
        
        # Create animation
        self.ani = animation.FuncAnimation(
            self.fig, self._update_animation, frames=len(self.animation_data),
            interval=interval, repeat=True, blit=False
        )
        
        plt.show()
    
    def _capture_ucs_steps(self, start: str, goal: str) -> List[Dict]:
        """Capture each step of UCS algorithm for animation."""
        steps = []
        
        # Modified UCS that captures each step
        frontier = [(0, start, [start])]
        cost_so_far = {start: 0}
        explored = set()
        
        while frontier:
            current_cost, current_node, current_path = heapq.heappop(frontier)
            
            # Capture current state
            steps.append({
                'current_node': current_node,
                'current_path': current_path.copy(),
                'current_cost': current_cost,
                'frontier': [(cost, node, path.copy()) for cost, node, path in frontier],
                'explored': explored.copy(),
                'cost_so_far': cost_so_far.copy(),
                'is_goal': current_node == goal
            })
            
            if current_node == goal:
                break
            
            if current_node in explored:
                continue
            
            explored.add(current_node)
            
            for neighbor, edge_cost in self.graph.get_neighbors(current_node):
                new_cost = current_cost + edge_cost
                
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    new_path = current_path + [neighbor]
                    heapq.heappush(frontier, (new_cost, neighbor, new_path))
        
        return steps
    
    def _update_animation(self, frame: int) -> None:
        """Update animation for each frame."""
        if frame >= len(self.animation_data):
            return
        
        step = self.animation_data[frame]
        
        # Clear previous highlights
        for patch in self.ax.patches[:]:
            if isinstance(patch, Circle):
                patch.set_color('lightblue')
        
        # Clear previous arrows
        for patch in self.ax.patches[:]:
            if isinstance(patch, FancyArrow):
                patch.remove()
        
        # Highlight current node
        if step['current_node'] in self.pos:
            x, y = self.pos[step['current_node']]
            for patch in self.ax.patches:
                if isinstance(patch, Circle) and abs(patch.center[0] - x) < 0.01 and abs(patch.center[1] - y) < 0.01:
                    patch.set_color('red')
                    break
        
        # Highlight explored nodes
        for node in step['explored']:
            if node in self.pos:
                x, y = self.pos[node]
                for patch in self.ax.patches:
                    if isinstance(patch, Circle) and abs(patch.center[0] - x) < 0.01 and abs(patch.center[1] - y) < 0.01:
                        patch.set_color('lightgreen')
                        break
        
        # Draw current path
        if len(step['current_path']) > 1:
            for i in range(len(step['current_path']) - 1):
                node1, node2 = step['current_path'][i], step['current_path'][i + 1]
                if node1 in self.pos and node2 in self.pos:
                    x1, y1 = self.pos[node1]
                    x2, y2 = self.pos[node2]
                    arrow = FancyArrow(x1, y1, x2 - x1, y2 - y1, 
                                     width=0.01, head_width=0.03, 
                                     color='blue', alpha=0.7)
                    self.ax.add_patch(arrow)
        
        # Update title with current information
        title = f'UCS Step {frame + 1}: Current Node: {step["current_node"]}, Cost: {step["current_cost"]:.2f}'
        if step['is_goal']:
            title += ' - GOAL REACHED!'
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.draw()
    
    def _step_forward(self, event) -> None:
        """Step forward one frame in animation."""
        if self.current_step < len(self.animation_data) - 1:
            self.current_step += 1
            self._update_animation(self.current_step)
    
    def _reset_animation(self, event) -> None:
        """Reset animation to beginning."""
        self.current_step = 0
        self._update_animation(0)


class InteractiveGraphExplorer:
    """
    Interactive graph explorer for UCS analysis.
    Allows users to click nodes to set start/goal and see optimal paths.
    """
    
    def __init__(self, graph: Graph):
        self.graph = graph
        self.pos = self._generate_positions()
        self.start_node = None
        self.goal_node = None
        self.current_path = None
        
    def _generate_positions(self) -> Dict[str, Tuple[float, float]]:
        """Generate node positions using spring layout."""
        G = nx.Graph()
        for node in self.graph.get_nodes():
            G.add_node(node)
        
        for node in self.graph.get_nodes():
            for neighbor, cost in self.graph.get_neighbors(node):
                G.add_edge(node, neighbor, weight=cost)
        
        return nx.spring_layout(G, k=2/np.sqrt(len(G.nodes())), iterations=50)
    
    def explore(self) -> None:
        """Start interactive exploration."""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_title('Interactive UCS Graph Explorer\nClick nodes to set start (green) and goal (red)', 
                         fontsize=14, fontweight='bold')
        self.ax.axis('off')
        
        # Draw initial graph
        self._draw_graph()
        
        # Connect mouse click event
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        
        # Add instructions
        self.ax.text(0.02, 0.98, 'Left click: Set start node\nRight click: Set goal node\nMiddle click: Find path',
                    transform=self.ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.show()
    
    def _draw_graph(self) -> None:
        """Draw the complete graph."""
        self.ax.clear()
        self.ax.set_title('Interactive UCS Graph Explorer', fontsize=14, fontweight='bold')
        self.ax.axis('off')
        
        # Draw edges
        for node in self.graph.get_nodes():
            for neighbor, cost in self.graph.get_neighbors(node):
                if neighbor in self.pos:
                    x1, y1 = self.pos[node]
                    x2, y2 = self.pos[neighbor]
                    
                    self.ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.3, linewidth=1)
                    
                    # Add cost label
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    self.ax.text(mid_x, mid_y, f'{cost:.1f}', fontsize=8, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        # Draw nodes
        for node, (x, y) in self.pos.items():
            color = 'lightblue'
            if node == self.start_node:
                color = 'lightgreen'
            elif node == self.goal_node:
                color = 'lightcoral'
            
            circle = Circle((x, y), 0.03, color=color, ec='black', linewidth=2)
            self.ax.add_patch(circle)
            self.ax.text(x, y, node, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Draw current path if exists
        if self.current_path:
            for i in range(len(self.current_path) - 1):
                node1, node2 = self.current_path[i], self.current_path[i + 1]
                if node1 in self.pos and node2 in self.pos:
                    x1, y1 = self.pos[node1]
                    x2, y2 = self.pos[node2]
                    self.ax.plot([x1, x2], [y1, y2], 'blue', linewidth=3, alpha=0.7)
        
        # Add instructions
        self.ax.text(0.02, 0.98, 'Left click: Set start node\nRight click: Set goal node\nMiddle click: Find path',
                    transform=self.ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.draw()
    
    def _on_click(self, event) -> None:
        """Handle mouse click events."""
        if event.inaxes != self.ax:
            return
        
        # Find closest node to click
        click_x, click_y = event.xdata, event.ydata
        closest_node = None
        min_distance = float('inf')
        
        for node, (x, y) in self.pos.items():
            distance = np.sqrt((x - click_x)**2 + (y - click_y)**2)
            if distance < min_distance and distance < 0.1:  # Threshold for selection
                min_distance = distance
                closest_node = node
        
        if closest_node is None:
            return
        
        # Handle different click types
        if event.button == 1:  # Left click - set start
            self.start_node = closest_node
            self.current_path = None
        elif event.button == 3:  # Right click - set goal
            self.goal_node = closest_node
            self.current_path = None
        elif event.button == 2:  # Middle click - find path
            if self.start_node and self.goal_node:
                ucs = UniformCostSearch(self.graph)
                path, cost = ucs.search(self.start_node, self.goal_node)
                if path:
                    self.current_path = path
                    print(f"Path found: {' -> '.join(path)}, Cost: {cost:.2f}")
                else:
                    print("No path found")
                    self.current_path = None
        
        # Redraw graph
        self._draw_graph()


class PerformanceDashboard:
    """
    Interactive dashboard for performance metrics visualization.
    Provides real-time monitoring and analysis of UCS performance.
    """
    
    def __init__(self):
        self.metrics_history = []
        self.fig = None
        
    def create_dashboard(self) -> None:
        """Create an interactive performance dashboard."""
        # Create subplots
        self.fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Execution Time', 'Memory Usage', 'Nodes Explored', 'Success Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Update layout
        self.fig.update_layout(
            title='UCS Performance Dashboard',
            height=800,
            showlegend=True
        )
        
        self.fig.show()
    
    def update_metrics(self, algorithm: str, metrics: Dict[str, Any]) -> None:
        """Update dashboard with new metrics."""
        timestamp = time.time()
        self.metrics_history.append({
            'timestamp': timestamp,
            'algorithm': algorithm,
            **metrics
        })
        
        # Update plots
        self._update_plots()
    
    def _update_plots(self) -> None:
        """Update all dashboard plots."""
        if not self.metrics_history:
            return
        
        df = pd.DataFrame(self.metrics_history)
        
        # Clear existing traces
        self.fig.data = []
        
        # Execution Time Plot
        for algorithm in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algorithm]
            self.fig.add_trace(
                go.Scatter(
                    x=algo_data['timestamp'],
                    y=algo_data['execution_time'],
                    mode='lines+markers',
                    name=f'{algorithm} - Time',
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # Memory Usage Plot
        for algorithm in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algorithm]
            self.fig.add_trace(
                go.Scatter(
                    x=algo_data['timestamp'],
                    y=algo_data['memory_usage'],
                    mode='lines+markers',
                    name=f'{algorithm} - Memory',
                    line=dict(width=2)
                ),
                row=1, col=2
            )
        
        # Nodes Explored Plot
        for algorithm in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algorithm]
            self.fig.add_trace(
                go.Scatter(
                    x=algo_data['timestamp'],
                    y=algo_data['nodes_explored'],
                    mode='lines+markers',
                    name=f'{algorithm} - Nodes',
                    line=dict(width=2)
                ),
                row=2, col=1
            )
        
        # Success Rate Plot (bar chart)
        success_rates = df.groupby('algorithm')['path_found'].mean().reset_index()
        self.fig.add_trace(
            go.Bar(
                x=success_rates['algorithm'],
                y=success_rates['path_found'],
                name='Success Rate',
                marker_color='lightblue'
            ),
            row=2, col=2
        )
        
        self.fig.show()


class Network3DVisualizer:
    """
    3D visualization of networks for better spatial understanding.
    Particularly useful for logistics and routing applications.
    """
    
    def __init__(self, graph: Graph, node_positions: Dict[str, Tuple[float, float, float]]):
        self.graph = graph
        self.node_positions = node_positions
    
    def visualize_3d(self, start: str, goal: str, path: Optional[List[str]] = None) -> None:
        """Create 3D visualization of the network."""
        fig = go.Figure()
        
        # Add edges
        edge_x = []
        edge_y = []
        edge_z = []
        
        for node in self.graph.get_nodes():
            for neighbor, cost in self.graph.get_neighbors(node):
                if neighbor in self.node_positions:
                    x1, y1, z1 = self.node_positions[node]
                    x2, y2, z2 = self.node_positions[neighbor]
                    
                    edge_x.extend([x1, x2, None])
                    edge_y.extend([y1, y2, None])
                    edge_z.extend([z1, z2, None])
        
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='gray', width=1),
            hoverinfo='none',
            name='Edges'
        ))
        
        # Add nodes
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        node_colors = []
        
        for node, (x, y, z) in self.node_positions.items():
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_text.append(node)
            
            if node == start:
                node_colors.append('green')
            elif node == goal:
                node_colors.append('red')
            elif path and node in path:
                node_colors.append('blue')
            else:
                node_colors.append('lightblue')
        
        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=8,
                color=node_colors,
                line=dict(width=2, color='black')
            ),
            text=node_text,
            textposition="middle center",
            name='Nodes'
        ))
        
        # Add path if provided
        if path and len(path) > 1:
            path_x = []
            path_y = []
            path_z = []
            
            for node in path:
                if node in self.node_positions:
                    x, y, z = self.node_positions[node]
                    path_x.append(x)
                    path_y.append(y)
                    path_z.append(z)
            
            fig.add_trace(go.Scatter3d(
                x=path_x, y=path_y, z=path_z,
                mode='lines+markers',
                line=dict(color='red', width=4),
                marker=dict(size=6),
                name='Optimal Path'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'3D Network Visualization: {start} → {goal}',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600
        )
        
        fig.show()


def create_sample_3d_positions(graph: Graph) -> Dict[str, Tuple[float, float, float]]:
    """Create sample 3D positions for graph nodes."""
    positions = {}
    nodes = list(graph.get_nodes())
    
    # Generate 3D positions using spherical coordinates
    n = len(nodes)
    for i, node in enumerate(nodes):
        theta = 2 * np.pi * i / n
        phi = np.pi * (i % 3) / 3  # Create 3 layers
        
        r = 5 + np.random.uniform(-1, 1)  # Random radius variation
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        positions[node] = (x, y, z)
    
    return positions


def main():
    """Main demonstration of visualization tools."""
    print("UCS Visualization Tools Demo")
    print("=" * 40)
    
    # Create sample graph
    graph = Graph(directed=False)
    edges = [
        ('A', 'B', 4), ('A', 'C', 2), ('B', 'C', 1),
        ('B', 'D', 5), ('C', 'D', 8), ('C', 'E', 10),
        ('D', 'E', 2), ('D', 'F', 6), ('E', 'F', 3)
    ]
    
    for from_node, to_node, cost in edges:
        graph.add_edge(from_node, to_node, cost)
    
    # 1. UCS Animation
    print("1. Starting UCS animation...")
    visualizer = UCSVisualizer(graph)
    visualizer.animate_ucs('A', 'F', interval=1500)
    
    # 2. Interactive Explorer
    print("2. Starting interactive graph explorer...")
    explorer = InteractiveGraphExplorer(graph)
    explorer.explore()
    
    # 3. Performance Dashboard
    print("3. Creating performance dashboard...")
    dashboard = PerformanceDashboard()
    dashboard.create_dashboard()
    
    # Simulate some metrics updates
    ucs = UniformCostSearch(graph)
    path, cost = ucs.search('A', 'F')
    metrics = ucs.get_performance_metrics()
    metrics['path_found'] = path is not None
    metrics['execution_time'] = 0.1  # Simulated
    metrics['memory_usage'] = 5.2  # Simulated
    
    dashboard.update_metrics('UCS', metrics)
    
    # 4. 3D Visualization
    print("4. Creating 3D network visualization...")
    positions_3d = create_sample_3d_positions(graph)
    visualizer_3d = Network3DVisualizer(graph, positions_3d)
    visualizer_3d.visualize_3d('A', 'F', path)
    
    print("Visualization demos complete!")


if __name__ == "__main__":
    main()
