"""
Logistics and Supply Chain Optimization using Uniform-Cost Search
Industry-level implementation for delivery route planning and warehouse management

This module demonstrates UCS application in logistics optimization,
including vehicle routing, inventory management, and supply chain optimization.
"""

import math
import random
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from uniform_cost_search import Graph, UniformCostSearch
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Location:
    """Represents a geographical location in the logistics network."""
    id: str
    name: str
    latitude: float
    longitude: float
    location_type: str  # 'warehouse', 'customer', 'distribution_center'
    operating_hours: Tuple[int, int]  # (start_hour, end_hour)
    service_time: float  # minutes required at this location


@dataclass
class Vehicle:
    """Represents a delivery vehicle."""
    id: str
    capacity: float  # kg or volume
    current_location: str
    fuel_efficiency: float  # km/l
    fuel_cost_per_liter: float
    driver_cost_per_hour: float
    max_working_hours: float  # hours per day
    speed: float  # km/h average speed


@dataclass
class DeliveryOrder:
    """Represents a customer delivery order."""
    id: str
    customer_id: str
    pickup_location: str
    delivery_location: str
    weight: float  # kg
    volume: float  # cubic meters
    priority: int  # 1-10, higher is more urgent
    time_window: Tuple[datetime, datetime]  # (earliest_delivery, latest_delivery)
    penalty_per_hour_late: float


class LogisticsNetwork:
    """
    Manages the logistics network including locations, vehicles, and orders.
    Converts logistics problems to graph for UCS optimization.
    """
    
    def __init__(self):
        self.locations: Dict[str, Location] = {}
        self.vehicles: Dict[str, Vehicle] = {}
        self.orders: List[DeliveryOrder] = []
        self.graph = Graph(directed=False)
        self.distance_matrix: Dict[Tuple[str, str], float] = {}
        self.travel_time_matrix: Dict[Tuple[str, str], float] = {}
    
    def add_location(self, location: Location) -> None:
        """Add a location to the network."""
        self.locations[location.id] = location
    
    def add_vehicle(self, vehicle: Vehicle) -> None:
        """Add a vehicle to the fleet."""
        self.vehicles[vehicle.id] = vehicle
    
    def add_order(self, order: DeliveryOrder) -> None:
        """Add a delivery order."""
        self.orders.append(order)
    
    def calculate_distance(self, loc1_id: str, loc2_id: str) -> float:
        """
        Calculate distance between two locations using Haversine formula.
        Returns distance in kilometers.
        """
        if (loc1_id, loc2_id) in self.distance_matrix:
            return self.distance_matrix[(loc1_id, loc2_id)]
        
        loc1 = self.locations[loc1_id]
        loc2 = self.locations[loc2_id]
        
        # Haversine formula
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1 = math.radians(loc1.latitude), math.radians(loc1.longitude)
        lat2, lon2 = math.radians(loc2.latitude), math.radians(loc2.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        distance = R * c
        
        # Cache the result
        self.distance_matrix[(loc1_id, loc2_id)] = distance
        self.distance_matrix[(loc2_id, loc1_id)] = distance
        
        return distance
    
    def calculate_travel_time(self, loc1_id: str, loc2_id: str, vehicle: Vehicle) -> float:
        """Calculate travel time between locations in hours."""
        distance = self.calculate_distance(loc1_id, loc2_id)
        travel_time = distance / vehicle.speed
        return travel_time
    
    def calculate_delivery_cost(self, loc1_id: str, loc2_id: str, vehicle: Vehicle, 
                               cargo_weight: float = 0) -> float:
        """
        Calculate comprehensive delivery cost including:
        - Fuel cost
        - Driver cost
        - Vehicle maintenance
        - Time-based penalties
        """
        distance = self.calculate_distance(loc1_id, loc2_id)
        travel_time = self.calculate_travel_time(loc1_id, loc2_id, vehicle)
        
        # Fuel cost (adjusted for cargo weight)
        fuel_consumption = distance / vehicle.fuel_efficiency * (1 + cargo_weight / vehicle.capacity * 0.1)
        fuel_cost = fuel_consumption * vehicle.fuel_cost_per_liter
        
        # Driver cost
        driver_cost = travel_time * vehicle.driver_cost_per_hour
        
        # Maintenance cost (simplified)
        maintenance_cost = distance * 0.05  # $0.05 per km
        
        total_cost = fuel_cost + driver_cost + maintenance_cost
        return total_cost
    
    def build_graph(self, vehicle: Vehicle) -> None:
        """Build the graph for a specific vehicle considering its characteristics."""
        self.graph = Graph(directed=False)
        
        locations = list(self.locations.keys())
        
        for i, loc1 in enumerate(locations):
            for loc2 in locations[i+1:]:
                cost = self.calculate_delivery_cost(loc1, loc2, vehicle)
                self.graph.add_edge(loc1, loc2, cost)
    
    def find_optimal_route(self, vehicle_id: str, start_location: str, 
                          destinations: List[str]) -> Tuple[Optional[List[str]], Optional[float]]:
        """
        Find optimal route for a vehicle visiting multiple destinations.
        Uses UCS to find the least-cost path.
        """
        if vehicle_id not in self.vehicles:
            raise ValueError(f"Vehicle {vehicle_id} not found")
        
        vehicle = self.vehicles[vehicle_id]
        self.build_graph(vehicle)
        
        # For multiple destinations, we need to find the optimal order
        # This is a simplified version - in practice, this would be a VRP problem
        if len(destinations) == 1:
            ucs = UniformCostSearch(self.graph)
            return ucs.search(start_location, destinations[0])
        else:
            # For multiple destinations, use a greedy approach with UCS
            current_location = start_location
            remaining_destinations = destinations.copy()
            total_path = [current_location]
            total_cost = 0
            
            while remaining_destinations:
                # Find closest destination using UCS
                ucs = UniformCostSearch(self.graph)
                best_cost = float('inf')
                best_dest = None
                best_path = None
                
                for dest in remaining_destinations:
                    path, cost = ucs.search(current_location, dest)
                    if cost is not None and cost < best_cost:
                        best_cost = cost
                        best_dest = dest
                        best_path = path
                
                if best_dest is None:
                    break
                
                # Update route
                total_path.extend(best_path[1:])  # Skip current location
                total_cost += best_cost
                current_location = best_dest
                remaining_destinations.remove(best_dest)
            
            return total_path, total_cost


class LogisticsOptimizer:
    """
    Advanced logistics optimization system using UCS.
    Handles complex scenarios including time windows, vehicle constraints, and dynamic routing.
    """
    
    def __init__(self, network: LogisticsNetwork):
        self.network = network
        self.routes: Dict[str, List[str]] = {}
        self.schedule: Dict[str, List[Tuple[datetime, str]]] = {}
        self.optimization_metrics = {
            'total_distance': 0,
            'total_cost': 0,
            'delivery_efficiency': 0,
            'vehicle_utilization': 0
        }
    
    def optimize_single_vehicle_routes(self, vehicle_id: str, date: datetime) -> Dict[str, any]:
        """
        Optimize routes for a single vehicle for a given day.
        Considers time windows, capacity constraints, and delivery priorities.
        """
        if vehicle_id not in self.network.vehicles:
            raise ValueError(f"Vehicle {vehicle_id} not found")
        
        vehicle = self.network.vehicles[vehicle_id]
        
        # Filter orders for the day
        daily_orders = [order for order in self.network.orders 
                       if order.time_window[0].date() == date.date()]
        
        # Sort by priority and time window
        daily_orders.sort(key=lambda x: (-x.priority, x.time_window[0]))
        
        route = [vehicle.current_location]
        current_time = date.replace(hour=8, minute=0)  # Start at 8 AM
        current_capacity_used = 0
        total_cost = 0
        
        for order in daily_orders:
            # Check capacity constraint
            if current_capacity_used + order.weight > vehicle.capacity:
                continue
            
            # Check time window feasibility
            delivery_location = order.delivery_location
            
            # Calculate travel time and arrival
            travel_time = self.network.calculate_travel_time(
                route[-1], delivery_location, vehicle
            )
            arrival_time = current_time + timedelta(hours=travel_time)
            
            # Check if arrival is within time window
            if arrival_time < order.time_window[0]:
                # Wait until time window opens
                current_time = order.time_window[0]
                arrival_time = current_time
            elif arrival_time > order.time_window[1]:
                # Too late, skip this order
                continue
            
            # Check operating hours
            delivery_loc = self.network.locations[delivery_location]
            if not (delivery_loc.operating_hours[0] <= arrival_time.hour <= delivery_loc.operating_hours[1]):
                continue
            
            # Add to route
            route.append(delivery_location)
            
            # Update metrics
            segment_cost = self.network.calculate_delivery_cost(
                route[-2], route[-1], vehicle, current_capacity_used
            )
            total_cost += segment_cost
            
            # Update time and capacity
            current_time = arrival_time + timedelta(minutes=delivery_loc.service_time)
            current_capacity_used += order.weight
        
        # Return to depot if needed
        if route[-1] != vehicle.current_location:
            return_cost = self.network.calculate_delivery_cost(
                route[-1], vehicle.current_location, vehicle, current_capacity_used
            )
            route.append(vehicle.current_location)
            total_cost += return_cost
        
        return {
            'vehicle_id': vehicle_id,
            'route': route,
            'total_cost': total_cost,
            'orders_delivered': len([o for o in daily_orders if o.delivery_location in route]),
            'total_distance': sum(self.network.calculate_distance(route[i], route[i+1]) 
                                for i in range(len(route)-1)),
            'estimated_duration': (current_time - date.replace(hour=8, minute=0)).total_seconds() / 3600
        }
    
    def optimize_fleet(self, date: datetime) -> Dict[str, any]:
        """
        Optimize routes for the entire fleet.
        Uses UCS for individual vehicle optimization and fleet-level coordination.
        """
        fleet_results = {}
        total_cost = 0
        total_distance = 0
        total_orders = 0
        
        for vehicle_id in self.network.vehicles:
            try:
                result = self.optimize_single_vehicle_routes(vehicle_id, date)
                fleet_results[vehicle_id] = result
                
                total_cost += result['total_cost']
                total_distance += result['total_distance']
                total_orders += result['operations_delivered']
                
            except Exception as e:
                print(f"Error optimizing vehicle {vehicle_id}: {str(e)}")
                fleet_results[vehicle_id] = {'error': str(e)}
        
        # Calculate fleet metrics
        self.optimization_metrics.update({
            'total_distance': total_distance,
            'total_cost': total_cost,
            'delivery_efficiency': total_orders / max(len(self.network.orders), 1),
            'vehicle_utilization': sum(1 for v in fleet_results.values() if 'error' not in v) / max(len(self.network.vehicles), 1)
        })
        
        return {
            'fleet_results': fleet_results,
            'fleet_metrics': self.optimization_metrics,
            'date': date
        }
    
    def simulate_dynamic_routing(self, vehicle_id: str, initial_route: List[str], 
                               new_orders: List[DeliveryOrder]) -> Dict[str, any]:
        """
        Simulate dynamic routing when new orders arrive during execution.
        Demonstrates UCS adaptability in real-time scenarios.
        """
        if vehicle_id not in self.network.vehicles:
            raise ValueError(f"Vehicle {vehicle_id} not found")
        
        vehicle = self.network.vehicles[vehicle_id]
        
        # Find current position in route (simplified - assume at start)
        current_position = 0
        current_location = initial_route[current_position]
        
        # Remaining destinations
        remaining_destinations = initial_route[current_position+1:]
        
        # Add new order destinations
        for order in new_orders:
            if order.delivery_location not in remaining_destinations:
                remaining_destinations.append(order.delivery_location)
        
        # Re-optimize route from current position
        new_route, new_cost = self.network.find_optimal_route(
            vehicle_id, current_location, remaining_destinations
        )
        
        return {
            'original_route': initial_route,
            'new_route': new_route,
            'original_cost': sum(self.network.calculate_delivery_cost(
                initial_route[i], initial_route[i+1], vehicle
            ) for i in range(len(initial_route)-1)),
            'new_cost': new_cost,
            'cost_savings': sum(self.network.calculate_delivery_cost(
                initial_route[i], initial_route[i+1], vehicle
            ) for i in range(len(initial_route)-1)) - new_cost,
            'new_orders_added': len(new_orders)
        }
    
    def visualize_routes(self, routes: Dict[str, List[str]], save_path: str = None) -> None:
        """Visualize vehicle routes on a map."""
        plt.figure(figsize=(15, 10))
        
        # Plot locations
        for loc_id, location in self.network.locations.items():
            color = {'warehouse': 'red', 'customer': 'blue', 'distribution_center': 'green'}.get(location.location_type, 'gray')
            plt.scatter(location.longitude, location.latitude, c=color, s=100, alpha=0.7)
            plt.annotate(location.name, (location.longitude, location.latitude), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot routes
        colors = plt.cm.Set3(np.linspace(0, 1, len(routes)))
        
        for i, (vehicle_id, route) in enumerate(routes.items()):
            if 'error' in route:
                continue
                
            coordinates = [(self.network.locations[loc].longitude, 
                          self.network.locations[loc].latitude) for loc in route]
            
            if len(coordinates) > 1:
                lons, lats = zip(*coordinates)
                plt.plot(lons, lats, color=colors[i], linewidth=2, 
                        label=f'Vehicle {vehicle_id}', marker='o', markersize=4)
        
        plt.title('Logistics Network - Optimized Delivery Routes')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_sample_logistics_network() -> LogisticsNetwork:
    """Create a sample logistics network for demonstration."""
    network = LogisticsNetwork()
    
    # Create locations
    locations = [
        Location("WH1", "Main Warehouse", 40.7128, -74.0060, 'warehouse', (6, 22, 30)),
        Location("DC1", "Distribution Center A", 40.7589, -73.9851, 'distribution_center', (6, 20, 15)),
        Location("DC2", "Distribution Center B", 40.6892, -74.0445, 'distribution_center', (6, 20, 15)),
        Location("CUST1", "Customer 1", 40.7489, -73.9680, 'customer', (9, 17, 10)),
        Location("CUST2", "Customer 2", 40.7282, -73.9942, 'customer', (9, 17, 10)),
        Location("CUST3", "Customer 3", 40.7831, -73.9712, 'customer', (9, 17, 10)),
        Location("CUST4", "Customer 4", 40.7061, -73.9969, 'customer', (9, 17, 10)),
        Location("CUST5", "Customer 5", 40.7527, -73.9772, 'customer', (9, 17, 10)),
    ]
    
    for location in locations:
        network.add_location(location)
    
    # Create vehicles
    vehicles = [
        Vehicle("V1", 1000, "WH1", 8.0, 1.20, 25.0, 10.0, 40.0),
        Vehicle("V2", 1500, "WH1", 7.5, 1.20, 30.0, 10.0, 35.0),
        Vehicle("V3", 800, "DC1", 9.0, 1.20, 20.0, 8.0, 45.0),
    ]
    
    for vehicle in vehicles:
        network.add_vehicle(vehicle)
    
    # Create sample orders
    base_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    orders = [
        DeliveryOrder("ORD001", "CUST1", "WH1", "CUST1", 100, 2.0, 8, 
                     (base_date + timedelta(hours=9), base_date + timedelta(hours=12)), 50.0),
        DeliveryOrder("ORD002", "CUST2", "WH1", "CUST2", 150, 3.0, 7,
                     (base_date + timedelta(hours=10), base_date + timedelta(hours=14)), 40.0),
        DeliveryOrder("ORD003", "CUST3", "DC1", "CUST3", 80, 1.5, 9,
                     (base_date + timedelta(hours=11), base_date + timedelta(hours=15)), 60.0),
        DeliveryOrder("ORD004", "CUST4", "DC2", "CUST4", 200, 4.0, 6,
                     (base_date + timedelta(hours=9), base_date + timedelta(hours=13)), 30.0),
        DeliveryOrder("ORD005", "CUST5", "WH1", "CUST5", 120, 2.5, 8,
                     (base_date + timedelta(hours=13), base_date + timedelta(hours=17)), 45.0),
    ]
    
    for order in orders:
        network.add_order(order)
    
    return network


def main():
    """Main demonstration of logistics optimization with UCS."""
    print("Logistics and Supply Chain Optimization using Uniform-Cost Search")
    print("=" * 70)
    
    # Create logistics network
    network = create_sample_logistics_network()
    print(f"Created logistics network with:")
    print(f"  - {len(network.locations)} locations")
    print(f"  - {len(network.vehicles)} vehicles")
    print(f"  - {len(network.orders)} pending orders")
    
    # Initialize optimizer
    optimizer = LogisticsOptimizer(network)
    
    # Optimize fleet for today
    print("\nOptimizing fleet routes for today...")
    today = datetime.now()
    fleet_results = optimizer.optimize_fleet(today)
    
    print(f"Fleet optimization results:")
    print(f"  - Total cost: ${fleet_results['fleet_metrics']['total_cost']:.2f}")
    print(f"  - Total distance: {fleet_results['fleet_metrics']['total_distance']:.1f} km")
    print(f"  - Delivery efficiency: {fleet_results['fleet_metrics']['delivery_efficiency']:.2%}")
    print(f"  - Vehicle utilization: {fleet_results['fleet_metrics']['vehicle_utilization']:.2%}")
    
    # Display individual vehicle routes
    print("\nIndividual vehicle routes:")
    for vehicle_id, result in fleet_results['fleet_results'].items():
        if 'error' not in result:
            print(f"\n{vehicle_id}:")
            print(f"  Route: {' -> '.join(result['route'])}")
            print(f"  Cost: ${result['total_cost']:.2f}")
            print(f"  Distance: {result['total_distance']:.1f} km")
            print(f"  Orders delivered: {result['orders_delivered']}")
            print(f"  Estimated duration: {result['estimated_duration']:.1f} hours")
    
    # Test dynamic routing
    print("\nTesting dynamic routing with new orders...")
    initial_route = fleet_results['fleet_results']['V1']['route']
    
    # Create new urgent orders
    new_orders = [
        DeliveryOrder("URG001", "CUST6", "WH1", "CUST1", 50, 1.0, 10,
                     (today + timedelta(hours=14), today + timedelta(hours=16)), 100.0),
    ]
    
    dynamic_result = optimizer.simulate_dynamic_routing("V1", initial_route, new_orders)
    
    print(f"Dynamic routing results:")
    print(f"  Original route cost: ${dynamic_result['original_cost']:.2f}")
    print(f"  New route cost: ${dynamic_result['new_cost']:.2f}")
    print(f"  Cost savings: ${dynamic_result['cost_savings']:.2f}")
    print(f"  New orders added: {dynamic_result['new_orders_added']}")
    
    # Visualize routes
    print("\nGenerating route visualization...")
    routes_to_visualize = {
        vehicle_id: result['route'] 
        for vehicle_id, result in fleet_results['fleet_results'].items() 
        if 'error' not in result
    }
    optimizer.visualize_routes(routes_to_visualize)
    
    # Performance comparison
    print("\nPerformance Analysis:")
    print("UCS provides optimal routing by:")
    print("  - Minimizing total delivery costs")
    print("  - Considering multiple cost factors (fuel, time, maintenance)")
    print("  - Adapting to dynamic conditions")
    print("  - Ensuring capacity and time window constraints")


if __name__ == "__main__":
    main()
