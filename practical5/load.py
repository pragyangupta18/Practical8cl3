import random
import time

# Server class to represent each server
class Server:
    def __init__(self, id):
        self.id = id
        self.current_load = 0  # Tracks the number of active requests
        self.total_requests = 0  # Tracks total requests handled

    def handle_request(self):
        self.current_load += 1
        self.total_requests += 1

    def finish_request(self):
        self.current_load -= 1


# Load Balancer class with multiple algorithms
class LoadBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.algorithm = "round_robin"
        self.last_index = 0

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm

    def round_robin(self, request):
        server = self.servers[self.last_index]
        self.last_index = (self.last_index + 1) % len(self.servers)
        print(f"Request {request} sent to Server {server.id}")
        server.handle_request()

    def least_connections(self, request):
        server = min(self.servers, key=lambda s: s.current_load)
        print(f"Request {request} sent to Server {server.id}")
        server.handle_request()

    def distribute_request(self, request):
        if self.algorithm == "round_robin":
            self.round_robin(request)
        elif self.algorithm == "least_connections":
            self.least_connections(request)


# Simulate client requests
def simulate_requests(load_balancer, num_requests):
    for i in range(1, num_requests + 1):
        load_balancer.distribute_request(i)
        time.sleep(random.uniform(0.1, 0.5))  # Simulating time between requests

    # Finish some requests to simulate load reduction
    for server in load_balancer.servers:
        if server.current_load > 0:
            server.finish_request()

# Create servers and load balancer
servers = [Server(id=i) for i in range(1, 4)]  # 3 servers
load_balancer = LoadBalancer(servers)

# Set the load balancing algorithm
load_balancer.set_algorithm("round_robin")  # Use round_robin or least_connections

# Simulate 10 requests
simulate_requests(load_balancer, 10)
