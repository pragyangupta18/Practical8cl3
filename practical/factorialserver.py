from xmlrpc.server import SimpleXMLRPCServer

def calculate_factorial(n):
    if n < 0:
        return "Invalid input! Factorial of negative number doesn't exist."
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

# Create server
server = SimpleXMLRPCServer(("localhost", 8000))
print("Server is running on port 8000...")

# Register function
server.register_function(calculate_factorial, "calculate_factorial")

# Run the server
server.serve_forever()
