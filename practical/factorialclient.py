import xmlrpc.client

# Connect to the server
proxy = xmlrpc.client.ServerProxy("http://localhost:8000/")

# Input from user
n = int(input("Enter an integer to compute factorial: "))

# Call remote procedure
result = proxy.calculate_factorial(n)

# Display result
print(f"Factorial of {n} is: {result}")
