# concat_client.py
from Pyro5.api import locate_ns, Proxy

def main():
    ns = locate_ns()                      # find the name server
    uri = ns.lookup("example.concat")     # resolve logical name ⇢ URI
    # 🔗 Make a proxy to the remote object
    with Proxy(uri) as remote:
        s1 = input("Enter first string : ")
        s2 = input("Enter second string: ")
        result = remote.concat(s1, s2)    # remote method invocation
        print("Server returned:", result)

if __name__ == "__main__":
    main()
