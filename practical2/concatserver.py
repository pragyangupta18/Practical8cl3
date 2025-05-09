# concat_server.py
from Pyro5.api import expose, Daemon, locate_ns

@expose               # mark methods as remotely accessible
class ConcatService:
    def concat(self, a: str, b: str) -> str:
        print(f"Serving request: {a!r} + {b!r}")
        return a + b

def main():
    # 1️⃣ Start (or find) the Pyro name server
    ns = locate_ns()                # will raise if nameserver isn't running
    # 2️⃣ Make a Pyro daemon and register the remote object
    with Daemon() as daemon:
        uri = daemon.register(ConcatService)      # create a URI
        ns.register("example.concat", uri)        # bind name ➜ URI
        print("ConcatService is ready:", uri)
        daemon.requestLoop()                      # event loop

if __name__ == "__main__":
    main()
