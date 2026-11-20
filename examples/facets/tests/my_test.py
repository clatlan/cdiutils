import socket
from ewokscore import Task

class GetHostname(Task, output_names=["hostname"]):
    def run(self):
        self.outputs.hostname = socket.gethostname()