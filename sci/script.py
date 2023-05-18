"""Custom topology to explore channel state estimation methods:
    host --- switch ---  host
"""

import argparse
from os import kill

from mininet.topo import Topo
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.cli import CLI
import numpy as np
import sched
import threading

# Custom Thread Class
class CustomThread(threading.Thread):
    """Class for handling exceptions when use threading."""
    
    def run(self, *args, **kwargs):
        self.exc = None          
        try:
            super().run(*args, **kwargs)
        except Exception as e:
            self.exc = e
       
    def join(self, *args, **kwargs):
        """Re-araise exception in main thread after join if any."""
        super().join(*args, **kwargs)
        if self.exc:
            raise self.exc
        

from time import sleep
MTU = 1400

BW1 = 1000
BW2 = 500
LOW_PRIO = 10
MIDDLE_PRIO = 5
HIGH_PRIO = 1
HIGHEST_PRIO = 0

server_port = "1230"
fec_algo = "fec_dummy"
fec_payload_size = "1000"
fec_batch_size = "8"
fec_leo_nrecovery = ""
duration = 5
PAYLOAD_SIZE = 972
LOG_GAP = 0.5
rate = "10"
loss = 0.00

QUEUE_TIME_SCALE = 0.1

def writeStats(node, dev, file):
    """If dev is None than write stats for all devs."""
    dev_cmd = f"dev {dev}"
    if dev is None:
        dev_cmd = ""
    node.cmd(f"tc -s -d qdisc show {dev_cmd} >> {file}")
    node.cmd(f"echo -ne '\n' >> {file}")

class MyTopo( Topo ):
    "Test topo host1---switch---host2"

    def build( self ):
        "Create custom topo"

        # Add hosts
        leftHost = self.addHost( 'h1' )
        rightHost = self.addHost( 'h2' )

        # Add switch
        switch = self.addSwitch('s1')

        # Add links
        #lParams = {'bw':100, 'delay': "3ms", "max_queue_size": 10000}

        self.addLink( leftHost, switch )#, **lParams )
        self.addLink( rightHost, switch )#, **lParams )


def createNetwork() -> Mininet:
    """Create network with configured links"""
    topo = MyTopo()
    net = Mininet( topo, link=TCLink , controller=None)
    # net = Mininet( topo, link=TCLink )
    # et = CustomMininet ( topo, link=CustomTCLink, controller=None )
    h1, h2, s1 = net.get( 'h1', 'h2', 's1' )
    link1 = net.linksBetween( h1, s1 )[0] #net.linksBetween() returns a list 
    link2 = net.linksBetween( h2, s1 )[0]
    # s1.cmd("ovs-vsctl set-fail-mode s1 standalone") # agreement: switch name is 's1'
    # Configure link's parameters on both interfaces
    #warning: intf.config(...) make NEW configuration so all params are reseted
    #(you need to set ALL old params again)

    #print('link1:', link1.intf1, link1.intf2)
    #print('link2:', link2.intf1, link2.intf2)

    intf1Params = {'bw':BW1, 'delay': "0ms", "max_queue_size": 10}
    intf2Params = {'bw':BW2, 'delay': "0ms", "max_queue_size": 10}
    intf3Params = {'bw':BW2, 'delay': "0ms", "max_queue_size": 10}
    intf4Params = {'bw':BW2, 'delay': "0ms", "max_queue_size": 1000}
    link1.intf1.config( **intf1Params )
    link1.intf2.config( **intf2Params )
    link2.intf1.config( **intf3Params )
    link2.intf2.config( **intf4Params )
    print()

    return net

def setupTunnel(net: Mininet, 
                server_path: str,
                client_path: str,
                server_port: int, 
                fec_algo: str,
                fec_batch_size: int,
                fec_payload_size: int,
                fec_leo_nrecovery: str, 
                server_log_path: str,
                client_log_path: str,
                loss: float):
    h1, h2 = net.get('h1', 'h2')

    h2.cmd(f"{server_path} 10.0.0.2 {server_port} "
                   f"{fec_algo} {fec_batch_size} "
                   f"{fec_payload_size} {fec_leo_nrecovery} &> {server_log_path} &")
    h1.cmd(f"{client_path} "
            f"10.0.0.2 {server_port} --loss={loss} "
            f"{fec_algo} {fec_batch_size} "
            f"{fec_payload_size} {fec_leo_nrecovery} &> {client_log_path} &")
    sleep(0.1)

    h2.cmd(f"ifconfig tun1 mtu {MTU} up 10.1.0.2 netmask 255.255.255.0")
    h1.cmd(f"ifconfig tun0 mtu {MTU} up 10.1.0.1 netmask 255.255.255.0")
    h2.cmd(f"ip route del 10.1.0.0/24 dev tun1")
    h2.cmd(f"ip route add 10.1.0.0/24 dev h2-eth0")
    # sleep(0.1)
    
def runIperf(net, duration, payload_size, log_gap, rate, srv_log_path, cli_log_path):
    h1, h2 = net.get('h1', 'h2')
    # h2.cmd("./scripts/tcpdump_start.sh h2-eth0 h2-eth0.pcap")
    h2.cmd(f"iperf -s --bind 10.1.0.2 -p 1235"
                   f"-t {duration + 1} -u -l {payload_size} -i {log_gap} > {srv_log_path} &")
    h1.cmd(f"iperf -c 10.1.0.2 -p 1235 -u -l {payload_size} -b {rate}m "
            f"-t {duration} -i {log_gap} > {cli_log_path} &")
    # for i in np.arange(1, duration + 0.01, 0.01):
    #     writeStats(net.switches[0], "s1-eth2", 'switch_queue.txt')
    # sleep(duration + 1)
    # h2.cmd("./scripts/tcpdump_stop.sh h2-eth0")
    
def iperfTest(net: Mininet, args) -> None:
    loss = args.loss
    fec_algo = str(args.fec)
    fec_batch_size = str(args.fec_batch_size)
    rate = str(args.rate)
    """Launch iperf tcp flow and collect trace with tcpdump."""
    print("Starting iperf test")
    TEST_TIME = 10
    h1, h2 = net.get('h1', 'h2')
    pidsToKill = []
    CLIENT_PATH = "../inopsy-agent/examples/tunnelin"
    CLIENT_LOG_PATH = f"quic_clt_log1.txt"
    SERVER_PATH = "../inopsy-agent/examples/tunnelout"
    SERVER_LOG_PATH = f"quic_srv_log2.txt"

    SRV_LOG_PATH = f"iperf_srv_log2.txt"
    CLI_LOG_PATH = f"iperf_cli_log1.txt"

    schedule = sched.scheduler()

    
    # Log switch queue lenght
    
    # net.switches[0].cmd(f"> switch_queue.txt")
    
    schedule.enter( 0, 
                    HIGHEST_PRIO, setupTunnel,
                    argument=(net, SERVER_PATH, CLIENT_PATH, server_port, fec_algo, fec_batch_size, fec_payload_size, fec_leo_nrecovery, SERVER_LOG_PATH, CLIENT_LOG_PATH, loss))
    # setupTunnel(net, SERVER_PATH, CLIENT_PATH, server_port, fec_algo, fec_batch_size, fec_payload_size, fec_leo_nrecovery, SERVER_LOG_PATH, CLIENT_LOG_PATH)

    schedule.enter( 0, MIDDLE_PRIO, runIperf, 
                        argument=(net, duration, PAYLOAD_SIZE, LOG_GAP, rate, SRV_LOG_PATH, CLI_LOG_PATH) )
    
    for i in np.arange(0.0, duration + QUEUE_TIME_SCALE, QUEUE_TIME_SCALE):
        schedule.enter( i, LOW_PRIO, writeStats, 
                        argument=(net.switches[0], "s1-eth2", 'switch_queue.txt') )
        schedule.enter( i, LOW_PRIO, writeStats, 
                        argument=(net.hosts[0], None, 'host1_queue.txt') )
        schedule.enter( i, LOW_PRIO, writeStats, 
                        argument=(net.hosts[1], None, 'host2_queue.txt') )
    schedule.enter( duration + 1, HIGHEST_PRIO, lambda *args: None )

    net.hosts[0].cmd(f"> host1_queue.txt")
    net.hosts[1].cmd(f"> host2_queue.txt")
    net.switches[0].cmd(f"> switch_queue.txt")

    schedule.run()
    sdwanThread = CustomThread( target=schedule.run )
    sdwanThread.start()
    return sdwanThread


def runExp(args) -> None:
    net = createNetwork()
    net.start()
    net.get("s1").cmd("ovs-vsctl set-fail-mode s1 standalone")

    # Perform test
    # if args.test:
    myThread = iperfTest( net, args )
    
    # Enter CLI mode
    if args.cli:
        CLI( net )
    try:
        myThread.join()         # wait for sdwan traffic to stop before exiting experiment
        pass
    finally:
        net.stop()


def parseArgs():
    parser = argparse.ArgumentParser(description="Runs experiment",
                                     usage="sudo ${PATH_TO_PYTHON_WITH_MININET_PACKAGE} ")
    parser.add_argument("--cli", action="store_true", 
                        help="enter mininet-CLI mode after starting network('exit' to leave)")
    parser.add_argument("--test", action="store_true", 
                        help="perform iperf test after network creation") 
    parser.add_argument("--loss", action="store", 
                        help="loss in channel")
    parser.add_argument("--fec", action="store", 
                        help="fec algorithm")                    
    parser.add_argument("--fec_batch_size", action="store", 
                        help="batch size")    
    parser.add_argument("--rate", action="store", 
                        help="rate")    
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    setLogLevel('info')
    args = parseArgs()
    runExp(args)