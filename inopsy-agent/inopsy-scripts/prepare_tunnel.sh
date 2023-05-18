# Script to prepare network namespaces and interfaces to run tunnel.
# sudo required

set -e

ip netns add client
ip netns add server

ip link add dev v1 type veth peer name v2
ip link set v1 netns client
ip link set v2 netns server

ip netns exec client ifconfig v1 up 10.0.1.1 netmask 255.255.255.0
ip netns exec server ifconfig v2 up 10.0.1.2 netmask 255.255.255.0

ip netns exec client ip link set lo up
ip netns exec server ip link set lo up

ip netns exec server route add -net 10.0.2.0/24 dev v2

# sudo ip netns exec server ./examples/quicout 10.0.1.2 1230 ../../cert/localhost.key ../../cert/localhost.crt -q fec_dummy 8 1024
# sudo ip netns exec client ./examples/quicin 10.0.1.2 1230 fec_dummy 8 1024

# sudo ip netns exec server ifconfig tun1 mtu 1400 up 10.0.2.2 netmask 255.255.255.0
# sudo ip netns exec client ifconfig tun0 mtu 1400 up 10.0.2.1 netmask 255.255.255.0
# sudo ip netns exec server route del -net 10.0.2.0/24 dev tun1

# sudo ip netns exec server iperf -s --bind 10.0.2.2 -p 1235
# sudo ip netns exec client iperf -c 10.0.2.2 -p 1235
