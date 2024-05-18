from prometheus_client import CollectorRegistry, Gauge, Counter, push_to_gateway, delete_from_gateway
registry = CollectorRegistry()
pkt_flow = Gauge("Pkt_Flow", "Pkt_Flow", ['Pkt_Flow'], registry=registry)
pkt_type_s = Gauge("Pkt_Type", "Type", ['Type'], registry=registry)
c = Gauge("DDoS", "DDoS", ['DDoS'], registry=registry)
c.clear()
delete_from_gateway('http://172.20.10.4:9091', job='XDP')
push_to_gateway('http://172.20.10.4:9091', job='xdp_data', registry=registry)
