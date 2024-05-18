import time
from prometheus_client import CollectorRegistry, Gauge, Counter, push_to_gateway
import random
registry = CollectorRegistry()
pkt_flow = Gauge("Pkt_Flow", "Pkt_Flow", ['Pkt_Flow'], registry=registry)
pkt_type_s = Gauge("Pkt_Type", "Type", ['Type'], registry=registry)
c = Gauge("DDoS", "DDoS", ['DDoS'], registry=registry)
pkt_flow.clear()
pkt_type_s.clear()
c.clear()


def int_to_ip_little_endian(ip_integer):
    # 将小端序的无符号整数转换为点分十进制形式的IP地址
    if not 0 <= ip_integer <= 0xFFFFFFFF:
        raise ValueError("无效的IP地址整数")

    ip_str = ".".join(str((ip_integer >> (8 * i)) & 255) for i in range(4))
    return ip_str

def ip_to_int_little_endian(ip_address):
    # 将点分十进制形式的IP地址转换为小端序的无符号整数
    ip_parts = list(map(int, ip_address.split(".")))
    if len(ip_parts) != 4 or not all(0 <= part <= 255 for part in ip_parts):
        raise ValueError("无效的IP地址格式")

    ip_integer = sum((ip_parts[i] << (8 * i)) for i in range(4))
    return ip_integer



def update(pkt, tcp, udp, flow, ddos):
    pkt_flow.labels("pkt").set(pkt)
    pkt_flow.labels("flow").set(flow)
    pkt_type_s.labels("tcp").set(tcp)
    pkt_type_s.labels("udp").set(udp)
    for k, v in ddos.items():
        c.labels(ip_to_int_little_endian(k)).set(v)
    try:
        push_to_gateway('http://172.20.10.4:9091', job='XDP', registry=registry)
        print("Metrics pushed successfully.")
    except Exception as e:
        print(f"Error pushing metrics: {e}")


if __name__ == "__main__":
    registry = CollectorRegistry()
    g = Gauge("statistic", "network statistic info", ['xdp_pps_type'], registry=registry)
    c = Counter("ddos", "DDoS attack", ['xdp_ddos_ip'], registry=registry)
    while True:
        pkt_cnt = random.randint(1, 100)
        tcp_cnt = random.randint(1, 100)
        udp_cnt = random.randint(1, 100)
        flow_cnt = random.randint(1, 100)

        print("pkt_cnt:", pkt_cnt)
        print("tcp_cnt:", tcp_cnt)
        print("udp_cnt:", udp_cnt)
        print("flow_cnt:", flow_cnt)

        g.labels("pkt").set(pkt_cnt)
        g.labels("tcp").set(tcp_cnt)
        g.labels("udp").set(udp_cnt)
        g.labels("flow").set(flow_cnt)

        ddos = {
            "192.168.0.1": random.randint(1, 10),
            "192.168.0.2": random.randint(1, 10)
        }
        print("ddos:", ddos)
        for k, v in ddos.items():
            c.labels(k).inc(v)
        try:
            push_to_gateway('http://172.20.10.4:9091', job='xdp_data', registry=registry)
            print("Metrics pushed successfully.")
        except Exception as e:
            print(f"Error pushing metrics: {e}")
        time.sleep(5)
