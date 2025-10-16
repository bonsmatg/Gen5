def calculate_reward(metrics):
    latency = metrics.get('latency', 100)
    throughput = metrics.get('throughput', 0)
    packet_loss = metrics.get('packet_loss', 0)
    utilization = metrics.get('link_utilization', 50)
    reward = max(0, 100-latency)*0.4 + throughput*0.3 + max(0,100-packet_loss)*0.2 + max(0,100-abs(utilization-50))*0.1
    return reward