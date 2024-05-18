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

# 示例
# given_integer = 33558956
# corresponding_ip = int_to_ip_little_endian(given_integer)
#
# print(f"整数 {given_integer} 对应的IP地址是: {corresponding_ip}")

# 反向验证，将IP地址转换为整数
print("172.17.0.1", ":", ip_to_int_little_endian("172.17.0.1"))
print("172.17.0.2", ":", ip_to_int_little_endian("172.17.0.2"))
print("172.17.0.3", ":",ip_to_int_little_endian("172.17.0.3"))
print("172.17.0.4", ":",ip_to_int_little_endian("172.17.0.4"))
print("172.17.0.5", ":",ip_to_int_little_endian("172.17.0.5"))
print(int_to_ip_little_endian(67769516))
print(int_to_ip_little_endian(50992300))
# print(f"IP地址 {corresponding_ip} 对应的整数是: {converted_integer}")
