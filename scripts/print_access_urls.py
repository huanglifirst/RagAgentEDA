#!/usr/bin/env python
from __future__ import annotations

import argparse
import ipaddress
import socket


def _list_ipv4_addresses() -> list[str]:
    ips: set[str] = set()
    hostname = socket.gethostname()

    try:
        _, _, host_ips = socket.gethostbyname_ex(hostname)
        for ip in host_ips:
            if "." in ip:
                ips.add(ip)
    except OSError:
        pass

    try:
        for info in socket.getaddrinfo(hostname, None):
            ip = info[4][0]
            if "." in ip:
                ips.add(ip)
    except OSError:
        pass

    filtered: list[str] = []
    for ip in sorted(ips):
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            continue
        if addr.version != 4:
            continue
        if addr.is_loopback:
            continue
        if addr.is_private:
            filtered.append(ip)
    return filtered


def main() -> int:
    parser = argparse.ArgumentParser(description="Print RagAgentEDA LAN URLs")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    port = args.port
    print("[INFO] Local URLs")
    print(f"  UI:   http://127.0.0.1:{port}/ragagent")
    print(f"  Docs: http://127.0.0.1:{port}/docs")
    print(f"  API:  http://127.0.0.1:{port}/health")
    print()

    lan_ips = _list_ipv4_addresses()
    if not lan_ips:
        print("[WARN] No private IPv4 addresses detected.")
        return 0

    print("[INFO] LAN URLs (share these with colleagues)")
    for ip in lan_ips:
        print(f"  UI:   http://{ip}:{port}/ragagent")
        print(f"  Docs: http://{ip}:{port}/docs")
        print(f"  API:  http://{ip}:{port}/health")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
