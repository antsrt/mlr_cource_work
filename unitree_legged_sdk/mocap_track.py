#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mocap_track.py — приём OptiTrack Motive (NatNet) + печать поз и XY-график.

Сеть:
- Linux (этот скрипт): 192.168.1.1/24 на eno1
- Windows (Motive):    192.168.1.2/24; Motive → Data Streaming:
  * Local Interface = 192.168.1.2
  * Data Port = 1511, Command Port = 1510
  * Multicast: 239.255.42.99  (или Unicast на 192.168.1.1)

Запуск (как у тебя):
  Multicast:
    python3 mocap_track.py \
      --mode multicast \
      --interface-ip 192.168.1.1 \
      --mcast-addr 239.255.42.99 \
      --data-port 1511 \
      --print-poses \
      --plot

  Unicast:
    python3 mocap_track.py \
      --mode unicast \
      --bind 0.0.0.0 \
      --data-port 1511 \
      --print-poses \
      --plot
"""

# --- безопасный backend для matplotlib (убираем libGL ошибки) ---
try:
    import matplotlib
    try:
        matplotlib.use("TkAgg")   # интерактив без OpenGL
    except Exception:
        matplotlib.use("Agg")     # фолбэк: без окна, но не падает
except Exception:
    pass

import argparse
import socket
import struct
import sys
import threading
import time
import os
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt

# NatNet message IDs
MSG_MODELDEF     = 0
MSG_SERVERINFO   = 5
MSG_FRAMEOFDATA  = 7

DEFAULT_MCAST_ADDR = "239.255.42.99"
DEFAULT_DATA_PORT  = 1511
PRINT_EVERY_SEC    = 1.0
MAX_PACKETSIZE     = 100000


# -------------------- СЕТЕВАЯ ЧАСТЬ --------------------

def _inet4(addr: str) -> bytes:
    return socket.inet_aton(addr)

class UdpRx:
    def __init__(self, mode: str, bind_addr: str, data_port: int,
                 interface_ip: Optional[str], mcast_addr: str, recv_buf: int = 512*1024):
        assert mode in ("multicast", "unicast")
        self.mode = mode
        self.bind_addr = bind_addr
        self.data_port = data_port
        self.interface_ip = interface_ip
        self.mcast_addr = mcast_addr
        self.recv_buf   = recv_buf
        self.sock: Optional[socket.socket] = None

    def open(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        try: s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except OSError: pass
        try: s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except OSError: pass

        try:
            s.bind((self.bind_addr, self.data_port))
        except Exception as e:
            print(f"[ERR] bind({self.bind_addr}:{self.data_port}) failed: {e}", file=sys.stderr)
            sys.exit(1)

        try: s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.recv_buf)
        except OSError: pass

        if self.mode == "multicast":
            group = _inet4(self.mcast_addr)
            if self.interface_ip:
                iface = _inet4(self.interface_ip)
                try:
                    s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, group + iface)
                    print(f"[OK] Joined multicast {self.mcast_addr} on interface {self.interface_ip}")
                except Exception as e:
                    print(f"[WARN] IP_ADD_MEMBERSHIP failed: {e}; trying INADDR_ANY…")
                    s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                                 struct.pack("4sl", group, socket.INADDR_ANY))
                    print(f"[OK] Joined multicast {self.mcast_addr} on INADDR_ANY")
            else:
                s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                             struct.pack("4sl", group, socket.INADDR_ANY))
                print(f"[OK] Joined multicast {self.mcast_addr} on INADDR_ANY")
        else:
            print("[OK] Unicast mode: no multicast join")

        s.setblocking(False)
        self.sock = s

    def recv(self) -> Tuple[bytes, Tuple[str, int]]:
        return self.sock.recvfrom(MAX_PACKETSIZE)

    def close(self):
        try:
            if self.sock: self.sock.close()
        finally:
            self.sock = None


# -------------------- КЛИЕНТ + ПАРСЕР (как в оригинале, но безопасно) --------------------

class NatNetClient:
    def __init__(self, mode="multicast", bind="0.0.0.0", data_port=DEFAULT_DATA_PORT,
                 interface_ip=None, mcast_addr=DEFAULT_MCAST_ADDR,
                 print_poses=False, plot=False, rb_filter=None):
        self.rx = UdpRx(mode, bind, data_port, interface_ip, mcast_addr)
        self.thread: Optional[threading.Thread] = None
        self.running = False

        self.print_poses = print_poses
        self.plot_enabled = plot
        self.rb_filter = set(rb_filter or [])
        self.xy_positions: List[Tuple[float, float]] = []

        # статистика
        self.pkts_total = 0
        self.bytes_total = 0
        self.pkts_last = 0
        self.bytes_last = 0
        self.t_last = time.time()

        # для лайв-графика
        self._last_plot_update = 0.0
        self._plot_update_period = 0.1  # 10 Гц

        if self.plot_enabled:
            self.fig = plt.figure("Mocap XY", figsize=(7,6))
            self.ax = self.fig.add_subplot(1,1,1)
            self.ax.set_title("Rigid Body XY Trajectory")
            self.ax.set_xlabel("X, m")
            self.ax.set_ylabel("Y, m")
            self.ax.grid(True)
            (self.line,) = self.ax.plot([], [], lw=1)

    # ----------- БЕЗОПАСНЫЕ ЧТЕНИЯ -----------

    @staticmethod
    def _read(fmt: str, buf: bytes, off: int):
        sz = struct.calcsize(fmt)
        if off + sz > len(buf):
            raise struct.error("out of range")
        vals = struct.unpack_from(fmt, buf, off)
        return (vals if len(vals) > 1 else vals[0], off + sz)

    @staticmethod
    def _read_cstr(buf: bytes, off: int):
        end = buf.find(b"\x00", off)
        if end == -1:
            raise struct.error("no zero terminator")
        s = buf[off:end]
        return s, end + 1

    # ----------- Парсер FrameOfData (минимальный, как в оригинале) -----------

    def parse_frame_of_data(self, payload: bytes):
        off = 0

        def read(fmt: str):
            nonlocal off
            val, off = self._read(fmt, payload, off)
            return val

        def read_cstr():
            nonlocal off
            s, off = self._read_cstr(payload, off)
            return s

        # ---- frame number
        frame_number = read("<i")
        print(f"Frame number: {frame_number}")

        # ---- marker set count
        marker_set_count = read("<i")
        print(f"Marker set count: {marker_set_count}")

        # ---- skip marker sets: name(zstr) + count + 3*float*count
        for _ in range(int(marker_set_count)):
            _name = read_cstr()
            mcnt = read("<i")
            skip = 12 * int(mcnt)
            if off + skip > len(payload):
                raise struct.error("out of range (marker positions)")
            off += skip

        # ---- unidentified markers
        umcnt = read("<i")
        skip = 12 * int(umcnt)
        if off + skip > len(payload):
            raise struct.error("out of range (unidentified markers)")
        off += skip

        # ---- rigid bodies
        rb_count = read("<i")
        print(f"Rigid body count: {rb_count}")

        for i in range(int(rb_count)):
            body_id = read("<i")
            px, py, pz = read("<fff")
            qx, qy, qz, qw = read("<ffff")

            if (not self.rb_filter) or (body_id in self.rb_filter):
                if self.print_poses:
                    print(f"  RigidBody {i}: id={body_id}, pos=({px:.3f}, {py:.3f}, {pz:.3f}), "
                          f"ori=({qx:.3f}, {qy:.3f}, {qz:.3f}, {qw:.3f})")
                # Копим точки для XY-графика
                self.xy_positions.append((float(px), float(py)))

            # ---- skip RB markers
            mcnt = read("<i")
            skip = 12 * int(mcnt)
            if off + skip > len(payload):  # positions
                break
            off += skip
            if mcnt > 0:
                skip = 4 * int(mcnt)  # IDs
                if off + skip > len(payload):
                    break
                off += skip
                skip = 4 * int(mcnt)  # sizes
                if off + skip > len(payload):
                    break
                off += skip

            # ---- optional mean error
            if off + 4 <= len(payload):
                _ = read("<f")
            # ---- optional params
            if off + 2 <= len(payload):
                _ = read("<h")

        # Остальные блоки кадра нам не нужны.

    # ---------------- Основной приём ----------------

    def start(self):
        self.rx.open()
        print(f"[RUN] mode={self.rx.mode} bind={self.rx.bind_addr}:{self.rx.data_port} "
              f"mcast={self.rx.mcast_addr if self.rx.mode=='multicast' else '-'} iface={self.rx.interface_ip or '-'}")
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.rx.close()

    def _loop(self):
        last_print = time.time()
        while self.running:
            try:
                data, _addr = self.rx.recv()
                if len(data) < 4:
                    continue
                msg_id, nbytes = struct.unpack_from('<HH', data, 0)
                # payload идёт сразу после 4 байт (msg_id + size)
                payload = data[4:]

                if msg_id == MSG_FRAMEOFDATA:
                    try:
                        self.parse_frame_of_data(payload)
                    except struct.error:
                        # мягко пропускаем неполные кадры
                        pass
                elif msg_id == MSG_SERVERINFO:
                    print("ServerInfo packet received")
                elif msg_id == MSG_MODELDEF:
                    print("ModelDef packet received")
                else:
                    # другие типы не обрабатываем
                    pass

                # статистика
                self.pkts_total += 1
                self.bytes_total += len(data)
                self.pkts_last  += 1
                self.bytes_last += len(data)

            except BlockingIOError:
                pass
            except Exception:
                # не валим цикл из-за разовых ошибок приёма
                pass

            # периодический принт
            now = time.time()
            if now - last_print >= PRINT_EVERY_SEC:
                dt = now - self.t_last
                pps = self.pkts_last / dt if dt > 0 else 0.0
                bps = self.bytes_last / dt if dt > 0 else 0.0
                print(f"[STATS] {pps:8.1f} pkts/s | {bps:10.1f} B/s | total={self.pkts_total} pkts")
                self.pkts_last = 0
                self.bytes_last = 0
                self.t_last = now
                last_print = now

            time.sleep(0.001)  # отпускаем CPU

    # ---------------- Лайв- график и сохранение ----------------

    def live_update_plot(self):
        """Обновляет график не чаще 10 Гц."""
        if not self.plot_enabled:
            return
        now = time.time()
        if now - self._last_plot_update < self._plot_update_period:
            return
        self._last_plot_update = now
        if not self.xy_positions:
            return
        xs, ys = zip(*self.xy_positions)
        self.line.set_data(xs, ys)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.001)

    def save_plot_and_txt(self):
        if not self.xy_positions:
            print("No data to plot.")
            return

        # автоинкремент имён
        n = 1
        while os.path.exists(f"trajectory_{n}.png") or os.path.exists(f"trajectory_{n}.txt"):
            n += 1
        image_filename = f"trajectory_{n}.png"
        data_filename  = f"trajectory_{n}.txt"

        xs, ys = zip(*self.xy_positions)

        # если окно уже есть — просто сохранim текущую фигуру
        if self.plot_enabled:
            self.fig.savefig(image_filename)
        else:
            # побстроим разок и сохраним
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(1,1,1)
            ax.plot(xs, ys, marker='o', linestyle='-', label='Trajectory (x, y)')
            ax.set_xlabel('X, m')
            ax.set_ylabel('Y, m')
            ax.set_title('Rigid Body XY Trajectory')
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            fig.savefig(image_filename)

        with open(data_filename, 'w') as f:
            for x, y in self.xy_positions:
                f.write(f"{x:.6f} {y:.6f}\n")

        print(f"Plot saved to {image_filename}")
        print(f"Data saved to {data_filename}")


# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(description="NatNet receiver (original-style printing & XY plotting).")
    ap.add_argument("--mode", choices=["multicast", "unicast"], required=True)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--data-port", type=int, default=DEFAULT_DATA_PORT)
    ap.add_argument("--mcast-addr", default=DEFAULT_MCAST_ADDR)
    ap.add_argument("--interface-ip", default=None)
    ap.add_argument("--print-poses", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--rb", type=int, nargs="*", default=[],
                    help="Если указать, печатаем/копим XY только для этих rigid body id.")
    args = ap.parse_args()

    client = NatNetClient(mode=args.mode, bind=args.bind, data_port=args.data_port,
                          interface_ip=args.interface_ip, mcast_addr=args.mcast_addr,
                          print_poses=args.print_poses, plot=args.plot, rb_filter=args.rb)
    client.start()
    try:
        if args.plot:
            # главный поток крутит лёгкий «рендер-цикл» графика
            while True:
                time.sleep(0.05)
                client.live_update_plot()
        else:
            # без графика — просто ждём Ctrl+C
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting…")
    finally:
        client.stop()
        # сохраняем траекторию (и PNG, и TXT)
        client.save_plot_and_txt()
        # если был интерактивный график — не держим окно
        if args.plot:
            try:
                plt.close('all')
            except Exception:
                pass


if __name__ == "__main__":
    main()
