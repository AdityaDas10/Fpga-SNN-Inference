"""
ide_snn_client.py  —  Local IDE client for Basys3 SNN FPGA
===========================================================

Run this directly in VS Code / PyCharm / Spyder / plain terminal.
Has full COM port access — no browser restrictions.

Features:
  - Interactive menu
  - Send any MNIST test image by index
  - Draw and send digits using a paint window (tkinter)
  - Upload any PNG file
  - Run full accuracy evaluation
  - Auto-generate input.mem for Vivado (no UART needed)

Setup:
  pip install pyserial tensorflow pillow numpy matplotlib

Usage:
  python ide_snn_client.py
  python ide_snn_client.py --port COM3 --index 7
  python ide_snn_client.py --port COM3 --draw
  python ide_snn_client.py --port COM3 --eval
  python ide_snn_client.py --generate --index 42   (no FPGA needed)
"""

import argparse
import sys
import time
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
BAUD_RATE   = 115_200
SYNC_BYTE   = 0xAB
RESULT_BYTE = 0xBB
SCALE       = 64
TIMEOUT_S   = 3.0

# ─────────────────────────────────────────────────────────────────────────────
# Quantisation
# ─────────────────────────────────────────────────────────────────────────────
def quantise(img_float):
    q = np.round(np.array(img_float).reshape(784) * SCALE).astype(np.int8)
    return q.view(np.uint8)

def save_mem(img_float, path="input.mem"):
    q = quantise(img_float)
    with open(path, "w") as f:
        for b in q:
            f.write(f"{int(b):02x}\n")
    print(f"Saved {path}")

# ─────────────────────────────────────────────────────────────────────────────
# UART communication
# ─────────────────────────────────────────────────────────────────────────────
def send_image(port, img_float):
    import serial
    pix = quantise(img_float)
    with serial.Serial(port, BAUD_RATE, timeout=TIMEOUT_S) as ser:
        time.sleep(0.05)
        ser.reset_input_buffer()
        ser.write(bytes([SYNC_BYTE]) + pix.tobytes())
        resp = ser.read(2)
    if len(resp) < 2:
        print("  TIMEOUT — check cable, COM port, and that FPGA is programmed")
        return None
    if resp[0] != RESULT_BYTE:
        print(f"  Unexpected marker: 0x{resp[0]:02X}")
        return None
    return int(resp[1])

def list_ports():
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found.")
    for p in ports:
        print(f"  {p.device:15s}  {p.description}")
    return [p.device for p in ports]

# ─────────────────────────────────────────────────────────────────────────────
# MNIST loader
# ─────────────────────────────────────────────────────────────────────────────
_mnist_cache = None
def load_mnist():
    global _mnist_cache
    if _mnist_cache: return _mnist_cache
    try:
        from tensorflow.keras.datasets import mnist
        (_, _), (x_test, y_test) = mnist.load_data()
        _mnist_cache = (x_test.reshape(-1, 784) / 255.0, y_test)
        return _mnist_cache
    except ImportError:
        print("TensorFlow not found: pip install tensorflow")
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Draw digit with tkinter paint window
# ─────────────────────────────────────────────────────────────────────────────
def draw_digit():
    """Opens a 280x280 paint window. Returns 784-element float array."""
    try:
        import tkinter as tk
        from PIL import Image, ImageDraw, ImageTk
    except ImportError:
        print("Pillow + tkinter required: pip install pillow")
        return None

    CANVAS_SIZE = 280   # 10× scale for drawing, then downsample to 28×28
    BRUSH_SIZE  = 18

    result = [None]

    root  = tk.Tk()
    root.title("Draw a digit (white on black) — press SEND when done")
    pil_img  = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    draw_obj = ImageDraw.Draw(pil_img)

    canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black",
                       cursor="crosshair")
    canvas.pack()

    tk_img = [ImageTk.PhotoImage(pil_img)]

    def paint(event):
        x, y = event.x, event.y
        r = BRUSH_SIZE
        canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        draw_obj.ellipse([x-r, y-r, x+r, y+r], fill=255)

    def clear():
        draw_obj.rectangle([0,0,CANVAS_SIZE,CANVAS_SIZE], fill=0)
        canvas.delete("all")
        canvas.config(bg="black")

    def send():
        small = pil_img.resize((28, 28), Image.LANCZOS)
        arr   = np.array(small, dtype=np.float32) / 255.0
        result[0] = arr.reshape(784)
        root.destroy()

    canvas.bind("<B1-Motion>", paint)
    canvas.bind("<Button-1>", paint)

    btn_frame = tk.Frame(root)
    btn_frame.pack(fill=tk.X)
    tk.Button(btn_frame, text="Clear", command=clear, width=10).pack(side=tk.LEFT,  padx=5, pady=5)
    tk.Button(btn_frame, text="Send →", command=send,  width=10,
              bg="#4CAF50", fg="white").pack(side=tk.RIGHT, padx=5, pady=5)

    root.mainloop()
    return result[0]

# ─────────────────────────────────────────────────────────────────────────────
# Load image from file
# ─────────────────────────────────────────────────────────────────────────────
def load_image_file(path):
    from PIL import Image
    img = Image.open(path).convert("L").resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.mean() > 0.5:
        arr = 1.0 - arr
        print("  (auto-inverted: detected black-on-white image)")
    return arr.reshape(784)

# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────
def show_image(img_flat, title=""):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(3,3))
        plt.imshow(img_flat.reshape(28,28), cmap='gray', vmin=0, vmax=1)
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.5)
    except Exception:
        pass   # matplotlib optional

# ─────────────────────────────────────────────────────────────────────────────
# Interactive menu
# ─────────────────────────────────────────────────────────────────────────────
def interactive_menu():
    print("\n" + "="*55)
    print("  Basys3 SNN FPGA Client")
    print("="*55)

    # Port selection
    print("\nAvailable serial ports:")
    ports = list_ports()
    if not ports:
        print("No ports found. Continuing in file-only mode.")
        port = None
    else:
        port_input = input("\nEnter COM port (or press Enter to skip UART): ").strip()
        port = port_input if port_input else None

    while True:
        print("\n──────────────────────────────────────")
        print("  1. Send MNIST test image by index")
        print("  2. Draw a digit (paint window)")
        print("  3. Send from image file (.png/.jpg)")
        print("  4. Generate input.mem (no UART)")
        print("  5. Run accuracy eval (all 10k images)")
        print("  6. Show sample images")
        print("  0. Exit")
        print("──────────────────────────────────────")
        choice = input("Choice: ").strip()

        if choice == "0":
            break

        elif choice == "1":
            x_test, y_test = load_mnist()
            idx = int(input("MNIST test index (0-9999): ").strip())
            img = x_test[idx]
            true = int(y_test[idx])
            show_image(img, f"Index {idx}  True: {true}")
            if port:
                t0 = time.time()
                pred = send_image(port, img)
                ms   = (time.time()-t0)*1000
                if pred is not None:
                    ok = "CORRECT ✓" if pred==true else f"WRONG ✗"
                    print(f"  Predicted: {pred}  True: {true}  {ok}  ({ms:.0f} ms)")
            else:
                save_mem(img)
                print("Saved input.mem — copy to Vivado project and re-synthesise.")

        elif choice == "2":
            print("Opening paint window...")
            img = draw_digit()
            if img is None:
                continue
            show_image(img, "Your drawing (28×28)")
            if port:
                t0   = time.time()
                pred = send_image(port, img)
                ms   = (time.time()-t0)*1000
                if pred is not None:
                    print(f"  FPGA predicted: {pred}  ({ms:.0f} ms)")
                save_mem(img, "input_drawn.mem")
                print("Also saved as input_drawn.mem")
            else:
                save_mem(img, "input_drawn.mem")
                print("Saved input_drawn.mem — rename to input.mem and re-synthesise.")

        elif choice == "3":
            path = input("Image path: ").strip().strip('"')
            try:
                img = load_image_file(path)
            except Exception as e:
                print(f"Error: {e}"); continue
            show_image(img, f"Preprocessed: {path}")
            if port:
                pred = send_image(port, img)
                if pred is not None:
                    print(f"  FPGA predicted: {pred}")
                save_mem(img, "input_file.mem")
            else:
                save_mem(img, "input_file.mem")
                print("Saved input_file.mem")

        elif choice == "4":
            x_test, y_test = load_mnist()
            idx = int(input("MNIST test index (0-9999): ").strip())
            save_mem(x_test[idx], f"input_{y_test[idx]}_idx{idx}.mem")
            print("Copy this file to Vivado project dir, rename to input.mem,")
            print("then re-run Generate Bitstream.")

        elif choice == "5":
            if not port:
                print("COM port required for eval."); continue
            x_test, y_test = load_mnist()
            n, correct = len(x_test), 0
            print(f"Evaluating {n} images at {BAUD_RATE} baud (~15 min)...")
            t0 = time.time()
            for i in range(n):
                pred = send_image(port, x_test[i])
                if pred is None:
                    print(f"Timeout at {i}"); break
                if pred == int(y_test[i]): correct += 1
                if (i+1) % 500 == 0:
                    elapsed = time.time()-t0
                    print(f"  {i+1:5d}/{n}  acc={correct/(i+1)*100:.2f}%  "
                          f"{(i+1)/elapsed:.1f} img/s  "
                          f"ETA {(n-i-1)/((i+1)/elapsed):.0f}s")
            print(f"\nFinal accuracy: {correct}/{n} = {correct/n*100:.2f}%")

        elif choice == "6":
            x_test, y_test = load_mnist()
            try:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(2, 5, figsize=(12,5))
                for d in range(10):
                    idx = np.where(y_test==d)[0][0]
                    ax  = axes[d//5][d%5]
                    ax.imshow(x_test[idx].reshape(28,28), cmap='gray')
                    ax.set_title(f'Digit {d} (idx {idx})')
                    ax.axis('off')
                plt.suptitle("MNIST test samples — one per digit")
                plt.tight_layout()
                plt.show()
            except Exception:
                print("matplotlib not available for display")

# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Basys3 SNN FPGA client")
    parser.add_argument("--port",     "-p", help="COM port")
    parser.add_argument("--index",    "-i", type=int, help="MNIST test index")
    parser.add_argument("--draw",           action="store_true")
    parser.add_argument("--image",    "-f", help="Image file path")
    parser.add_argument("--eval",           action="store_true")
    parser.add_argument("--generate",       action="store_true",
                        help="Generate input.mem only (no UART)")
    parser.add_argument("--list-ports",     action="store_true")
    args = parser.parse_args()

    if args.list_ports:
        list_ports(); return

    # No arguments → interactive menu
    if len(sys.argv) == 1:
        interactive_menu(); return

    x_test = y_test = None

    if args.generate:
        x_test, y_test = load_mnist()
        idx = args.index or 0
        save_mem(x_test[idx], "input.mem")
        print(f"input.mem generated for index {idx} (digit {y_test[idx]})")
        return

    if args.index is not None:
        x_test, y_test = load_mnist()
        img  = x_test[args.index]
        true = int(y_test[args.index])
        if args.port:
            pred = send_image(args.port, img)
            if pred is not None:
                ok = "CORRECT ✓" if pred==true else f"WRONG ✗ (expected {true})"
                print(f"Digit {true} → predicted {pred}  {ok}")
        else:
            save_mem(img); print(f"Generated input.mem for digit {true}")

    elif args.draw:
        img = draw_digit()
        if img is not None and args.port:
            pred = send_image(args.port, img)
            if pred is not None:
                print(f"FPGA predicted: {pred}")

    elif args.image:
        img = load_image_file(args.image)
        if args.port:
            pred = send_image(args.port, img)
            if pred is not None:
                print(f"FPGA predicted: {pred}")
        else:
            save_mem(img)

    elif args.eval and args.port:
        x_test, y_test = load_mnist()
        n, correct = len(x_test), 0
        t0 = time.time()
        for i in range(n):
            pred = send_image(args.port, x_test[i])
            if pred is None: print(f"Timeout at {i}"); break
            if pred == int(y_test[i]): correct += 1
            if (i+1) % 500 == 0:
                print(f"  {i+1}/{n}  {correct/(i+1)*100:.2f}%")
        print(f"Final: {correct}/{n} = {correct/n*100:.2f}%")

if __name__ == "__main__":
    main()
