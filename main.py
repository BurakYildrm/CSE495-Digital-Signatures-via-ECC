import hashlib
import math
import random
import secrets
import subprocess
import threading
import tkinter as tk
from operator import itemgetter
from tkinter import filedialog, ttk

import numpy as np
from sympy import isprime, nextprime, sqrt_mod


class App:
    def __init__(self):
        # Create the main window
        self.window = tk.Tk()
        self.window.resizable(False, False)
        self.window.title("Elliptic Curve Digital Signatures")
        self.window.bind(
            "<Button-1>",
            lambda event: self.window.focus_set()
            if event.widget == self.window
            else None,
        )

        # Center the window
        width = 750
        height = 600
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width / 2) - (width / 2)
        y = (screen_height / 2) - (height / 2)
        self.window.geometry("%dx%d+%d+%d" % (width, height, x, y))

        # Set the style of the tabs
        style = ttk.Style()
        style.configure("TNotebook.Tab", padding=[30, 7.5, 30, 5])

        # Initialize the variables
        self.domain_parameters = None
        self.key_pair = None

        # Create the notebook widget
        self.notebook = ttk.Notebook(self.window)
        self.notebook.bind(
            "<Button-1>",
            lambda event: self.notebook.focus_set()
            if event.widget == self.notebook
            else None,
        )

        # Create the tabs
        self.sign_tab = SignTab(self)
        self.verify_tab = VerifyTab(self)

        # Add the tabs to the notebook
        self.notebook.add(self.sign_tab.tab, text="Sign")
        self.notebook.add(self.verify_tab.tab, text="Verify")
        self.notebook.pack(expand=True, fill="both")

        self.window.bind(
            "<<NotebookTabChanged>>", lambda event: self.window.update_idletasks()
        )


class Input:
    def __init__(self, parent):
        # Initialize the variables
        self.parent = parent
        self.file_data = ""
        self.file_path = ""

        # Set the styles of the tabs
        style = ttk.Style()
        style.configure("Input.TNotebook.Tab", padding=[12, 3, 12, 2])

        # Create the notebook widget
        self.notebook = ttk.Notebook(parent.tab, style="Input.TNotebook")
        self.notebook.bind(
            "<Button-1>",
            lambda event: self.notebook.focus_set()
            if event.widget == self.notebook
            else None,
        )

        # Create the tabs
        self.message_tab = ttk.Frame(self.notebook, name="message")
        self.document_tab = ttk.Frame(self.notebook, name="document")

        # Add the tabs to the notebook
        self.notebook.add(self.message_tab, text="Message")
        self.notebook.add(self.document_tab, text="Document")
        self.notebook.pack(expand=True, fill="both")

        self.message_tab.bind(
            "<Button-1>",
            lambda event: self.message_tab.focus_set()
            if event.widget == self.message_tab
            else None,
        )

        # Add widgets to the message tab
        ttk.Label(self.message_tab, text="Your message").pack(
            pady=(20, 5), padx=20, anchor="nw"
        )
        self.message_entry = ttk.Entry(self.message_tab, width=60)
        self.message_entry.pack(pady=5, padx=20, anchor="nw")

        # Add widgets to the document tab
        ttk.Label(self.document_tab, text="Your document").pack(
            pady=5, padx=20, anchor="nw"
        )
        self.file_path_display = ttk.Entry(self.document_tab, width=60)
        self.file_path_display.pack(pady=5, padx=20, anchor="nw")
        self.file_path_display.config(state="disabled", cursor="arrow")
        self.upload_button = ttk.Button(
            self.document_tab,
            text="Upload Document",
            command=self.upload_file,
            cursor="pointinghand",
        )
        self.upload_button.pack(pady=5, padx=20, anchor="nw")

        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

    def on_tab_change(self, event):
        selected_tab = event.widget.select().split(".")[-1]

        if isinstance(self.parent, SignTab):
            self.parent.button.config(
                text="Sign Message" if selected_tab == "message" else "Sign Document"
            )

    def upload_file(self):
        # Disable the upload button and file path display while the file is being read
        self.upload_button.config(state="disabled")
        self.file_path_display.config(state="normal")
        self.file_path_display.delete(0, tk.END)
        self.file_path_display.insert(0, "Uploading file...")
        self.file_path_display.config(state="disabled")

        # Update the window
        self.parent.app.window.update()

        # Open the file dialog and read the file
        file_path = filedialog.askopenfilename(initialdir=".", title="Select a file")
        if file_path:
            with open(file_path, "r") as file:
                data = file.read()
            self.file_path = file_path
            self.file_data = data
            self.file_path_display.config(state="normal")
            self.file_path_display.delete(0, tk.END)
            self.file_path_display.insert(0, file_path)
            self.file_path_display.config(state="disabled")

        # Enable the upload button
        self.upload_button.config(state="normal")

    def get_data(self):
        # Return the message if the message tab is selected
        if self.notebook.select().split(".")[-1] == "message":
            return self.message_entry.get()
        # Return the file data if the document tab is selected
        else:
            return self.file_data

    def get_input_type(self):
        # Return the input type
        return self.notebook.select().split(".")[-1]

    def get_file_path(self):
        # Return the file path
        return self.file_path


class SignTab:
    def __init__(self, app: App):
        # Create the tab
        self.app = app
        self.tab = ttk.Frame(app.notebook)
        self.tab.bind(
            "<Button-1>",
            lambda event: self.tab.focus_set() if event.widget == self.tab else None,
        )

        # Create the input widget
        self.input = Input(self)

        # Add widgets to the tab
        self.checkbox_value = tk.BooleanVar()
        self.checkbox = ttk.Checkbutton(
            self.tab,
            text="Regenerate Curve per Signature",
            variable=self.checkbox_value,
            cursor="pointinghand",
        )
        self.checkbox.pack(pady=(5, 10), padx=20, anchor="nw")
        self.button = ttk.Button(
            self.tab, text="Sign Message", command=self.sign, cursor="pointinghand"
        )
        self.button.pack(pady=(5, 0), padx=20, anchor="nw")
        self.output = tk.Text(self.tab, wrap="word", cursor="arrow")
        self.output.pack(pady=20, padx=20, anchor="nw", expand=True, fill="both")
        self.output.insert(
            "1.0", "Signature and elliptic curve details will be shown here."
        )
        self.output.config(state="disabled")

    def sign(self):
        # Disable the sign button and output while the signature is being generated
        self.button.config(state="disabled")
        self.output.config(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", "Generating signature and elliptic curve details...")
        self.output.config(state="disabled")

        def task():
            # Get the input data
            message = self.input.get_data()
            input_type = self.input.get_input_type()
            file_path = self.input.get_file_path()
            regen_curve = self.checkbox_value.get()

            # Generate the signature and elliptic curve details
            if regen_curve or not self.app.domain_parameters:
                p = select_prime(160)
            else:
                p = itemgetter("p")(self.app.domain_parameters)

            while True:
                if regen_curve or not self.app.domain_parameters:
                    a, b, P = generate_curve(p)
                    n, N = curve_properties(*P, a, b, p)
                else:
                    a, b, P, n, N = itemgetter("a", "b", "P", "n", "N")(
                        self.app.domain_parameters
                    )
                Q, d = key_generation(a, p, P, n)
                r, s = ecdsa_sign(a, p, P, n, d, message)
                if r is None or s is None:
                    continue
                break

            # Store the elliptic curve details
            self.app.domain_parameters = {
                "p": p,
                "a": a,
                "b": b,
                "P": P,
                "n": n,
                "N": N,
            }
            self.app.key_pair = {"Q": Q, "d": d}

            # Display the signature and elliptic curve details
            output = (
                f"{'Message' if input_type == 'message' else 'Document'}: "
                f"{message if input_type == 'message' else file_path}\n\n"
                f"Domain Parameters:\n"
                f"  Prime (p): {p}\n"
                f"  Coefficient (a): {a}\n"
                f"  Coefficient (b): {b}\n"
                f"  Base Point (P): {P}\n"
                f"  Order of Base Point (n): {n}\n"
                f"  Number of Points (N): {N}\n\n"
                f"Key Pair:\n"
                f"  Public Key (Q): {Q}\n"
                f"  Private Key (d): {d}\n\n"
                f"Signature:\n"
                f"  r: {r}\n"
                f"  s: {s}"
            )
            self.output.config(state="normal")
            self.output.delete("1.0", "end")
            self.output.insert("1.0", output)
            self.output.config(state="disabled")
            self.button.config(state="normal")

        # Start the task in a separate thread
        thread = threading.Thread(target=task)
        thread.start()


class VerifyTab:
    def __init__(self, app):
        # Create the tab
        self.app = app
        self.tab = ttk.Frame(app.notebook)
        self.tab.bind(
            "<Button-1>",
            lambda event: self.tab.focus_set() if event.widget == self.tab else None,
        )

        # Create the input widget
        self.input = Input(self)

        # Add widgets to the tab
        ttk.Label(self.tab, text="Signature 'r'").pack(
            pady=(20, 5), padx=20, anchor="nw"
        )
        self.r_entry = ttk.Entry(self.tab, width=60)
        self.r_entry.pack(pady=5, padx=20, anchor="nw")
        ttk.Label(self.tab, text="Signature 's'").pack(
            pady=(20, 5), padx=20, anchor="nw"
        )
        self.s_entry = ttk.Entry(self.tab, width=60)
        self.s_entry.pack(pady=5, padx=20, anchor="nw")
        self.button = ttk.Button(
            self.tab, text="Verify Signature", command=self.verify_signature
        )
        self.button.pack(pady=10, padx=20, anchor="nw")
        self.output = tk.Text(self.tab, wrap="word")
        self.output.pack(pady=20, padx=20, anchor="nw", fill="both", expand=True)
        self.output.insert("1.0", "Verification result will be shown here.")
        self.output.config(state="disabled")

    def verify_signature(self):
        # Disable the verify button and output while the signature is being verified
        self.button.config(state="disabled")
        self.output.config(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", "Verifying signature...")
        self.output.config(state="disabled")

        def task():
            # Get the input data
            message = self.input.get_data()
            r = self.r_entry.get()
            s = self.s_entry.get()

            # Verify the signature
            a, p, P, n = itemgetter("a", "p", "P", "n")(self.app.domain_parameters)
            Q = itemgetter("Q")(self.app.key_pair)
            result = ecdsa_verify(a, p, P, n, Q, message, int(r), int(s))

            # Display the verification result
            output = f"Verification Result: {'Accepted' if result else 'Rejected'}"
            self.output.config(state="normal")
            self.output.delete("1.0", "end")
            self.output.insert("1.0", output)
            self.output.config(state="disabled")
            self.button.config(state="normal")

        # Start the task in a separate thread
        thread = threading.Thread(target=task)
        thread.start()


# greatest common divisor of a and b
def gcd(a, b):
    s0, s1 = 1, 0
    t0, t1 = 0, 1

    while b:
        q = a // b
        s1, s0 = s0 - q * s1, s1
        t1, t0 = t0 - q * t1, t1
        a, b = b, a % b
    return a, s0, t0


# modular inverse of a mod m
def modulo_inverse(a, m):
    (gcd_, s, t) = gcd(a, m)
    return s % m if gcd_ == 1 else 0


# point addition on elliptic curve y^2 = x^3 + ax + b (mod p) with points (x1, y1) and (x2, y2)
def point_addition(x1, y1, x2, y2, a, p):
    if x1 == 0 and y1 == 0:
        return x2, y2
    elif x2 == 0 and y2 == 0:
        return x1, y1
    elif x1 == x2 and y1 == -y2 % p:
        return 0, 0
    elif x1 == x2 and y1 == y2:
        s = (3 * x1**2 + a) * modulo_inverse(2 * y1, p)
        x3 = s**2 - 2 * x1
        y3 = s * (x1 - x3) - y1
    else:
        s = ((y2 - y1) * modulo_inverse(x2 - x1, p)) % p
        x3 = s**2 - x1 - x2
        y3 = s * (x1 - x3) - y1

    return x3 % p, y3 % p


# scalar multiplication on elliptic curve y^2 = x^3 + ax + b (mod p) with point (x, y) and scalar k
def scalar_multiplication(x, y, k, a, p):
    if k == 0:
        return 0, 0
    if k == 1:
        return x, y

    qx, qy = 0, 0
    k = format(k, "b")

    for bit in k:
        qx, qy = point_addition(qx, qy, qx, qy, a, p)
        if int(bit) & 1:
            qx, qy = point_addition(qx, qy, x, y, a, p)

    return qx, qy


# check if point (x, y) is on elliptic curve y^2 = x^3 + ax + b (mod p)
def is_on_curve(x, y, a, b, p):
    return (y**2 - x**3 - a * x - b) % p == 0


# check if elliptic curve y^2 = x^3 + ax + b (mod p) is valid
def is_curve_valid(a, b, p):
    return (4 * a**3 + 27 * b**2) % p != 0


# calculate the order of point (x, y) on elliptic curve y^2 = x^3 + ax + b (mod p) and the number of points on the curve
def curve_properties(x, y, a, b, p):
    try:
        result = subprocess.run(
            [
                "sage",
                "utils.sage",
                str(p),
                str(a),
                str(b),
                "-x",
                str(x),
                "-y",
                str(y),
                "-A",
            ],
            stdout=subprocess.PIPE,
        )
        [order_of_point, number_of_points] = result.stdout.decode("utf-8").split()
        return int(order_of_point), int(number_of_points)
    except:
        return 0


# generate a key pair (Q, d) on elliptic curve y^2 = x^3 + ax + b (mod p) with base point (x, y)
def key_generation(a, p, P, n):
    d = random.randint(1, n - 1)
    qx, qy = scalar_multiplication(P[0], P[1], d, a, p)
    return (qx, qy), d


# generate a signature (r, s) on elliptic curve y^2 = x^3 + ax + b (mod p) with base point (x, y)
# and private key d for message m
def ecdsa_sign(a, p, P, n, d, m):
    r, s = 0, 0
    iteration = 0
    while (r == 0 or s == 0 or modulo_inverse(s, n) == 0) and iteration < 100:
        iteration += 1
        k = random.randint(1, n - 1)
        qx, _ = scalar_multiplication(P[0], P[1], k, a, p)
        r = qx % n
        if r == 0:
            continue
        e = hashlib.sha256(m.encode()).hexdigest()
        e = int(e, 16)
        s = (modulo_inverse(k, n) * (e + d * r)) % n
    return (r, s) if iteration < 100 else (None, None)


# verify a signature (r, s) on elliptic curve y^2 = x^3 + ax + b (mod p) with base point (x, y)
# and public key Q for message m
def ecdsa_verify(a, p, P, n, Q, m, r, s):
    if r < 1 or r > n - 1:
        return False
    if s < 1 or s > n - 1:
        return False
    e = hashlib.sha256(m.encode()).hexdigest()
    e = int(e, 16)
    w = modulo_inverse(s, n)
    u1 = (e * w) % n
    u2 = (r * w) % n
    qx, qy = point_addition(
        *scalar_multiplication(P[0], P[1], u1, a, p),
        *scalar_multiplication(Q[0], Q[1], u2, a, p),
        a,
        p,
    )
    if qx == 0 and qy == 0:
        return False
    return r == qx % n


# make the length of a hex string even
def even_hex(n):
    prefix = ""

    if "0x" in n:
        n = n[2:]
        prefix = "0x"

    h = n

    if len(h) % 2 == 1:
        h = prefix + "0" + h

    return h


# find an integer of length (l - n) bits using the seed s
def find_integer(s, l, n):
    v = math.floor((l - 1) / 256)
    w = l - 256 * v - n
    h = hashlib.sha256(bytearray.fromhex(s)).hexdigest()
    h = np.base_repr(int(h, 16), base=2)
    h0 = h[-w:]
    z = int(s, 16)
    h_array = [h0]

    for i in range(1, v + 1):
        zi = (z + i) % 2**256
        si = bytearray.fromhex(even_hex(np.base_repr(zi, base=16)))
        hi = hashlib.sha256(si).hexdigest()
        hi = np.base_repr(int(hi, 16), base=2)
        h_array.append(hi)

    return int("".join(h_array), 2)


# update the seed s
def update_seed(s):
    z = int(s, 16)
    return even_hex(np.base_repr((z + 1) % 2**256, base=16))


# calculate the legendre symbol of "a" and p
def legendre_symbol(a, p):
    return pow(a, (p - 1) // 2, p)


# check if "a" is a fourth power residue mod p
def is_fourth_power_residue(A, p):
    a = (-3 * modulo_inverse(A, p)) % p
    f = p - 1
    k = f // (4 if f % 4 == 0 else 2)
    return pow(a, k, p) == 1


# generate a prime number of length l bits using the seed s
def select_prime(l, seed=None):
    s = seed if seed else secrets.token_hex(20)

    while True:
        c = find_integer(s, l, 0)

        if isprime(c) and c % 4 == 3:
            p = c
        else:
            i = 1
            while True:
                p = nextprime(c, i)
                if p % 4 == 3:
                    break
                i += 1

        if 2 ** (l - 1) <= p <= 2**l - 1:
            return p

        s = update_seed(s)


# generate an elliptic curve y^2 = x^3 + ax + b (mod p) with prime p using the seed s
def generate_curve(p, seed=None):
    s = seed if seed else secrets.token_hex(20)

    while True:
        a = find_integer(s, p.bit_length(), 1)

        if not is_fourth_power_residue(a, p):
            s = update_seed(s)
            continue

        s = update_seed(s)

        while True:
            b = find_integer(s, p.bit_length(), 1)
            if pow(b, (p - 1) // 2, p) == 1:
                s = update_seed(s)
                continue
            break

        if not is_curve_valid(a, b, p):
            s = update_seed(s)
            continue

        s = update_seed(s)
        k = find_integer(s, p.bit_length(), 1)
        y_values = []
        x = 0

        for i in range(p):
            x = i
            rhs = pow(x, 3) + a * x + b
            if legendre_symbol(rhs, p) == 1:
                y_values = sqrt_mod(rhs, p, all_roots=True)
                break

        points = [(x, y) for y in y_values]
        P = scalar_multiplication(*(random.choice(points)), k, a, p)
        return a, b, P


if __name__ == "__main__":
    ecdsa_app = App()
    ecdsa_app.window.mainloop()
