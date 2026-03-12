import tkinter as tk
import threading
import sys


class TrashScreenApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("800x600")
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="black")
        self.root.bind("<Escape>", lambda e: sys.exit(0))

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.root.bind("<Configure>", self.on_resize)

        self.show_waiting()

        self.root.update_idletasks()
        self.start_input_thread()

        self.root.mainloop()

    def on_resize(self, event):
        self.draw_current_state()

    def draw_current_state(self):
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        if w <= 1 or h <= 1:
            return
        self._draw_state(w, h)

    def _draw_state(self, w, h):
        self.canvas.delete("all")

        self.canvas.create_text(
            w // 2,
            h // 2,
            text=self.current_text,
            fill=self.current_color,
            font=("Arial", 64, "bold"),
        )

        arrow_size = 60
        padding = 60

        if self.current_state == "recyclable":
            x_tip = padding
            y_tip = h - padding
            self.canvas.create_polygon(
                x_tip,
                y_tip,
                x_tip,
                y_tip - arrow_size,
                x_tip + arrow_size,
                y_tip,
                fill="#00FF00",
                outline="#00FF00",
            )
        elif self.current_state == "not_recyclable":
            x_tip = w - padding
            y_tip = h - padding
            self.canvas.create_polygon(
                x_tip,
                y_tip,
                x_tip,
                y_tip - arrow_size,
                x_tip - arrow_size,
                y_tip,
                fill="#FF0000",
                outline="#FF0000",
            )

    def show_waiting(self):
        self.current_state = "waiting"
        self.current_text = "Waiting for voice command"
        self.current_color = "white"
        self.draw_current_state()

    def show_recyclable(self):
        self.current_state = "recyclable"
        self.current_text = "Recyclable"
        self.current_color = "#00FF00"
        self.draw_current_state()

    def show_not_recyclable(self):
        self.current_state = "not_recyclable"
        self.current_text = "Not recyclable"
        self.current_color = "#FF0000"
        self.draw_current_state()

    def handle_input(self, line):
        line = line.strip()
        if line == "1":
            self.root.after(0, self.show_recyclable)
        elif line == "2":
            self.root.after(0, self.show_not_recyclable)
        elif line == "0":
            self.root.after(0, self.show_waiting)

    def start_input_thread(self):
        thread = threading.Thread(target=self.read_stdin, daemon=True)
        thread.start()

    def read_stdin(self):
        for line in sys.stdin:
            self.handle_input(line)


def main():
    TrashScreenApp()


if __name__ == "__main__":
    main()
