import tkinter as tk

from visualizer.controller import MCTSController

if __name__ == '__main__':
    root = tk.Tk()
    app = MCTSController(root)
    root.mainloop()
