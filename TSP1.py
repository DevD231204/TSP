import tkinter as tk
from tkinter import Button, Canvas, Label
import itertools
import math

class TSPGUI:
    def __init__(self, master):
        self.master = master
        master.title("TSP Town Solver - Animated")
        master.configure(bg="black")

        # Info label
        self.info_label = Label(master, text="Click to add towns, then press Solve", 
                                bg="black", fg="white", font=("Arial", 12))
        self.info_label.pack(pady=5)

        # Canvas setup
        self.canvas = Canvas(master, width=900, height=600, bg="black", highlightthickness=0)
        self.canvas.pack()

        # Solve button
        self.solve_btn = Button(master, text="Solve", command=self.start_solve, 
                                font=("Arial", 14), bg="#444", fg="white")
        self.solve_btn.pack(pady=10)

        # Data storage
        self.points = []
        self.best_path = []
        self.best_distance = float('inf')
        self.solving = False

        # Bind click
        self.canvas.bind("<Button-1>", self.add_point)
        self.redraw()

    def add_point(self, event):
        if not self.solving:
            self.points.append((event.x, event.y))
            self.best_path = []
            self.redraw()

    def redraw(self, current_path=None, current_dist=None, tested=None):
        self.canvas.delete("all")
        
        # Draw current path being tested (light gray)
        if current_path and len(current_path) > 1:
            for i in range(len(current_path)):
                x1, y1 = self.points[current_path[i]]
                x2, y2 = self.points[current_path[(i+1)%len(current_path)]]
                self.canvas.create_line(x1, y1, x2, y2, fill="#666666", width=2)
        
        # Draw best path found so far (green)
        if self.best_path:
            for i in range(len(self.best_path)):
                x1, y1 = self.points[self.best_path[i]]
                x2, y2 = self.points[self.best_path[(i+1)%len(self.best_path)]]
                self.canvas.create_line(x1, y1, x2, y2, fill="#00ff00", width=3)
        
        # Draw points
        for x, y in self.points:
            self.canvas.create_oval(x-8, y-8, x+8, y+8, fill="white", outline="white")
        
        # Update info label
        if tested is not None and current_dist is not None:
            info_text = f"Testing: {tested} | Current: {current_dist:.2f} km | Best: {self.best_distance:.2f} km"
            self.info_label.config(text=info_text)
    
    def calc_distance(self, path):
        n = len(path)
        total = 0
        for i in range(n):
            x1, y1 = self.points[path[i]]
            x2, y2 = self.points[path[(i+1)%n]]
            total += math.hypot(x2-x1, y2-y1)
        return total

    def start_solve(self):
        n = len(self.points)
        if n < 2:
            self.info_label.config(text="Need at least 2 towns!")
            return
        
        if self.solving:
            return
            
        self.solving = True
        self.solve_btn.config(state="disabled")
        self.best_path = []
        self.best_distance = float('inf')
        
        # Generate all permutations
        all_idxs = list(range(n))
        self.perms = list(itertools.permutations(all_idxs))
        self.perm_idx = 0
        self.total_perms = len(self.perms)
        
        # Start animation
        self.animate_next()
    
    def animate_next(self):
        if self.perm_idx >= self.total_perms:
            self.solving = False
            self.solve_btn.config(state="normal")
            self.info_label.config(text=f"Solved! Best distance: {self.best_distance:.2f} km | Searched: {self.total_perms} paths")
            self.redraw()
            return
        
        # Get current permutation
        current_path = list(self.perms[self.perm_idx])
        current_dist = self.calc_distance(current_path)
        
        # Update best if better
        if current_dist < self.best_distance:
            self.best_distance = current_dist
            self.best_path = current_path
        
        # Draw current state
        self.redraw(current_path, current_dist, f"{self.perm_idx+1}/{self.total_perms}")
        
        self.perm_idx += 1
        
        # Schedule next iteration (500ms delay)
        self.master.after(5, self.animate_next)

if __name__ == "__main__":
    root = tk.Tk()
    gui = TSPGUI(root)
    root.mainloop()
