import tkinter as tk
from tkinter import Button, Canvas, Label
import math
import random
import numpy as np

class ACO_TSP_GUI:
    def __init__(self, master):
        self.master = master
        master.title("TSP Solver - ACO with RBF Selection")
        master.configure(bg="black")

        # Info label
        self.info_label = Label(master, text="Click to add towns, then press Solve", bg="black", fg="white", font=("Arial", 12))
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

        # ACO parameters
        self.num_ants = 100
        self.num_iterations = 500
        self.alpha = 2.5  # Pheromone importance
        self.beta = 2.75   # Distance importance
        self.evaporation = 0.1  # Pheromone evaporation rate
        self.q = 100  # Pheromone deposit factor
        self.rbf_sigma = 65.0  # RBF kernel width (Gaussian sigma)
        self.pheromone = []
        self.iteration = 0

        # Bind click
        self.canvas.bind("<Button-1>", self.add_point)
        self.redraw()

    def add_point(self, event):
        if not self.solving:
            self.points.append((event.x, event.y))
            self.best_path = []
            self.best_distance = float('inf')
            self.redraw()

    def redraw(self, current_paths=None, current_best=None):
        self.canvas.delete("all")

        if self.iteration == self.num_iterations:
            if self.best_path:
                for i in range(len(self.best_path)):
                    x1, y1 = self.points[self.best_path[i]]
                    x2, y2 = self.points[self.best_path[(i+1)%len(self.best_path)]]
                    self.canvas.create_line(x1, y1, x2, y2, fill="#00ff00", width=3)
            for x, y in self.points:
                self.canvas.create_oval(x-8, y-8, x+8, y+8, fill="white", outline="white")
            if current_best is not None:
                info_text = f"Iteration: {self.iteration}/{self.num_iterations} | Ants: {self.num_ants} | Best: {self.best_distance:.2f} km"
                self.info_label.config(text=info_text)
            return
        
        # Draw pheromone trails (thickness indicates strength)
        if self.pheromone:
            n = len(self.points)
            max_pheromone = max(max(row) for row in self.pheromone) + 1e-10
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        pheromone_val = self.pheromone[i][j]
                        intensity = min(1.0, pheromone_val / max_pheromone)
                        
                        if intensity > 0.05:  # Only draw significant trails
                            x1, y1 = self.points[i]
                            x2, y2 = self.points[j]
                            
                            # Color intensity: brighter grey = stronger pheromone
                            color_val = int(50 + 150 * intensity)
                            color = f"#{int(50*intensity):02x}{int(50*intensity):02x}{int(50*intensity):02x}"
                            width = max(1, int(7 * intensity))
                            
                            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)
        
        # Draw current best paths of this iteration (light gray)
        if current_paths:
            for path in current_paths[:3]:  # Show top 3 paths
                if len(path) > 1:
                    for i in range(len(path)):
                        x1, y1 = self.points[path[i]]
                        x2, y2 = self.points[path[(i+1)%len(path)]]
                        self.canvas.create_line(x1, y1, x2, y2, fill="#444444", width=1, dash=(2, 2))
        
        # Draw global best path (bright green)
        if self.best_path:
            for i in range(len(self.best_path)):
                x1, y1 = self.points[self.best_path[i]]
                x2, y2 = self.points[self.best_path[(i+1)%len(self.best_path)]]
                self.canvas.create_line(x1, y1, x2, y2, fill="#ffffff", width=3)
        
        # Draw towns (white dots)
        for x, y in self.points:
            self.canvas.create_oval(x-8, y-8, x+8, y+8, fill="white", outline="white")
        
        # Update info label
        if current_best is not None:
            info_text = f"Iteration: {self.iteration}/{self.num_iterations} | Ants: {self.num_ants} | Best: {self.best_distance:.2f} km | Selection: RBF"
            self.info_label.config(text=info_text)

    def calc_distance(self, path):
        """Calculate total distance of a path"""
        n = len(path)
        total = 0
        for i in range(n):
            x1, y1 = self.points[path[i]]
            x2, y2 = self.points[path[(i+1)%n]]
            total += math.hypot(x2-x1, y2-y1)
        return total

    def initialize_pheromone(self):
        """Initialize pheromone matrix"""
        n = len(self.points)
        self.pheromone = [[0.1 for _ in range(n)] for _ in range(n)]

    def rbf_kernel(self, value, center=0.0, sigma=1.0):
        """Gaussian RBF Kernel: exp(-(value - center)^2 / (2 * sigma^2))"""
        return math.exp(-((value - center) ** 2) / (2 * (sigma ** 2)))

    def construct_solution(self, ant_id):
        """Construct a single ant's tour using RBF selection"""
        n = len(self.points)
        unvisited = set(range(n))
        
        # Start at random city
        current = random.randint(0, n-1)
        tour = [current]
        unvisited.remove(current)
        
        # Build rest of tour using RBF selection
        while unvisited:
            next_city = self.select_next_city_rbf(current, unvisited)
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        
        return tour

    def select_next_city_rbf(self, current, unvisited):
        """Select next city using RBF kernel-based selection"""
        n = len(self.points)
        
        # Compute RBF scores for each unvisited city
        rbf_scores = []
        
        # Get statistics for normalization
        distances_to_unvisited = []
        pheromones_to_unvisited = []
        
        for city in unvisited:
            x1, y1 = self.points[current]
            x2, y2 = self.points[city]
            distance = math.hypot(x2-x1, y2-y1)
            pheromone_val = self.pheromone[current][city]
            distances_to_unvisited.append(distance)
            pheromones_to_unvisited.append(pheromone_val)
        
        # Normalize distances and pheromones
        max_distance = max(distances_to_unvisited) + 1e-10
        max_pheromone = max(pheromones_to_unvisited) + 1e-10
        
        normalized_distances = [d / max_distance for d in distances_to_unvisited]
        normalized_pheromones = [p / max_pheromone for p in pheromones_to_unvisited]
        
        # Compute RBF kernels
        for i, city in enumerate(unvisited):
            # RBF based on distance (lower distance = higher score, so center at 0)
            distance_rbf = self.rbf_kernel(normalized_distances[i], center=0.0, sigma=self.rbf_sigma/100.0)
            
            # RBF based on pheromone (higher pheromone = higher score, so center at 1)
            pheromone_rbf = self.rbf_kernel(normalized_pheromones[i], center=1.0, sigma=self.rbf_sigma/100.0)
            
            # Combined RBF score: both distance and pheromone contribute
            combined_score = (distance_rbf ** self.beta) * (pheromone_rbf ** self.alpha)
            rbf_scores.append(combined_score)
        
        # Roulette wheel selection based on RBF scores
        total_score = sum(rbf_scores)
        if total_score == 0:
            return random.choice(list(unvisited))
        
        probabilities = [score / total_score for score in rbf_scores]
        r = random.random()
        cumsum = 0
        unvisited_list = list(unvisited)
        
        for i, prob in enumerate(probabilities):
            cumsum += prob
            if r <= cumsum:
                return unvisited_list[i]
        
        return unvisited_list[-1]

    def deposit_pheromone(self, tours, distances):
        """Deposit pheromone based on tour quality"""
        n = len(self.points)
        
        # Evaporate pheromone
        for i in range(n):
            for j in range(n):
                self.pheromone[i][j] *= (1 - self.evaporation)
        
        # Deposit pheromone for each tour
        for tour, distance in zip(tours, distances):
            pheromone_deposit = self.q / distance
            for i in range(len(tour)):
                from_city = tour[i]
                to_city = tour[(i+1) % len(tour)]
                self.pheromone[from_city][to_city] += pheromone_deposit
                self.pheromone[to_city][from_city] += pheromone_deposit

    def start_solve(self):
        n = len(self.points)
        if n < 2:
            self.info_label.config(text="Need at least 2 towns!")
            return
        
        if self.solving:
            return
        
        self.solving = True
        self.solve_btn.config(state="disabled")
        self.best_distance = float('inf')
        self.best_path = []
        self.iteration = 0
        
        # Initialize pheromone
        self.initialize_pheromone()
        
        # Start ACO iterations
        self.run_iteration()
    
    def run_iteration(self):
        if self.iteration >= self.num_iterations:
            # Finished
            self.solving = False
            self.solve_btn.config(state="normal")
            self.info_label.config(text=f"Solved! Best distance: {self.best_distance:.2f} km | Iterations: {self.iteration} | Method: RBF Selection")
            self.redraw()
            return
        
        # Construct solutions for all ants using RBF selection
        tours = []
        distances = []
        
        for ant_id in range(self.num_ants):
            tour = self.construct_solution(ant_id)
            distance = self.calc_distance(tour)
            tours.append(tour)
            distances.append(distance)
            
            # Update global best
            if distance < self.best_distance:
                self.best_distance = distance
                self.best_path = tour[:]
        
        # Sort tours by distance
        sorted_pairs = sorted(zip(tours, distances), key=lambda x: x[1])
        sorted_tours = [t[0] for t in sorted_pairs]
        sorted_distances = [t[1] for t in sorted_pairs]
        
        # Deposit pheromone
        self.deposit_pheromone(sorted_tours, sorted_distances)
        
        # Draw current state
        self.redraw(sorted_tours[:5], True)
        
        self.iteration += 1
        
        # Schedule next iteration (500ms delay)
        self.master.after(5, self.run_iteration)

if __name__ == "__main__":
    root = tk.Tk()
    gui = ACO_TSP_GUI(root)
    root.mainloop()
