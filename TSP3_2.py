import tkinter as tk
from tkinter import Button, Canvas, Label, OptionMenu, StringVar
import math
import random
import numpy as np

class ACO_TSP_GUI:
    def __init__(self, master):
        self.master = master
        master.title("TSP Solver - ACO with Dynamic RBF Sigma")
        master.configure(bg="black")

        # Info label
        self.info_label = Label(master, text="Click to add towns, then press Solve", 
                                bg="black", fg="white", font=("Arial", 12))
        self.info_label.pack(pady=5)

        # Canvas setup
        self.canvas = Canvas(master, width=900, height=600, bg="black", highlightthickness=0)
        self.canvas.pack()

        # Control panel frame
        control_frame = tk.Frame(master, bg="black")
        control_frame.pack(pady=10)

        # Sigma decay strategy selector
        tk.Label(control_frame, text="Sigma Strategy:", bg="black", fg="white").pack(side=tk.LEFT, padx=5)
        self.sigma_strategy = StringVar(value="exponential")
        strategies = [("Linear Decay", "linear"), 
                     ("Exponential Decay", "exponential"), 
                     ("Step-wise Decay", "stepwise"),
                     ("Adaptive (Performance-based)", "adaptive"),
                     ("Constant (No Change)", "constant")]
        
        for text, value in strategies:
            tk.Radiobutton(control_frame, text=text, variable=self.sigma_strategy, 
                          value=value, bg="black", fg="white", selectcolor="gray").pack(side=tk.LEFT, padx=2)

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
        self.num_ants = 50
        self.num_iterations = 500
        self.alpha = 2.25
        self.beta = 2.75
        self.evaporation = 0.15
        self.q = 100
        
        # Dynamic sigma parameters
        self.rbf_sigma_initial = 75.0
        self.rbf_sigma_final = 5.0
        self.rbf_sigma = self.rbf_sigma_initial
        self.sigma_step_interval = (self.rbf_sigma_initial-self.rbf_sigma_final)//10  # Change every N iterations
        
        self.pheromone = []
        self.iteration = 0
        self.prev_best_distance = float('inf')

        # Bind click
        self.canvas.bind("<Button-1>", self.add_point)
        self.redraw()

    def add_point(self, event):
        if not self.solving:
            self.points.append((event.x, event.y))
            self.best_path = []
            self.best_distance = float('inf')
            self.prev_best_distance = float('inf')
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
        
        # Draw pheromone trails
        if self.pheromone:
            n = len(self.points)
            max_pheromone = max(max(row) for row in self.pheromone) + 1e-10
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        pheromone_val = self.pheromone[i][j]
                        intensity = min(1.0, pheromone_val / max_pheromone)
                        
                        if intensity > 0.05:
                            x1, y1 = self.points[i]
                            x2, y2 = self.points[j]
                            
                            color_val = int(50 + 150 * intensity)
                            color = f"#{int(75*intensity):02x}{int(75*intensity):02x}{int(75*intensity):02x}"
                            width = max(1, int(5 * intensity))
                            
                            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)
        
        # Draw current best paths
        if current_paths:
            for path in current_paths[:3]:
                if len(path) > 1:
                    for i in range(len(path)):
                        x1, y1 = self.points[path[i]]
                        x2, y2 = self.points[path[(i+1)%len(path)]]
                        self.canvas.create_line(x1, y1, x2, y2, fill="#444444", width=1, dash=(2, 2))
        
        # Draw global best path
        if self.best_path:
            for i in range(len(self.best_path)):
                x1, y1 = self.points[self.best_path[i]]
                x2, y2 = self.points[self.best_path[(i+1)%len(self.best_path)]]
                self.canvas.create_line(x1, y1, x2, y2, fill="#00ff00", width=5)
        
        # Draw towns
        for x, y in self.points:
            self.canvas.create_oval(x-8, y-8, x+8, y+8, fill="white", outline="white")
        
        # Update info label
        if current_best is not None:
            strategy_name = self.sigma_strategy.get()
            info_text = f"Iter: {self.iteration}/{self.num_iterations} | Best: {self.best_distance:.2f} km | Sigma: {self.rbf_sigma:.2f} | Strategy: {strategy_name} | Num_Ants: {self.num_ants}"
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

    def update_sigma(self):
        """Update RBF sigma based on selected strategy"""
        strategy = self.sigma_strategy.get()
        progress = self.iteration / self.num_iterations
        
        if strategy == "linear":
            # Linear decay from initial to final
            self.rbf_sigma = self.rbf_sigma_initial - (self.rbf_sigma_initial - self.rbf_sigma_final) * progress
        
        elif strategy == "exponential":
            # Exponential decay
            decay_rate = math.log(self.rbf_sigma_final / self.rbf_sigma_initial) / self.num_iterations
            self.rbf_sigma = self.rbf_sigma_initial * math.exp(decay_rate * self.iteration)
        
        elif strategy == "stepwise":
            # Change every N iterations
            step = self.iteration // self.sigma_step_interval
            total_steps = self.num_iterations // self.sigma_step_interval + 1
            if total_steps > 1:
                self.rbf_sigma = self.rbf_sigma_initial - (self.rbf_sigma_initial - self.rbf_sigma_final) * (step / total_steps)
            else:
                self.rbf_sigma = self.rbf_sigma_final
        
        elif strategy == "adaptive":
            # Adapt based on improvement
            improvement = self.prev_best_distance - self.best_distance
            improvement_rate = improvement / (self.prev_best_distance + 1e-10)
            
            if improvement_rate > 0.01:  # Good improvement
                # Increase sigma (more exploration)
                self.rbf_sigma = min(self.rbf_sigma_initial, self.rbf_sigma * 1.05)
            else:  # Little improvement
                # Decrease sigma (more exploitation)
                self.rbf_sigma = max(self.rbf_sigma_final, self.rbf_sigma * 0.95)
        
        elif strategy == "constant":
            # No change
            self.rbf_sigma = self.rbf_sigma_initial
        
        # Clamp sigma to valid range
        self.rbf_sigma = max(self.rbf_sigma_final, min(self.rbf_sigma_initial, self.rbf_sigma))

    def rbf_kernel(self, value, center=random.uniform(-1, 1), sigma=1.0):
        """Gaussian RBF Kernel"""
        return math.exp(-((value - center) ** 2) / (2 * (sigma ** 2)))

    def construct_solution(self, ant_id):
        """Construct a single ant's tour using RBF selection with dynamic sigma"""
        n = len(self.points)
        unvisited = set(range(n))
        
        current = random.randint(0, n-1)
        tour = [current]
        unvisited.remove(current)
        
        while unvisited:
            next_city = self.select_next_city_rbf(current, unvisited)
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        
        return tour

    def select_next_city_rbf(self, current, unvisited):
        """Select next city using dynamic RBF kernel-based selection"""
        n = len(self.points)
        
        rbf_scores = []
        distances_to_unvisited = []
        pheromones_to_unvisited = []
        
        for city in unvisited:
            x1, y1 = self.points[current]
            x2, y2 = self.points[city]
            distance = math.hypot(x2-x1, y2-y1)
            pheromone_val = self.pheromone[current][city]
            distances_to_unvisited.append(distance)
            pheromones_to_unvisited.append(pheromone_val)
        
        max_distance = max(distances_to_unvisited) + 1e-10
        max_pheromone = max(pheromones_to_unvisited) + 1e-10
        
        normalized_distances = [d / max_distance for d in distances_to_unvisited]
        normalized_pheromones = [p / max_pheromone for p in pheromones_to_unvisited]
        
        # Use current dynamic sigma value
        for i, city in enumerate(unvisited):
            # RBF based on distance
            distance_rbf = self.rbf_kernel(normalized_distances[i], center=0.0, sigma=self.rbf_sigma/100.0)
            
            # RBF based on pheromone
            pheromone_rbf = self.rbf_kernel(normalized_pheromones[i], center=1.0, sigma=self.rbf_sigma/100.0)
            
            # Combined score
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
        
        # Deposit pheromone
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
        self.prev_best_distance = float('inf')
        self.iteration = 0
        self.rbf_sigma = self.rbf_sigma_initial
        
        self.initialize_pheromone()
        
        self.run_iteration()
    
    def run_iteration(self):
        if self.iteration >= self.num_iterations:
            self.solving = False
            self.solve_btn.config(state="normal")
            strategy_name = self.sigma_strategy.get()
            self.info_label.config(text=f"Solved! Best: {self.best_distance:.2f} km | Iterations: {self.iteration} | Strategy: {strategy_name} | Final Sigma: {self.rbf_sigma:.2f} | Num_Ants: {self.num_ants}")
            self.redraw()
            return
        
        # Store previous best for adaptive strategy
        self.prev_best_distance = self.best_distance
        
        # Update sigma dynamically
        self.update_sigma()
        
        # Construct solutions
        tours = []
        distances = []
        
        for ant_id in range(self.num_ants):
            tour = self.construct_solution(ant_id)
            distance = self.calc_distance(tour)
            tours.append(tour)
            distances.append(distance)
            
            if distance < self.best_distance:
                self.best_distance = distance
                self.best_path = tour[:]
        
        # Sort tours
        sorted_pairs = sorted(zip(tours, distances), key=lambda x: x[1])
        sorted_tours = [t[0] for t in sorted_pairs]
        sorted_distances = [t[1] for t in sorted_pairs]
        
        # Deposit pheromone
        self.deposit_pheromone(sorted_tours, sorted_distances)
        
        # Draw
        self.redraw(sorted_tours[:5], True)
        
        self.iteration += 1

        self.num_ants += 2
        
        self.master.after(1, self.run_iteration)

if __name__ == "__main__":
    root = tk.Tk()
    gui = ACO_TSP_GUI(root)
    root.mainloop()
